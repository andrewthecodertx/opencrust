use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tracing::{info, warn};

/// Raw audio bytes returned by a TTS provider.
/// Always OGG/Opus so every voice-capable channel can send it directly.
pub type AudioBytes = Vec<u8>;

/// Default character limit sent to a TTS backend.
/// OpenAI's hard limit is 4096; we default to 4000 to leave a safe margin.
pub const TTS_DEFAULT_MAX_CHARS: usize = 4000;

/// Timeout for TTS HTTP requests (synthesis + body download).
const TTS_HTTP_TIMEOUT: Duration = Duration::from_secs(30);

/// Abstraction over any text-to-speech backend.
#[async_trait]
pub trait TtsProvider: Send + Sync {
    /// Convert `text` to speech and return raw audio bytes (OGG/Opus).
    async fn synthesize(&self, text: &str) -> Result<AudioBytes, String>;

    /// Short identifier used in log messages.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Shared reqwest client (connection-pool reuse)
// ---------------------------------------------------------------------------

fn tts_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(TTS_HTTP_TIMEOUT)
        .build()
        .expect("failed to build TTS HTTP client")
}

// ---------------------------------------------------------------------------
// OpenAI TTS  (tts-1 / tts-1-hd, or any OpenAI-compatible endpoint)
// ---------------------------------------------------------------------------

/// Calls an OpenAI-compatible `/v1/audio/speech` endpoint.
/// Defaults to `https://api.openai.com`; set `base_url` for self-hosted servers.
pub struct OpenAiTts {
    client: reqwest::Client,
    api_key: String,
    model: String,
    voice: String,
    /// Base URL without trailing slash, e.g. `https://api.openai.com`.
    base_url: String,
}

impl OpenAiTts {
    pub fn new(api_key: String, model: Option<String>, voice: Option<String>) -> Self {
        Self::with_base_url(api_key, model, voice, None)
    }

    pub fn with_base_url(
        api_key: String,
        model: Option<String>,
        voice: Option<String>,
        base_url: Option<String>,
    ) -> Self {
        Self {
            client: tts_http_client(),
            api_key,
            model: model.unwrap_or_else(|| "tts-1".to_string()),
            voice: voice.unwrap_or_else(|| "alloy".to_string()),
            base_url: base_url
                .unwrap_or_else(|| "https://api.openai.com".to_string())
                .trim_end_matches('/')
                .to_string(),
        }
    }
}

#[async_trait]
impl TtsProvider for OpenAiTts {
    fn name(&self) -> &'static str {
        "openai"
    }

    async fn synthesize(&self, text: &str) -> Result<AudioBytes, String> {
        info!("openai tts: synthesizing {} chars", text.len());
        let url = format!("{}/v1/audio/speech", self.base_url);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&serde_json::json!({
                "model": self.model,
                "input": text,
                "voice": self.voice,
                "response_format": "opus",
            }))
            .send()
            .await
            .map_err(|e| format!("openai tts request failed: {e}"))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("openai tts error {status}: {body}"));
        }

        resp.bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| format!("openai tts read body failed: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Kokoro TTS (self-hosted via kokoro-fastapi)
//
// Enable with:  cargo build --features tts-kokoro
//
// Expects a running Kokoro FastAPI server (https://github.com/remsky/Kokoro-FastAPI).
// Default base URL: http://localhost:8880
//
// Config example:
//   voice:
//     tts_provider: kokoro
//     tts_base_url: http://localhost:8880
//     voice: af_heart
//     auto_reply_voice: true
// ---------------------------------------------------------------------------

#[cfg(feature = "tts-kokoro")]
pub struct KokoroTts {
    client: reqwest::Client,
    base_url: String,
    voice: String,
}

#[cfg(feature = "tts-kokoro")]
impl KokoroTts {
    pub fn new(base_url: Option<String>, voice: Option<String>) -> Self {
        Self {
            client: tts_http_client(),
            base_url: base_url
                .unwrap_or_else(|| "http://localhost:8880".to_string())
                .trim_end_matches('/')
                .to_string(),
            voice: voice.unwrap_or_else(|| "af_heart".to_string()),
        }
    }
}

#[cfg(feature = "tts-kokoro")]
#[async_trait]
impl TtsProvider for KokoroTts {
    fn name(&self) -> &'static str {
        "kokoro"
    }

    async fn synthesize(&self, text: &str) -> Result<AudioBytes, String> {
        info!("kokoro tts: synthesizing {} chars", text.len());
        let url = format!("{}/v1/audio/speech", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({
                "model": "kokoro",
                "input": text,
                "voice": self.voice,
                "response_format": "opus",
            }))
            .send()
            .await
            .map_err(|e| format!("kokoro tts request failed: {e}"))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("kokoro tts error {status}: {body}"));
        }

        resp.bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| format!("kokoro tts read body failed: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Build a `TtsProvider` from config values.
/// Returns `None` if `tts_provider` is not set or unrecognised.
#[allow(unused_variables)]
pub fn build_tts_provider(
    tts_provider: Option<&str>,
    api_key: Option<String>,
    model: Option<String>,
    voice: Option<String>,
    tts_base_url: Option<String>,
) -> Option<Arc<dyn TtsProvider>> {
    match tts_provider? {
        "openai" => {
            let key = api_key?;
            Some(Arc::new(OpenAiTts::with_base_url(
                key,
                model,
                voice,
                tts_base_url,
            )))
        }
        #[cfg(feature = "tts-kokoro")]
        "kokoro" => Some(Arc::new(KokoroTts::new(tts_base_url, voice))),
        #[cfg(not(feature = "tts-kokoro"))]
        "kokoro" => {
            tracing::warn!(
                "tts_provider 'kokoro' requires the `tts-kokoro` feature flag. \
                 Rebuild with: cargo build --features tts-kokoro"
            );
            None
        }
        other => {
            tracing::warn!("unknown tts_provider '{other}' — ignoring");
            None
        }
    }
}

/// Truncate `text` to `max_chars` Unicode characters before sending to TTS.
/// Logs a warning if truncation occurs.
pub fn truncate_for_tts(text: &str, max_chars: usize) -> &str {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text;
    }
    warn!(
        "tts: response ({} chars) exceeds limit ({}), truncating",
        char_count, max_chars
    );
    // Find byte offset of the max_chars-th char boundary.
    let byte_end = text
        .char_indices()
        .nth(max_chars)
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    &text[..byte_end]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // Fake OGG/Opus header (4 bytes) — just enough for the "non-empty bytes" check.
    const FAKE_AUDIO: &[u8] = b"OggS";

    // -----------------------------------------------------------------------
    // build_tts_provider
    // -----------------------------------------------------------------------

    #[test]
    fn build_tts_provider_none_when_no_provider() {
        assert!(build_tts_provider(None, None, None, None, None).is_none());
    }

    #[test]
    fn build_tts_provider_none_when_openai_missing_key() {
        assert!(build_tts_provider(Some("openai"), None, None, None, None).is_none());
    }

    #[test]
    fn build_tts_provider_openai_returns_provider() {
        let p = build_tts_provider(Some("openai"), Some("sk-test".into()), None, None, None);
        assert!(p.is_some());
        assert_eq!(p.unwrap().name(), "openai");
    }

    #[test]
    fn build_tts_provider_unknown_returns_none() {
        assert!(build_tts_provider(Some("elevenlabs"), None, None, None, None).is_none());
    }

    // -----------------------------------------------------------------------
    // truncate_for_tts
    // -----------------------------------------------------------------------

    #[test]
    fn truncate_for_tts_short_text_unchanged() {
        assert_eq!(truncate_for_tts("hello", 10), "hello");
    }

    #[test]
    fn truncate_for_tts_exact_limit_unchanged() {
        let s = "a".repeat(100);
        assert_eq!(truncate_for_tts(&s, 100), s);
    }

    #[test]
    fn truncate_for_tts_long_text_truncated() {
        let s = "a".repeat(4001);
        let t = truncate_for_tts(&s, 4000);
        assert_eq!(t.chars().count(), 4000);
    }

    #[test]
    fn truncate_for_tts_unicode_boundary() {
        // 3-byte UTF-8 chars (Thai) — ensure we don't split mid-char
        let s = "ก".repeat(10); // 10 Thai chars = 30 bytes
        let t = truncate_for_tts(&s, 5);
        assert_eq!(t.chars().count(), 5);
        assert!(std::str::from_utf8(t.as_bytes()).is_ok());
    }

    // -----------------------------------------------------------------------
    // OpenAiTts
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn openai_tts_success() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .and(header("authorization", "Bearer sk-test"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(FAKE_AUDIO)
                    .insert_header("content-type", "audio/ogg"),
            )
            .mount(&server)
            .await;

        let tts = OpenAiTts::with_base_url("sk-test".into(), None, None, Some(server.uri()));
        let audio = tts.synthesize("hello world").await.unwrap();
        assert_eq!(audio, FAKE_AUDIO);
    }

    #[tokio::test]
    async fn openai_tts_error_response_returns_err() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .respond_with(ResponseTemplate::new(401).set_body_string(r#"{"error":"invalid key"}"#))
            .mount(&server)
            .await;

        let tts = OpenAiTts::with_base_url("bad-key".into(), None, None, Some(server.uri()));
        let err = tts.synthesize("hello").await.unwrap_err();
        assert!(err.contains("401"), "expected 401 in error: {err}");
    }

    #[tokio::test]
    async fn openai_tts_custom_base_url() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(FAKE_AUDIO)
                    .insert_header("content-type", "audio/ogg"),
            )
            .mount(&server)
            .await;

        // Verify tts_base_url routes to a non-openai.com host
        let tts = OpenAiTts::with_base_url("sk-test".into(), None, None, Some(server.uri()));
        let audio = tts.synthesize("test").await.unwrap();
        assert_eq!(audio, FAKE_AUDIO);
    }

    // -----------------------------------------------------------------------
    // KokoroTts (only compiled when feature flag is on)
    // -----------------------------------------------------------------------

    #[cfg(feature = "tts-kokoro")]
    #[tokio::test]
    async fn kokoro_tts_success() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(FAKE_AUDIO)
                    .insert_header("content-type", "audio/ogg"),
            )
            .mount(&server)
            .await;

        let tts = KokoroTts::new(Some(server.uri()), None);
        let audio = tts.synthesize("こんにちは").await.unwrap();
        assert_eq!(audio, FAKE_AUDIO);
    }

    #[cfg(feature = "tts-kokoro")]
    #[tokio::test]
    async fn kokoro_tts_error_response_returns_err() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
            .mount(&server)
            .await;

        let tts = KokoroTts::new(Some(server.uri()), None);
        let err = tts.synthesize("test").await.unwrap_err();
        assert!(err.contains("500"), "expected 500 in error: {err}");
    }

    #[cfg(feature = "tts-kokoro")]
    #[tokio::test]
    async fn build_tts_provider_kokoro_returns_provider() {
        let p = build_tts_provider(Some("kokoro"), None, None, Some("af_sky".into()), None);
        assert!(p.is_some());
        assert_eq!(p.unwrap().name(), "kokoro");
    }
}
