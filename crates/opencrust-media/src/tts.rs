use async_trait::async_trait;
use tracing::info;

/// Raw audio bytes returned by a TTS provider.
/// Always OGG/Opus so every voice-capable channel can send it directly.
pub type AudioBytes = Vec<u8>;

/// Abstraction over any text-to-speech backend.
#[async_trait]
pub trait TtsProvider: Send + Sync {
    /// Convert `text` to speech and return raw audio bytes (OGG/Opus).
    async fn synthesize(&self, text: &str) -> Result<AudioBytes, String>;

    /// Short identifier used in log messages.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// OpenAI TTS  (tts-1 / tts-1-hd)
// ---------------------------------------------------------------------------

/// Calls the OpenAI `/v1/audio/speech` endpoint.
pub struct OpenAiTts {
    client: reqwest::Client,
    api_key: String,
    model: String,
    voice: String,
}

impl OpenAiTts {
    pub fn new(api_key: String, model: Option<String>, voice: Option<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model: model.unwrap_or_else(|| "tts-1".to_string()),
            voice: voice.unwrap_or_else(|| "alloy".to_string()),
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
        let resp = self
            .client
            .post("https://api.openai.com/v1/audio/speech")
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
//     base_url: http://localhost:8880
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
            client: reqwest::Client::new(),
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

use std::sync::Arc;

/// Build a `TtsProvider` from config values.
/// Returns `None` if `tts_provider` is not set or unrecognised.
pub fn build_tts_provider(
    tts_provider: Option<&str>,
    api_key: Option<String>,
    model: Option<String>,
    voice: Option<String>,
    _base_url: Option<String>,
) -> Option<Arc<dyn TtsProvider>> {
    match tts_provider? {
        "openai" => {
            let key = api_key?;
            Some(Arc::new(OpenAiTts::new(key, model, voice)))
        }
        #[cfg(feature = "tts-kokoro")]
        "kokoro" => Some(Arc::new(KokoroTts::new(_base_url, voice))),
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
