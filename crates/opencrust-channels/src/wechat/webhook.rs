use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{Query, Request, State};
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use ring::digest;
use serde::Deserialize;
use subtle::ConstantTimeEq;
use tracing::{info, warn};

/// WeChat requires a passive-reply within 5 seconds or it retries.
/// We use a 4-second budget to leave headroom for serialization.
const WECHAT_SYNC_TIMEOUT: Duration = Duration::from_secs(4);

use crate::traits::ChannelResponse;

use super::WeChatChannel;
use super::WeChatFile;
use super::api;
use super::fmt;

/// Shared state passed to WeChat webhook handlers.
pub type WeChatWebhookState = Arc<Vec<Arc<WeChatChannel>>>;

#[derive(Debug, Deserialize)]
pub struct WeChatWebhookParams {
    pub signature: Option<String>,
    pub timestamp: Option<String>,
    pub nonce: Option<String>,
    pub echostr: Option<String>,
}

/// Verify WeChat webhook signature.
///
/// WeChat signs requests by sorting [token, timestamp, nonce] lexicographically,
/// concatenating them, and computing SHA-1. Uses a byte-by-byte comparison to
/// avoid leaking timing information.
fn verify_signature(token: &str, timestamp: &str, nonce: &str, signature: &str) -> bool {
    let mut parts = [token, timestamp, nonce];
    parts.sort_unstable();
    let joined = parts.concat();

    let hash = digest::digest(&digest::SHA1_FOR_LEGACY_USE_ONLY, joined.as_bytes());
    let computed = hash
        .as_ref()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();

    computed.as_bytes().ct_eq(signature.as_bytes()).into()
}

/// GET /wechat/webhook — server ownership verification requested by WeChat.
///
/// WeChat sends `signature`, `timestamp`, `nonce`, and `echostr`. If the
/// signature is valid, respond with `echostr` to confirm server ownership.
pub async fn wechat_webhook_verify(
    State(channels): State<WeChatWebhookState>,
    Query(params): Query<WeChatWebhookParams>,
) -> impl IntoResponse {
    let signature = params.signature.as_deref().unwrap_or("");
    let timestamp = params.timestamp.as_deref().unwrap_or("");
    let nonce = params.nonce.as_deref().unwrap_or("");
    let echostr = params.echostr.as_deref().unwrap_or("");

    let valid = channels
        .iter()
        .any(|ch| verify_signature(&ch.token, timestamp, nonce, signature));

    if valid {
        info!("wechat: server verification succeeded");
        (StatusCode::OK, echostr.to_string())
    } else {
        warn!("wechat: server verification failed — invalid signature");
        (StatusCode::UNAUTHORIZED, String::new())
    }
}

/// POST /wechat/webhook — receives message events from the WeChat platform.
///
/// Verifies the signature on the query params, then parses the XML body and
/// dispatches text messages to the configured channel.
pub async fn wechat_webhook(
    State(channels): State<WeChatWebhookState>,
    Query(params): Query<WeChatWebhookParams>,
    req: Request,
) -> impl IntoResponse {
    let signature = params.signature.as_deref().unwrap_or("").to_string();
    let timestamp = params.timestamp.as_deref().unwrap_or("").to_string();
    let nonce = params.nonce.as_deref().unwrap_or("").to_string();

    let channel = match channels
        .iter()
        .find(|ch| verify_signature(&ch.token, &timestamp, &nonce, &signature))
    {
        Some(ch) => Arc::clone(ch),
        None => {
            warn!("wechat: no channel matched signature — request rejected");
            return (
                StatusCode::UNAUTHORIZED,
                [(header::CONTENT_TYPE, "text/xml")],
                String::new(),
            );
        }
    };

    let body: Bytes = match axum::body::to_bytes(req.into_body(), 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                [(header::CONTENT_TYPE, "text/xml")],
                String::new(),
            );
        }
    };

    let xml = match std::str::from_utf8(&body) {
        Ok(s) => s.to_string(),
        Err(_) => {
            warn!("wechat: invalid UTF-8 in webhook body");
            return (
                StatusCode::BAD_REQUEST,
                [(header::CONTENT_TYPE, "text/xml")],
                String::new(),
            );
        }
    };

    let msg_type = fmt::extract_xml_field(&xml, "MsgType").unwrap_or("");

    // Build the content string and optional file download info for on_message.
    // For "image" messages the PicUrl is a public CDN URL (no auth needed);
    // we pass it as (url, filename) and download inside the spawned task.
    let (content, file_dl): (String, Option<(String, String)>) = match msg_type {
        "text" => match fmt::extract_xml_field(&xml, "Content") {
            Some(t) if !t.trim().is_empty() => (t.to_string(), None),
            _ => {
                return (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "text/xml")],
                    String::new(),
                );
            }
        },
        "image" => {
            let pic_url = fmt::extract_xml_field(&xml, "PicUrl")
                .unwrap_or("")
                .to_string();
            let msg_id = fmt::extract_xml_field(&xml, "MsgId")
                .unwrap_or("")
                .to_string();
            (
                String::new(),
                Some((pic_url, format!("image_{msg_id}.jpg"))),
            )
        }
        "voice" => {
            let media_id = fmt::extract_xml_field(&xml, "MediaId").unwrap_or("");
            let format = fmt::extract_xml_field(&xml, "Format").unwrap_or("");
            (
                format!("[voice: media_id={media_id} format={format}]"),
                None,
            )
        }
        "video" | "shortvideo" => {
            let media_id = fmt::extract_xml_field(&xml, "MediaId").unwrap_or("");
            let thumb_id = fmt::extract_xml_field(&xml, "ThumbMediaId").unwrap_or("");
            (
                format!("[video: media_id={media_id} thumb_media_id={thumb_id}]"),
                None,
            )
        }
        "location" => {
            let lat = fmt::extract_xml_field(&xml, "Location_X").unwrap_or("");
            let lon = fmt::extract_xml_field(&xml, "Location_Y").unwrap_or("");
            let label = fmt::extract_xml_field(&xml, "Label").unwrap_or("");
            (
                format!("[location: lat={lat} lon={lon} label={label}]"),
                None,
            )
        }
        // Unsupported event types (follow, scan, etc.) — acknowledge silently.
        _ => {
            return (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/xml")],
                String::new(),
            );
        }
    };

    let from_user = fmt::extract_xml_field(&xml, "FromUserName")
        .unwrap_or("")
        .to_string();
    let to_user = fmt::extract_xml_field(&xml, "ToUserName")
        .unwrap_or("")
        .to_string();
    let msg_id = fmt::extract_xml_field(&xml, "MsgId")
        .unwrap_or("")
        .to_string();

    // WeChat does not have a concept of group chats on the Official Account API.
    let is_group = false;

    if !channel.group_filter()(is_group) {
        return (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/xml")],
            String::new(),
        );
    }

    // Deduplicate: WeChat retries the same MsgId up to 3 times when no
    // response arrives within 5 seconds. Drop already-seen messages.
    if channel.check_and_mark_msg_id(&msg_id).await {
        info!("wechat: duplicate MsgId={msg_id} from openid={from_user}, dropping");
        return (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/xml")],
            String::new(),
        );
    }

    info!(
        "wechat: message from openid={} ({} chars)",
        from_user,
        content.len()
    );

    // Spawn the LLM work as an independent task so its JoinHandle can be
    // reused in the background push if the sync window expires.  This ensures
    // the LLM processes the message exactly once regardless of whether we reply
    // synchronously or asynchronously.
    //
    // For image messages, download the PicUrl inside the task before calling
    // handle_incoming so that the file bytes are available to the callback.
    let ch_llm = Arc::clone(&channel);
    let from_user_llm = from_user.clone();
    let content_llm = content.clone();
    let client_llm = channel.client().clone();
    let mut llm_task = tokio::spawn(async move {
        let file = if let Some((url, filename)) = file_dl {
            match api::download_pic(&client_llm, &url).await {
                Ok(data) => Some(WeChatFile {
                    filename,
                    data,
                    mime_type: Some("image/jpeg".to_string()),
                }),
                Err(e) => {
                    return Err(format!("image download failed: {e}"));
                }
            }
        } else {
            None
        };
        ch_llm
            .handle_incoming(&from_user_llm, &from_user_llm, &content_llm, is_group, file)
            .await
    });

    tokio::select! {
        // LLM finished within the 4-second sync window.
        join_result = &mut llm_task => {
            let result = join_result.unwrap_or_else(|e| Err(format!("llm task panicked: {e}")));
            match result {
                // Replied in time — send passive XML reply.
                // WeChat passive reply only supports text; Voice falls back to text here
                // and the audio is pushed asynchronously via Customer Service API.
                Ok(response) => {
                    if let ChannelResponse::Voice { ref audio, .. } = response {
                        // Passive XML can only carry text — push audio async.
                        let ch2 = Arc::clone(&channel);
                        let openid2 = from_user.clone();
                        let audio2 = audio.clone();
                        tokio::spawn(async move {
                            match ch2.get_cached_token().await {
                                Ok(token) => {
                                    match api::upload_voice(
                                        ch2.client(),
                                        &token,
                                        &audio2,
                                        ch2.api_base_url(),
                                    )
                                    .await
                                    {
                                        Ok(media_id) => {
                                            if let Err(e) = api::push_voice_msg(
                                                ch2.client(),
                                                &token,
                                                &openid2,
                                                &media_id,
                                                ch2.api_base_url(),
                                            )
                                            .await
                                            {
                                                warn!("wechat: async voice push failed: {e}");
                                            }
                                        }
                                        Err(e) => warn!("wechat: voice upload failed: {e}"),
                                    }
                                }
                                Err(e) => warn!("wechat: token fetch failed for voice push: {e}"),
                            }
                        });
                    }
                    let reply_text = fmt::to_wechat_text(response.text());
                    let reply_xml = fmt::build_reply_xml(&from_user, &to_user, &reply_text);
                    info!("wechat: sync reply sent to openid={from_user}");
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "text/xml")],
                        reply_xml,
                    )
                }
                // Silently blocked (e.g. unauthorized user).
                Err(e) if e == "__blocked__" => (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "text/xml")],
                    String::new(),
                ),
                // LLM error — push error message asynchronously and return empty.
                Err(e) => {
                    warn!("wechat: error processing message from openid={from_user}: {e}");
                    let ch = Arc::clone(&channel);
                    let openid = from_user.clone();
                    tokio::spawn(async move {
                        match ch.get_cached_token().await {
                            Ok(token) => {
                                let _ = api::push(
                                    ch.client(),
                                    &token,
                                    &openid,
                                    "Sorry, an error occurred.",
                                    ch.api_base_url(),
                                )
                                .await;
                            }
                            Err(e) => {
                                warn!("wechat: failed to get access token for error push: {e}")
                            }
                        }
                    });
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "text/xml")],
                        String::new(),
                    )
                }
            }
        }
        // Sync window expired — return empty immediately so WeChat does not
        // retry, then await the *same* LLM task and push once it completes.
        // The LLM is never called a second time.
        _ = tokio::time::sleep(WECHAT_SYNC_TIMEOUT) => {
            info!("wechat: LLM timeout for openid={from_user}, spawning async push");
            let ch = Arc::clone(&channel);
            let openid = from_user.clone();
            tokio::spawn(async move {
                let result = match llm_task.await {
                    Ok(r) => r,
                    Err(e) => Err(format!("llm task panicked: {e}")),
                };
                let response = match result {
                    Ok(r) => r,
                    Err(e) => {
                        warn!("wechat: async LLM error for openid={openid}: {e}");
                        ChannelResponse::Text("Sorry, an error occurred.".to_string())
                    }
                };
                match ch.get_cached_token().await {
                    Ok(token) => {
                        // For Voice: upload audio and push as voice message.
                        if let ChannelResponse::Voice { ref audio, .. } = response {
                            match api::upload_voice(ch.client(), &token, audio, ch.api_base_url())
                                .await
                            {
                                Ok(media_id) => {
                                    if let Err(e) = api::push_voice_msg(
                                        ch.client(),
                                        &token,
                                        &openid,
                                        &media_id,
                                        ch.api_base_url(),
                                    )
                                    .await
                                    {
                                        warn!("wechat: async voice push failed: {e}");
                                    } else {
                                        info!("wechat: async voice push sent to openid={openid}");
                                        return;
                                    }
                                }
                                Err(e) => warn!("wechat: voice upload failed, sending text: {e}"),
                            }
                        }
                        // Text path (or Voice fallback after upload failure).
                        let reply = fmt::to_wechat_text(response.text());
                        if let Err(e) =
                            api::push(ch.client(), &token, &openid, &reply, ch.api_base_url())
                                .await
                        {
                            warn!("wechat: async push failed for openid={openid}: {e}");
                        }
                    }
                    Err(e) => warn!("wechat: failed to get access token for async push: {e}"),
                }
            });
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/xml")],
                String::new(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ChannelResponse;
    use crate::wechat::{WeChatChannel, WeChatOnMessageFn};
    use axum::body::Body;
    use axum::{
        Router,
        routing::{get, post},
    };
    use ring::digest;
    use tower::ServiceExt;

    fn sha1_hex(input: &str) -> String {
        let hash = digest::digest(&digest::SHA1_FOR_LEGACY_USE_ONLY, input.as_bytes());
        hash.as_ref().iter().map(|b| format!("{b:02x}")).collect()
    }

    fn sign(token: &str, timestamp: &str, nonce: &str) -> String {
        let mut parts = [token, timestamp, nonce];
        parts.sort_unstable();
        sha1_hex(&parts.concat())
    }

    fn make_state(token: &str) -> WeChatWebhookState {
        let on_msg: WeChatOnMessageFn = Arc::new(|_uid, _ctx, _text, _is_group, _file, _| {
            Box::pin(async { Ok(ChannelResponse::Text("reply".to_string())) })
        });
        let ch = Arc::new(WeChatChannel::new(
            "appid".to_string(),
            "secret".to_string(),
            token.to_string(),
            on_msg,
        ));
        Arc::new(vec![ch])
    }

    fn make_router(state: WeChatWebhookState) -> Router {
        Router::new()
            .route("/wechat/webhook", get(wechat_webhook_verify))
            .route("/wechat/webhook", post(wechat_webhook))
            .with_state(state)
    }

    fn text_message_xml(from_user: &str, to_user: &str, content: &str) -> String {
        text_message_xml_with_id(from_user, to_user, content, "1234567890")
    }

    fn text_message_xml_with_id(
        from_user: &str,
        to_user: &str,
        content: &str,
        msg_id: &str,
    ) -> String {
        format!(
            "<xml>\
                <ToUserName><![CDATA[{to_user}]]></ToUserName>\
                <FromUserName><![CDATA[{from_user}]]></FromUserName>\
                <CreateTime>1234567890</CreateTime>\
                <MsgType><![CDATA[text]]></MsgType>\
                <Content><![CDATA[{content}]]></Content>\
                <MsgId>{msg_id}</MsgId>\
            </xml>"
        )
    }

    #[tokio::test]
    async fn server_verification_valid_signature_returns_echostr() {
        let token = "mytoken";
        let timestamp = "1700000000";
        let nonce = "abc123";
        let sig = sign(token, timestamp, nonce);

        let uri = format!(
            "/wechat/webhook?signature={sig}&timestamp={timestamp}&nonce={nonce}&echostr=hello"
        );
        let resp = make_router(make_state(token))
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        assert_eq!(body, "hello");
    }

    #[tokio::test]
    async fn server_verification_invalid_signature_returns_401() {
        let uri = "/wechat/webhook?signature=badsig&timestamp=1700000000&nonce=abc&echostr=hi";
        let resp = make_router(make_state("mytoken"))
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn post_valid_text_message_returns_xml_reply() {
        let token = "mytoken";
        let timestamp = "1700000000";
        let nonce = "abc123";
        let sig = sign(token, timestamp, nonce);
        let body = text_message_xml("oOpenId123", "gh_account", "hello");

        let uri = format!("/wechat/webhook?signature={sig}&timestamp={timestamp}&nonce={nonce}");
        let resp = make_router(make_state(token))
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "text/xml")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let reply = std::str::from_utf8(&bytes).unwrap();
        assert!(reply.contains("<![CDATA[reply]]>"));
        assert!(reply.contains("<![CDATA[oOpenId123]]>"));
    }

    #[tokio::test]
    async fn post_invalid_signature_returns_401() {
        let body = text_message_xml("oOpenId123", "gh_account", "hello");
        let uri = "/wechat/webhook?signature=badsig&timestamp=1700000000&nonce=abc";

        let resp = make_router(make_state("mytoken"))
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "text/xml")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn post_non_text_event_returns_200_empty() {
        let token = "mytoken";
        let timestamp = "1700000000";
        let nonce = "abc123";
        let sig = sign(token, timestamp, nonce);
        // "follow" is an event type that is not dispatched to on_message.
        let body = "<xml>\
                <ToUserName><![CDATA[gh_account]]></ToUserName>\
                <FromUserName><![CDATA[oOpenId123]]></FromUserName>\
                <CreateTime>1234567890</CreateTime>\
                <MsgType><![CDATA[follow]]></MsgType>\
            </xml>"
            .to_string();

        let uri = format!("/wechat/webhook?signature={sig}&timestamp={timestamp}&nonce={nonce}");
        let resp = make_router(make_state(token))
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "text/xml")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 256).await.unwrap();
        assert!(bytes.is_empty());
    }

    #[tokio::test]
    async fn post_image_message_downloads_pic_and_dispatches_to_on_message() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        // Spin up a mock CDN that serves the PicUrl image.
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/photo.jpg"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(b"\xff\xd8\xff\xe0" as &[u8]) // minimal JPEG header
                    .append_header("content-type", "image/jpeg"),
            )
            .mount(&mock_server)
            .await;

        let pic_url = format!("{}/photo.jpg", mock_server.uri());

        // on_message receives the WeChatFile and returns "reply".
        let on_msg: WeChatOnMessageFn =
            Arc::new(|_uid, _ctx, _text, _is_group, file, _delta_tx| {
                Box::pin(async move {
                    // File should be present with non-empty data.
                    assert!(file.is_some());
                    let f = file.unwrap();
                    assert!(!f.data.is_empty());
                    assert_eq!(f.mime_type.as_deref(), Some("image/jpeg"));
                    Ok(ChannelResponse::Text("reply".to_string()))
                })
            });
        let token = "mytoken";
        let ch = Arc::new(WeChatChannel::new(
            "appid".to_string(),
            "secret".to_string(),
            token.to_string(),
            on_msg,
        ));
        let state: WeChatWebhookState = Arc::new(vec![ch]);

        let timestamp = "1700000000";
        let nonce = "abc123";
        let sig = sign(token, timestamp, nonce);
        let body = format!(
            "<xml>\
                <ToUserName><![CDATA[gh_account]]></ToUserName>\
                <FromUserName><![CDATA[oOpenId123]]></FromUserName>\
                <CreateTime>1234567890</CreateTime>\
                <MsgType><![CDATA[image]]></MsgType>\
                <PicUrl><![CDATA[{pic_url}]]></PicUrl>\
                <MediaId><![CDATA[media_abc]]></MediaId>\
                <MsgId>1234567890</MsgId>\
            </xml>"
        );

        let uri = format!("/wechat/webhook?signature={sig}&timestamp={timestamp}&nonce={nonce}");
        let resp = make_router(state)
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "text/xml")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let reply = std::str::from_utf8(&bytes).unwrap();
        // on_message returns "reply", so the XML should contain it.
        assert!(reply.contains("<![CDATA[reply]]>"));
    }

    #[tokio::test]
    async fn post_image_download_failure_returns_error_response() {
        // PicUrl points to a non-existent server — download should fail.
        let on_msg: WeChatOnMessageFn =
            Arc::new(|_uid, _ctx, _text, _is_group, _file, _delta_tx| {
                Box::pin(async { Ok(ChannelResponse::Text("reply".to_string())) })
            });
        let token = "mytoken";
        let ch = Arc::new(WeChatChannel::new(
            "appid".to_string(),
            "secret".to_string(),
            token.to_string(),
            on_msg,
        ));
        let state: WeChatWebhookState = Arc::new(vec![ch]);

        let timestamp = "1700000000";
        let nonce = "abc123";
        let sig = sign(token, timestamp, nonce);
        // Port 1 is reserved and will be refused immediately.
        let body = "<xml>\
                <ToUserName><![CDATA[gh_account]]></ToUserName>\
                <FromUserName><![CDATA[oOpenId123]]></FromUserName>\
                <CreateTime>1234567890</CreateTime>\
                <MsgType><![CDATA[image]]></MsgType>\
                <PicUrl><![CDATA[http://127.0.0.1:1/bad.jpg]]></PicUrl>\
                <MediaId><![CDATA[media_xyz]]></MediaId>\
                <MsgId>9999999999</MsgId>\
            </xml>"
            .to_string();

        let uri = format!("/wechat/webhook?signature={sig}&timestamp={timestamp}&nonce={nonce}");
        // The webhook should still return 200 (error is handled internally).
        let resp = make_router(state)
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "text/xml")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn post_voice_message_dispatches_to_on_message() {
        let token = "mytoken";
        let timestamp = "1700000000";
        let nonce = "abc123";
        let sig = sign(token, timestamp, nonce);
        let body = "<xml>\
                <ToUserName><![CDATA[gh_account]]></ToUserName>\
                <FromUserName><![CDATA[oOpenId123]]></FromUserName>\
                <CreateTime>1234567890</CreateTime>\
                <MsgType><![CDATA[voice]]></MsgType>\
                <MediaId><![CDATA[voice_media_id]]></MediaId>\
                <Format><![CDATA[amr]]></Format>\
                <MsgId>1234567890</MsgId>\
            </xml>"
            .to_string();

        let uri = format!("/wechat/webhook?signature={sig}&timestamp={timestamp}&nonce={nonce}");
        let resp = make_router(make_state(token))
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "text/xml")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let reply = std::str::from_utf8(&bytes).unwrap();
        assert!(reply.contains("<![CDATA[reply]]>"));
    }

    #[tokio::test]
    async fn post_location_message_dispatches_to_on_message() {
        let token = "mytoken";
        let timestamp = "1700000000";
        let nonce = "abc123";
        let sig = sign(token, timestamp, nonce);
        let body = "<xml>\
                <ToUserName><![CDATA[gh_account]]></ToUserName>\
                <FromUserName><![CDATA[oOpenId123]]></FromUserName>\
                <CreateTime>1234567890</CreateTime>\
                <MsgType><![CDATA[location]]></MsgType>\
                <Location_X>13.7563</Location_X>\
                <Location_Y>100.5018</Location_Y>\
                <Scale>15</Scale>\
                <Label><![CDATA[Bangkok]]></Label>\
                <MsgId>1234567890</MsgId>\
            </xml>"
            .to_string();

        let uri = format!("/wechat/webhook?signature={sig}&timestamp={timestamp}&nonce={nonce}");
        let resp = make_router(make_state(token))
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "text/xml")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let reply = std::str::from_utf8(&bytes).unwrap();
        assert!(reply.contains("<![CDATA[reply]]>"));
    }

    #[tokio::test]
    async fn duplicate_msg_id_returns_empty_response() {
        let token = "mytoken";
        let timestamp = "1700000000";
        let nonce = "abc123";
        let sig = sign(token, timestamp, nonce);
        let state = make_state(token);
        let router = make_router(Arc::clone(&state));

        let body = text_message_xml_with_id("oUser", "gh_account", "hello", "msg_unique_99");
        let uri = format!("/wechat/webhook?signature={sig}&timestamp={timestamp}&nonce={nonce}");

        // First request — should be processed normally.
        let resp1 = router
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(&uri)
                    .header("content-type", "text/xml")
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp1.status(), StatusCode::OK);
        let bytes1 = axum::body::to_bytes(resp1.into_body(), 4096).await.unwrap();
        assert!(!bytes1.is_empty(), "first request should return a reply");

        // Second request with identical MsgId — should be dropped (empty body).
        let resp2 = router
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(&uri)
                    .header("content-type", "text/xml")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp2.status(), StatusCode::OK);
        let bytes2 = axum::body::to_bytes(resp2.into_body(), 256).await.unwrap();
        assert!(
            bytes2.is_empty(),
            "duplicate MsgId should return empty body"
        );
    }

    #[tokio::test]
    async fn timeout_returns_empty_response() {
        let token = "mytoken";
        let timestamp = "1700000000";
        let nonce = "abc123";
        let sig = sign(token, timestamp, nonce);

        // on_message that takes longer than WECHAT_SYNC_TIMEOUT.
        let on_msg: WeChatOnMessageFn = Arc::new(|_uid, _ctx, _text, _is_group, _file, _| {
            Box::pin(async {
                tokio::time::sleep(Duration::from_secs(10)).await;
                Ok(ChannelResponse::Text("late reply".to_string()))
            })
        });
        let ch = Arc::new(WeChatChannel::new(
            "appid".to_string(),
            "secret".to_string(),
            token.to_string(),
            on_msg,
        ));
        let state: WeChatWebhookState = Arc::new(vec![ch]);
        let router = make_router(state);

        let body = text_message_xml_with_id("oUser", "gh_account", "slow", "msg_slow_1");
        let uri = format!("/wechat/webhook?signature={sig}&timestamp={timestamp}&nonce={nonce}");

        let resp = router
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "text/xml")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 256).await.unwrap();
        assert!(
            bytes.is_empty(),
            "timed-out request should return empty body so WeChat does not retry"
        );
    }
}
