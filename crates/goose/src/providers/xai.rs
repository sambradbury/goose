use super::errors::ProviderError;
use super::base::{ConfigKey, Provider, ProviderMetadata, ProviderUsage, Usage, ImageGenerationResult};
use crate::message::{Message, MessageContent};
use crate::model::ModelConfig;
use crate::providers::formats::openai::{create_request, get_usage, response_to_message};
use crate::providers::utils::get_model;
use anyhow::Result;
use async_trait::async_trait;
use mcp_core::Tool;
use mcp_core::content::ImageContent;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tracing;

pub const XAI_API_HOST: &str = "https://api.x.ai/v1";
pub const XAI_DEFAULT_MODEL: &str = "grok-3";
pub const XAI_KNOWN_MODELS: &[&str] = &[
    "grok-3",
    "grok-3-fast",
    "grok-3-mini",
    "grok-3-mini-fast",
    "grok-2-vision-1212",
    "grok-2-image-1212",
    "grok-2-1212",
    "grok-3-latest",
    "grok-3-fast-latest",
    "grok-3-mini-latest",
    "grok-3-mini-fast-latest",
    "grok-2-vision",
    "grok-2-vision-latest",
    "grok-2-image",
    "grok-2-image-latest",
    "grok-2",
    "grok-2-latest",
];

pub const XAI_DOC_URL: &str = "https://docs.x.ai/docs/overview";

#[derive(serde::Serialize)]
pub struct XaiProvider {
    #[serde(skip)]
    client: Client,
    host: String,
    api_key: String,
    model: ModelConfig,
}

impl Default for XaiProvider {
    fn default() -> Self {
        let model = ModelConfig::new(XaiProvider::metadata().default_model);
        XaiProvider::from_env(model).expect("Failed to initialize xAI provider")
    }
}

impl XaiProvider {
    pub fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let api_key: String = config.get_secret("XAI_API_KEY")?;
        let host: String = config
            .get_param("XAI_HOST")
            .unwrap_or_else(|_| XAI_API_HOST.to_string());

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(120)) // 2 minute timeout
            .build()?;

        Ok(Self {
            client,
            host,
            api_key,
            model,
        })
    }

    /// Check if the current model supports image generation
    fn is_image_generation_model(&self) -> bool {
        self.model.model_name.contains("image")
    }

    async fn post(&self, payload: Value) -> anyhow::Result<Value, ProviderError> {
        let url = format!("{}/chat/completions", self.host);

        tracing::debug!(
            url = %url,
            payload = ?payload,
            "Making request to xAI API"
        );

        let response = self.client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&payload)
            .send()
            .await?;

        let status = response.status();
        let response_data = response.json::<Value>().await.ok();

        tracing::debug!(
            status = %status,
            response_data = ?response_data,
            "Received response from xAI API"
        );

        match status {
            StatusCode::OK => {
                response_data.ok_or_else(|| {
                    ProviderError::RequestFailed("No response data received".to_string())
                })
            }
            StatusCode::BAD_REQUEST => {
                tracing::error!("❌ Bad request with status: {}", status);
                tracing::error!("📦 Request payload that caused bad request: {}", serde_json::to_string(&payload).unwrap());

                let error_msg = if let Some(data) = response_data {
                    tracing::error!("📄 Response data: {}", serde_json::to_string(&data).unwrap());
                    let data_owned = data.clone();
                    let error_val = data_owned.get("error");
                    let message_val = match error_val {
                        Some(e) => e.get("message"),
                        None => None,
                    };
                    let msg = match message_val {
                        Some(m) => m.as_str(),
                        None => None,
                    };
                    match msg {
                        Some(s) => s.to_string(),
                        None => "Bad request".to_string(),
                    }
                } else {
                    "Bad request".to_string()
                };

                Err(ProviderError::RequestFailed(format!(
                    "Request failed: {}", error_msg
                )))
            }
            StatusCode::UNAUTHORIZED => {
                Err(ProviderError::Authentication(
                    "Invalid API key".to_string(),
                ))
            }
            StatusCode::NOT_FOUND => {
                tracing::error!(
                    "404 Not Found for URL: {}/chat/completions",
                    self.host
                );
                Err(ProviderError::RequestFailed(
                    "API endpoint not found (404)".to_string(),
                ))
            }
            _ => {
                let error_msg = if let Some(data) = response_data {
                    let data_owned = data.clone();
                    let error_val = data_owned.get("error");
                    let message_val = match error_val {
                        Some(e) => e.get("message"),
                        None => None,
                    };
                    let msg = match message_val {
                        Some(m) => m.as_str(),
                        None => None,
                    };
                    match msg {
                        Some(s) => s.to_string(),
                        None => format!("HTTP {}", status),
                    }
                } else {
                    format!("HTTP {}", status)
                };

                Err(ProviderError::RequestFailed(error_msg))
            }
        }
    }
}

#[async_trait]
impl Provider for XaiProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            "xai",
            "xAI",
            "Grok models from xAI, including reasoning and multimodal capabilities",
            XAI_DEFAULT_MODEL,
            XAI_KNOWN_MODELS.to_vec(),
            XAI_DOC_URL,
            vec![
                ConfigKey::new("XAI_API_KEY", true, true, None),
                ConfigKey::new("XAI_HOST", false, false, Some(XAI_API_HOST)),
            ],
        )
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    #[tracing::instrument(
        skip(self, system, messages, tools),
        fields(model_config, input, output, input_tokens, output_tokens, total_tokens)
    )]
    async fn complete(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> anyhow::Result<(Message, ProviderUsage), ProviderError> {
        tracing::info!("🔍 xAI complete: received {} messages", messages.len());

        // Log the content types of messages received by xAI
        for (i, msg) in messages.iter().enumerate() {
            let content_types: Vec<&str> = msg.content.iter().map(|c| match c {
                MessageContent::Text(_) => "text",
                MessageContent::Image(_) => "image",
                MessageContent::ToolRequest(_) => "tool_request",
                MessageContent::ToolResponse(_) => "tool_response",
                _ => "other",
            }).collect();
            tracing::info!("🔍 xAI complete: message {} content types: {:?}", i, content_types);
        }

        // Check if this is an image generation model being used for text completion
        if self.is_image_generation_model() {
            return Err(ProviderError::ExecutionError(
                format!("Model {} is for image generation, not text completion", self.model.model_name)
            ));
        }

        let payload = create_request(
            &self.model,
            system,
            messages,
            tools,
            &super::utils::ImageFormat::OpenAi,
        )?;

        let response = self.post(payload.clone()).await?;

        let message = response_to_message(response.clone())?;
        let usage = match get_usage(&response) {
            Ok(usage) => usage,
            Err(ProviderError::UsageError(e)) => {
                tracing::debug!("Failed to get usage data: {}", e);
                Usage::default()
            }
            Err(e) => return Err(e),
        };
        let model = get_model(&response);
        super::utils::emit_debug_trace(&self.model, &payload, &response, &usage);
        Ok((message, ProviderUsage::new(model, usage)))
    }

    fn supports_image_generation(&self) -> bool {
        self.is_image_generation_model()
    }

    /// Generate images using xAI's simplified API (only supports prompt)
    async fn generate_images(
        &self,
        prompt: String,
    ) -> Result<ImageGenerationResult, ProviderError> {
        let start_time = std::time::Instant::now();
        tracing::info!(
            "🎨 Starting xAI image generation - model: {}, prompt: '{}'",
            self.model.model_name,
            prompt
        );

        // Check if this is an image generation model
        if !self.is_image_generation_model() {
            return Err(ProviderError::ExecutionError(
                format!("Model {} does not support image generation", self.model.model_name)
            ));
        }

        // Use the correct xAI images endpoint
        let url = format!("{}/images/generations", self.host);
        tracing::info!("🔗 Constructed URL: {}", url);

        // xAI only supports: model, prompt, n, response_format
        let payload = json!({
            "model": self.model.model_name,
            "prompt": prompt,
            "n": 1,
            "response_format": "b64_json"
        });

        tracing::info!("📦 Request payload: {}", serde_json::to_string(&payload).unwrap());
        tracing::info!("🔑 Using API key: {}...", &self.api_key[..8.min(self.api_key.len())]);

        let http_start = std::time::Instant::now();
        tracing::info!("📡 Making HTTP request to xAI API...");

        let response = match self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&payload)
            .send()
            .await {
                Ok(resp) => {
                    tracing::info!("✅ HTTP request succeeded");
                    resp
                }
                Err(e) => {
                    tracing::error!("❌ HTTP request failed: {}", e);
                    return Err(ProviderError::RequestFailed(format!("HTTP request failed: {}", e)));
                }
            };

        let http_duration = http_start.elapsed();
        tracing::info!("📥 HTTP request completed in {:?}, status: {}", http_duration, response.status());

        // Log response headers
        tracing::info!("📋 Response headers:");
        for (name, value) in response.headers() {
            if let Ok(value_str) = value.to_str() {
                tracing::info!("  {}: {}", name, value_str);
            }
        }

        let status = response.status();
        let json_start = std::time::Instant::now();
        tracing::info!("🔄 Parsing response JSON...");

        // First, let's get the raw response text to see what we're actually getting
        let response_text = match response.text().await {
            Ok(text) => {
                tracing::info!("✅ Response text received, length: {}", text.len());
                text
            }
            Err(e) => {
                tracing::error!("❌ Failed to get response text: {}", e);
                return Err(ProviderError::RequestFailed(format!("Failed to get response text: {}", e)));
            }
        };

        tracing::info!("📄 Raw response text (first 500 chars): {}", &response_text[..response_text.len().min(500)]);

        // Now parse it as JSON
        let response_data: Result<Value, _> = serde_json::from_str(&response_text);
        let response_data = match response_data {
            Ok(data) => {
                tracing::info!("✅ JSON parsing successful");
                Some(data)
            }
            Err(e) => {
                tracing::error!("❌ JSON parsing failed: {}", e);
                None
            }
        };

        let json_duration = json_start.elapsed();
        tracing::info!("✅ JSON parsing completed in {:?}", json_duration);

        tracing::info!("📊 Processing response with status: {}", status);

        match status {
            StatusCode::OK => {
                tracing::info!("✅ Status OK, processing response data...");
                let data = response_data.ok_or_else(|| {
                    ProviderError::RequestFailed("Response body is not valid JSON".to_string())
                })?;

                tracing::info!("🔍 Extracting image data from response...");
                // Parse the response to extract image data
                let images = data.get("data")
                    .and_then(|d| d.as_array())
                    .ok_or_else(|| {
                        ProviderError::RequestFailed("Invalid response format: missing data array".to_string())
                    })?;

                tracing::info!("🖼️ Found {} images in response", images.len());

                let mut image_contents = Vec::new();
                for (i, image_data) in images.iter().enumerate() {
                    tracing::info!("📸 Processing image {} of {}", i + 1, images.len());
                    let b64_data = image_data.get("b64_json")
                        .and_then(|b| b.as_str())
                        .ok_or_else(|| {
                            ProviderError::RequestFailed("Invalid image data: missing b64_json".to_string())
                        })?;

                    tracing::info!("📏 Image {} base64 data length: {} characters", i + 1, b64_data.len());

                    image_contents.push(ImageContent {
                        mime_type: "image/jpeg".to_string(),
                        data: b64_data.to_string(),
                        annotations: None,
                    });
                }

                let usage = Usage::new(None, None, None);
                let provider_usage = ProviderUsage::new(self.model.model_name.clone(), usage);

                let total_duration = start_time.elapsed();
                tracing::info!("✅ Image generation completed in {:?} - generated {} images", total_duration, image_contents.len());

                Ok(ImageGenerationResult {
                    images: image_contents,
                    usage: provider_usage,
                })
            }
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                tracing::error!("❌ Authentication failed with status: {}", status);
                Err(ProviderError::Authentication(format!(
                    "Authentication failed. Status: {}. Response: {:?}", status, response_data
                )))
            }
            StatusCode::BAD_REQUEST => {
                tracing::error!("❌ Bad request with status: {}", status);
                tracing::error!("📦 Request payload that caused bad request: {}", serde_json::to_string(&payload).unwrap());

                let error_msg = if let Some(data) = response_data {
                    tracing::error!("📄 Response data: {}", serde_json::to_string(&data).unwrap());
                    let data_owned = data.clone();
                    let error_val = data_owned.get("error");
                    let message_val = match error_val {
                        Some(e) => e.get("message"),
                        None => None,
                    };
                    let msg = match message_val {
                        Some(m) => m.as_str(),
                        None => None,
                    };
                    match msg {
                        Some(s) => s.to_string(),
                        None => "Bad request".to_string(),
                    }
                } else {
                    "Bad request".to_string()
                };

                Err(ProviderError::RequestFailed(format!(
                    "Request failed: {}", error_msg
                )))
            }
            StatusCode::TOO_MANY_REQUESTS => {
                tracing::error!("❌ Rate limit exceeded");
                Err(ProviderError::RateLimitExceeded(format!("{:?}", response_data)))
            }
            StatusCode::INTERNAL_SERVER_ERROR | StatusCode::SERVICE_UNAVAILABLE => {
                tracing::error!("❌ Server error with status: {}", status);
                Err(ProviderError::ServerError(format!("{:?}", response_data)))
            }
            _ => {
                tracing::error!("❌ Unexpected status: {}", status);
                Err(ProviderError::RequestFailed(format!(
                    "Image generation failed with status: {}", status
                )))
            }
        }
    }
}
