use async_trait::async_trait;
use opencrust_common::Result;
use std::path::PathBuf;

use super::{Tool, ToolContext, ToolOutput};

/// Allow the agent to save a reusable skill to the skills directory.
///
/// The skill is validated (name format, non-empty description and body),
/// written to `{skills_dir}/{name}.md`, and immediately picked up by the
/// hot-reload watcher without requiring a gateway restart.
pub struct CreateSkillTool {
    skills_dir: PathBuf,
}

impl CreateSkillTool {
    pub fn new(skills_dir: impl Into<PathBuf>) -> Self {
        Self {
            skills_dir: skills_dir.into(),
        }
    }
}

#[async_trait]
impl Tool for CreateSkillTool {
    fn name(&self) -> &str {
        "create_skill"
    }

    fn description(&self) -> &str {
        "Save a reusable skill to the skills directory. The skill will be immediately \
         available in future conversations without restarting the gateway."
    }

    fn system_hint(&self) -> Option<&str> {
        Some(
            "Use `create_skill` when you solve a novel multi-step problem or discover \
             a reusable pattern. Good candidates: recurring task workflows, domain-specific \
             sequences, or multi-tool chains you had to figure out step by step. \
             Do NOT save trivial one-liners or things already in your base knowledge.",
        )
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unique skill name in hyphen-case (e.g. 'disk-cleanup'). Only alphanumeric characters and hyphens."
                },
                "description": {
                    "type": "string",
                    "description": "One-line description of what this skill does."
                },
                "body": {
                    "type": "string",
                    "description": "Markdown instructions for the skill — the reusable knowledge or step-by-step procedure."
                },
                "triggers": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional keywords or phrases that suggest using this skill (e.g. ['disk full', 'free space'])."
                }
            },
            "required": ["name", "description", "body"]
        })
    }

    async fn execute(
        &self,
        _context: &ToolContext,
        input: serde_json::Value,
    ) -> Result<ToolOutput> {
        let name = match input.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => return Ok(ToolOutput::error("missing required parameter: 'name'")),
        };
        let description = match input.get("description").and_then(|v| v.as_str()) {
            Some(d) => d.to_string(),
            None => {
                return Ok(ToolOutput::error(
                    "missing required parameter: 'description'",
                ));
            }
        };
        let body = match input.get("body").and_then(|v| v.as_str()) {
            Some(b) => b.to_string(),
            None => return Ok(ToolOutput::error("missing required parameter: 'body'")),
        };
        let triggers: Vec<String> = input
            .get("triggers")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        // Build SKILL.md content
        let mut content = format!("---\nname: {name}\ndescription: {description}\n");
        if !triggers.is_empty() {
            content.push_str("triggers:\n");
            for t in &triggers {
                content.push_str(&format!("  - {t}\n"));
            }
        }
        content.push_str("---\n\n");
        content.push_str(&body);
        content.push('\n');

        // Write via SkillInstaller — handles dir creation + validation
        let installer = opencrust_skills::SkillInstaller::new(&self.skills_dir);
        let tmp = std::env::temp_dir().join(format!("opencrust_skill_{name}.md"));
        if let Err(e) = std::fs::write(&tmp, &content) {
            return Ok(ToolOutput::error(format!(
                "failed to stage skill file: {e}"
            )));
        }

        match installer.install_from_path(&tmp) {
            Ok(skill) => {
                let _ = std::fs::remove_file(&tmp);
                Ok(ToolOutput::success(format!(
                    "skill '{}' saved to {} — active immediately",
                    skill.frontmatter.name,
                    self.skills_dir.join(format!("{name}.md")).display()
                )))
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                Ok(ToolOutput::error(format!("invalid skill: {e}")))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ToolContext {
        ToolContext {
            session_id: "test".into(),
            user_id: None,
            heartbeat_depth: 0,
            allowed_tools: None,
        }
    }

    #[tokio::test]
    async fn creates_skill_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let tool = CreateSkillTool::new(dir.path());

        let out = tool
            .execute(
                &ctx(),
                serde_json::json!({
                    "name": "disk-cleanup",
                    "description": "Free up disk space on macOS",
                    "body": "Run `df -h` to check usage, then `brew cleanup` to remove caches.",
                    "triggers": ["disk full", "free space"]
                }),
            )
            .await
            .unwrap();

        assert!(!out.is_error, "unexpected error: {}", out.content);
        assert!(dir.path().join("disk-cleanup.md").exists());

        let written = std::fs::read_to_string(dir.path().join("disk-cleanup.md")).unwrap();
        assert!(written.contains("name: disk-cleanup"));
        assert!(written.contains("triggers:"));
        assert!(written.contains("brew cleanup"));
    }

    #[tokio::test]
    async fn rejects_invalid_name() {
        let dir = tempfile::TempDir::new().unwrap();
        let tool = CreateSkillTool::new(dir.path());

        let out = tool
            .execute(
                &ctx(),
                serde_json::json!({
                    "name": "bad name!",
                    "description": "test",
                    "body": "something"
                }),
            )
            .await
            .unwrap();

        assert!(out.is_error);
        assert!(out.content.contains("invalid"));
    }

    #[tokio::test]
    async fn rejects_missing_name() {
        let dir = tempfile::TempDir::new().unwrap();
        let tool = CreateSkillTool::new(dir.path());

        let out = tool
            .execute(
                &ctx(),
                serde_json::json!({"description": "test", "body": "body"}),
            )
            .await
            .unwrap();

        assert!(out.is_error);
    }

    #[tokio::test]
    async fn works_without_triggers() {
        let dir = tempfile::TempDir::new().unwrap();
        let tool = CreateSkillTool::new(dir.path());

        let out = tool
            .execute(
                &ctx(),
                serde_json::json!({
                    "name": "simple-skill",
                    "description": "A simple skill",
                    "body": "Do something useful."
                }),
            )
            .await
            .unwrap();

        assert!(!out.is_error, "{}", out.content);
        let content = std::fs::read_to_string(dir.path().join("simple-skill.md")).unwrap();
        assert!(!content.contains("triggers:"));
    }
}
