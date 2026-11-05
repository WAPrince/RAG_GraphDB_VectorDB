import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class AgentSettings(BaseSettings):

    aws_region: str = "us-east-1"
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    bedrock_model_id: str =  (
        "us.anthropic.claude-sonnet-4-6"
    )
    model_config = SettingsConfigDict(
        env_file=".env"
    )

settings = AgentSettings()