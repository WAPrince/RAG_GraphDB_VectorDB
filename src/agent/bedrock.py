
from langchain_aws import ChatBedrockConverse

def create_bedrock_llm() -> ChatBedrockConverse:

    return ChatBedrockConverse(
        model="us.anthropic.claude-sonnet-4-6",
        region_name="us-east-1",
        temperature=0,
        max_tokens=1000
    )