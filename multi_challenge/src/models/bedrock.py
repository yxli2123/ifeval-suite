import os
import boto3
from typing import Any


class BedrockModel:
    """Bedrock model provider (Claude via Converse API)."""

    def __init__(
        self,
        model_id: str,
        temp: float = 0.2,
        region: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,  # for short-term creds
    ):
        region = region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        if not region:
            raise ValueError("Missing region. Set AWS_REGION (or pass region=...).")

        aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")  # optional unless using temp creds

        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError(
                "Missing AWS credentials. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
                "(and AWS_SESSION_TOKEN if using short-term credentials)."
            )

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )
        self.model_id = model_id
        self.temp = float(temp)

    def generate(self, prompt: Any, max_tokens: int = 8192) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": [{"text": prompt}]}]
        elif (
            isinstance(prompt, list)
            and all(isinstance(m, dict) and m.get("role") in {"user", "assistant"} for m in prompt)
        ):
            # Convert OpenAI-like message objects to Bedrock content blocks
            messages = []
            for m in prompt:
                content = m.get("content", "")
                if not isinstance(content, str):
                    raise ValueError("Each message['content'] must be a string.")
                messages.append({"role": m["role"], "content": [{"text": content}]})
        else:
            raise ValueError(
                "Prompt must be a string or a list of dicts like "
                "[{'role': 'user'|'assistant', 'content': '...'}]."
            )

        resp = self.client.converse(
            modelId=self.model_id,
            messages=messages,
            inferenceConfig={
                "temperature": self.temp,
                "maxTokens": int(max_tokens),
            },
        )

        blocks = resp["output"]["message"]["content"]  # list of content blocks
        return "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
