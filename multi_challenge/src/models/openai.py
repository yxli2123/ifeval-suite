from openai import OpenAI
import os
from typing import Any, Dict
from .base import ModelProvider

class OpenAIModel(ModelProvider):
    """OpenAI model provider that uses GPT-4 for evaluation."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        temp: float,
        api_key: str | None = None,
    ):
        """Initialize OpenAI API with the environment variable and other necessary parameters."""
        api_key = os.getenv("OPENAI_API_KEY") if not api_key else api_key
        if not api_key:
            raise ValueError("Missing api_key or OPENAI_API_KEY not set in the .env file.")
        
        self.client = OpenAI(base_url=base_url, api_key=api_key)

        self.model = model_name
        self.temp = float(temp)

    def generate(self, prompt:Any):
        """Generate a response using the OpenAI GPT-4 model."""
        if type(prompt) == str:
            prompt = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) and 'role' in item and item['role'] in ['user', 'assistant'] for item in prompt):
            pass 
        else:
            raise ValueError("Prompt must be a string or a list of dictionaries with 'role' keys as 'user' or 'assistant'.")
        
        response = self.client.chat.completions.create(
                model = self.model,
                messages = prompt,
                temperature = self.temp
            )
        return response.choices[0].message.content