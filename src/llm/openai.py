from typing import Dict, Any
import tiktoken
from openai import AsyncOpenAI
from .base import BaseLLM, LLMResponse, ModelConfig

class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = AsyncOpenAI(api_key=api_key)
        self.tokenizers = {}
        
    def get_available_models(self) -> Dict[str, ModelConfig]:
        """Get available OpenAI models and their configurations."""
        return {
            'gpt-4': ModelConfig(
                name='GPT-4',
                max_tokens=8192,
                temperature=0.7,
                description='Most capable model, best for complex tasks requiring deep understanding',
                cost_per_1k_tokens=0.03,
                typical_latency='2-3s'
            ),
            'gpt-4-turbo': ModelConfig(
                name='GPT-4 Turbo',
                max_tokens=128000,
                temperature=0.7,
                description='Latest GPT-4 model with larger context window',
                cost_per_1k_tokens=0.01,
                typical_latency='2-3s'
            ),
            'gpt-3.5-turbo': ModelConfig(
                name='GPT-3.5 Turbo',
                max_tokens=4096,
                temperature=0.7,
                description='Fast and cost-effective for most tasks',
                cost_per_1k_tokens=0.0015,
                typical_latency='0.5-1s'
            ),
            'gpt-3.5-turbo-16k': ModelConfig(
                name='GPT-3.5 Turbo 16K',
                max_tokens=16384,
                temperature=0.7,
                description='GPT-3.5 with larger context window',
                cost_per_1k_tokens=0.003,
                typical_latency='1-2s'
            )
        }
        
    async def generate(self, 
                      prompt: str,
                      model: str,
                      temperature: float = 0.7,
                      max_tokens: int = 2000) -> LLMResponse:
        """Generate a response using OpenAI's API.
        
        Args:
            prompt (str): The input prompt
            model (str): The model to use
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            LLMResponse: Object containing the response and metadata
        """
        @self.measure_latency
        async def _generate():
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response
            
        response, latency = await _generate()
        tokens_used = response.usage.total_tokens
        
        return LLMResponse(
            text=response.choices[0].message.content,
            tokens_used=tokens_used,
            latency=latency,
            model=model,
            cost=self.calculate_cost(tokens_used, model),
            raw_response=response.model_dump()
        )
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text using tiktoken.
        
        Args:
            text (str): Text to count tokens for
            model (str): Model to use for token counting
            
        Returns:
            int: Number of tokens
        """
        if model not in self.tokenizers:
            self.tokenizers[model] = tiktoken.encoding_for_model(model)
            
        return len(self.tokenizers[model].encode(text)) 