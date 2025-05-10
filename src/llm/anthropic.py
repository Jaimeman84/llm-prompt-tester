from typing import Dict
from anthropic import AsyncAnthropic
from .base import BaseLLM, LLMResponse, ModelConfig

class AnthropicLLM(BaseLLM):
    """Anthropic LLM implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = AsyncAnthropic(api_key=api_key)
        
    def get_available_models(self) -> Dict[str, ModelConfig]:
        """Get available Anthropic models and their configurations."""
        return {
            'claude-3-opus-20240229': ModelConfig(
                name='Claude 3 Opus',
                max_tokens=4096,
                temperature=0.7,
                description='Most capable Claude model, best for complex tasks',
                cost_per_1k_tokens=0.015,
                typical_latency='2-3s'
            ),
            'claude-3-sonnet-20240229': ModelConfig(
                name='Claude 3 Sonnet',
                max_tokens=4096,
                temperature=0.7,
                description='Balanced performance and cost',
                cost_per_1k_tokens=0.003,
                typical_latency='1-2s'
            ),
            'claude-3-haiku-20240307': ModelConfig(
                name='Claude 3 Haiku',
                max_tokens=4096,
                temperature=0.7,
                description='Fastest and most cost-effective Claude model',
                cost_per_1k_tokens=0.0015,
                typical_latency='<1s'
            )
        }
        
    async def generate(self, 
                      prompt: str,
                      model: str,
                      temperature: float = 0.7,
                      max_tokens: int = 2000) -> LLMResponse:
        """Generate a response using Anthropic's API.
        
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
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response
            
        response, latency = await _generate()
        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        
        return LLMResponse(
            text=response.content[0].text,
            tokens_used=tokens_used,
            latency=latency,
            model=model,
            cost=self.calculate_cost(tokens_used, model),
            raw_response=response.model_dump()
        )
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text using Anthropic's token counting.
        
        Args:
            text (str): Text to count tokens for
            model (str): Model to use for token counting
            
        Returns:
            int: Number of tokens
        """
        # Anthropic's API doesn't provide a direct token counting method
        # Using an approximation of 4 characters per token
        return len(text) // 4 