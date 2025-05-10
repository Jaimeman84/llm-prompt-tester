from typing import Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
from mistralai.client import MistralClient
from .base import BaseLLM, LLMResponse, ModelConfig

class MistralLLM(BaseLLM):
    """Mistral LLM implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = MistralClient(api_key=api_key)
        self._executor = ThreadPoolExecutor(max_workers=1)
        
    def get_available_models(self) -> Dict[str, ModelConfig]:
        """Get available Mistral models and their configurations."""
        return {
            'mistral-small-latest': ModelConfig(
                name='Mistral Small',
                max_tokens=128000,  # 128k tokens
                temperature=0.7,
                description='A new leader in the small models category with image understanding capabilities',
                cost_per_1k_tokens=0.002,
                typical_latency='<1s'
            ),
            'pixtral-12b-2409': ModelConfig(
                name='Pixtral',
                max_tokens=128000,  # 128k tokens
                temperature=0.7,
                description='A 12B model with image understanding capabilities',
                cost_per_1k_tokens=0.007,
                typical_latency='1-2s'
            ),
            'open-mistral-nemo': ModelConfig(
                name='Mistral Nemo',
                max_tokens=128000,  # 128k tokens
                temperature=0.7,
                description='Best multilingual open source model',
                cost_per_1k_tokens=0.002,
                typical_latency='1-2s'
            )
        }
        
    async def generate(self, 
                      prompt: str,
                      model: str,
                      temperature: float = 0.7,
                      max_tokens: int = 2000) -> LLMResponse:
        """Generate a response using Mistral's API.
        
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
            messages = [{"role": "user", "content": prompt}]
            
            # Run synchronous API call in thread pool since we're in an async context
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self._executor,
                lambda: self.client.chat(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            )
            # Handle both list and single response formats
            if isinstance(response, list):
                response = response[0]
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
        """Count tokens in text using Mistral's token counting.
        
        Args:
            text (str): Text to count tokens for
            model (str): Model to use for token counting
            
        Returns:
            int: Number of tokens
        """
        # Mistral's API doesn't provide a direct token counting method
        # Using an approximation of 4 characters per token
        return len(text) // 4 