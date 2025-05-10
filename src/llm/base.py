from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import time

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    max_tokens: int
    temperature: float
    description: str
    cost_per_1k_tokens: float
    typical_latency: str  # e.g. "~0.5s", "1-2s", etc.

@dataclass
class LLMResponse:
    """Data class to store LLM response and metadata."""
    text: str
    tokens_used: int
    latency: float
    model: str
    cost: float
    raw_response: Optional[Dict[str, Any]] = None

class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.models = self.get_available_models()
        
    @abstractmethod
    def get_available_models(self) -> Dict[str, ModelConfig]:
        """Get available models for this LLM provider.
        
        Returns:
            Dict[str, ModelConfig]: Dictionary of model configurations
        """
        pass
    
    @abstractmethod
    async def generate(self, 
                      prompt: str, 
                      model: str,
                      temperature: float = 0.7,
                      max_tokens: int = 2000) -> LLMResponse:
        """Generate a response for the given prompt.
        
        Args:
            prompt (str): The input prompt to send to the LLM
            model (str): The specific model to use
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            LLMResponse: Object containing the response and metadata
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str, model: str) -> int:
        """Count the number of tokens in the given text.
        
        Args:
            text (str): The text to count tokens for
            model (str): The model to use for token counting
            
        Returns:
            int: Number of tokens
        """
        pass
    
    def measure_latency(self, func):
        """Decorator to measure function execution time."""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            return result, end_time - start_time
        return wrapper
    
    def calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate the cost of the API call.
        
        Args:
            tokens (int): Number of tokens used
            model (str): Model used for generation
            
        Returns:
            float: Cost in USD
        """
        cost_per_1k = self.models[model].cost_per_1k_tokens
        return (tokens / 1000) * cost_per_1k 