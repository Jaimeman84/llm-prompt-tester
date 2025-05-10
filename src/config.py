import os
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv

@dataclass
class LLMConfig:
    """Configuration for LLM models."""
    model_name: str
    api_key: str
    default_params: Dict[str, Any]

@dataclass
class AppConfig:
    """Application configuration."""
    debug: bool = False
    log_level: str = "INFO"
    max_tokens: int = 2000
    default_temperature: float = 0.7
    batch_size: int = 10
    timeout: int = 30

class Config:
    """Global configuration singleton."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize configuration from environment variables."""
        load_dotenv()
        
        # LLM configurations
        self.llm_configs = {
            'openai': LLMConfig(
                model_name='gpt-4',
                api_key=os.getenv('OPENAI_API_KEY', ''),
                default_params={
                    'temperature': 0.7,
                    'max_tokens': 2000,
                    'top_p': 1.0
                }
            ),
            'anthropic': LLMConfig(
                model_name='claude-3-opus-20240229',
                api_key=os.getenv('ANTHROPIC_API_KEY', ''),
                default_params={
                    'temperature': 0.7,
                    'max_tokens': 2000
                }
            ),
            'mistral': LLMConfig(
                model_name='mistral-large-latest',
                api_key=os.getenv('MISTRAL_API_KEY', ''),
                default_params={
                    'temperature': 0.7,
                    'max_tokens': 2000,
                    'top_p': 0.95
                }
            )
        }
        
        # App configuration
        self.app = AppConfig(
            debug=os.getenv('DEBUG', 'false').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            max_tokens=int(os.getenv('MAX_TOKENS', '2000')),
            default_temperature=float(os.getenv('DEFAULT_TEMPERATURE', '0.7')),
            batch_size=int(os.getenv('BATCH_SIZE', '10')),
            timeout=int(os.getenv('TIMEOUT', '30'))
        )
    
    @property
    def openai_config(self) -> LLMConfig:
        """Get OpenAI configuration."""
        return self.llm_configs['openai']
    
    @property
    def anthropic_config(self) -> LLMConfig:
        """Get Anthropic configuration."""
        return self.llm_configs['anthropic']
    
    @property
    def mistral_config(self) -> LLMConfig:
        """Get Mistral configuration."""
        return self.llm_configs['mistral'] 