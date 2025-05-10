import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from src.llm import OpenAILLM, AnthropicLLM, MistralLLM, LLMResponse
from datetime import datetime

@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    response = MagicMock()
    response.choices = [
        MagicMock(message=MagicMock(content="Test response"))
    ]
    response.usage = MagicMock(total_tokens=10)
    response.model = "gpt-4"
    return response

@pytest.fixture
def openai_llm():
    """Create an OpenAI LLM instance with test API key."""
    return OpenAILLM(api_key="test-key")

@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response."""
    response = MagicMock()
    response.content = [MagicMock(text="Test response")]
    response.usage = MagicMock(input_tokens=5, output_tokens=5)
    response.model = "claude-3-opus-20240229"
    return response

@pytest.fixture
def mock_mistral_response():
    """Create a mock Mistral API response."""
    response = MagicMock()
    response.choices = [
        MagicMock(message=MagicMock(content="Test response"))
    ]
    response.usage = MagicMock(total_tokens=10)
    response.model = "mistral-small-latest"
    return response

@pytest.mark.asyncio
async def test_generate_success(mock_openai_response):
    """Test successful response generation."""
    llm = OpenAILLM(api_key="test-key")
    
    # Mock the OpenAI client's create method
    llm.client.chat.completions.create = AsyncMock(
        return_value=mock_openai_response
    )
    
    response = await llm.generate(
        prompt="Test prompt",
        model="gpt-4",
        temperature=0.7,
        max_tokens=2000
    )
    
    assert isinstance(response, LLMResponse)
    assert response.text == "Test response"
    assert response.tokens_used == 10
    assert response.model == "gpt-4"
    assert isinstance(response.latency, float)

@pytest.mark.asyncio
async def test_generate_api_error():
    """Test handling of API errors."""
    llm = OpenAILLM(api_key="test-key")
    
    # Mock API error
    llm.client.chat.completions.create = AsyncMock(
        side_effect=Exception("API Error")
    )
    
    with pytest.raises(Exception) as exc_info:
        await llm.generate(
            prompt="Test prompt",
            model="gpt-4"
        )
    assert str(exc_info.value) == "API Error"

def test_count_tokens():
    """Test token counting functionality."""
    llm = OpenAILLM(api_key="test-key")
    text = "This is a test prompt."
    
    # Mock tiktoken encoding
    with patch('tiktoken.encoding_for_model') as mock_encoding:
        mock_encoding.return_value.encode.return_value = [1, 2, 3, 4, 5]
        token_count = llm.count_tokens(text, model="gpt-4")
        
        assert isinstance(token_count, int)
        assert token_count == 5

@pytest.mark.asyncio
async def test_generate_with_parameters(mock_openai_response):
    """Test generation with custom parameters."""
    llm = OpenAILLM(api_key="test-key")
    
    # Mock the OpenAI client's create method
    llm.client.chat.completions.create = AsyncMock(
        return_value=mock_openai_response
    )
    
    await llm.generate(
        prompt="Test prompt",
        model="gpt-4",
        temperature=0.5,
        max_tokens=1000
    )
    
    # Verify the API was called with correct parameters
    llm.client.chat.completions.create.assert_called_once_with(
        model="gpt-4",
        messages=[{"role": "user", "content": "Test prompt"}],
        temperature=0.5,
        max_tokens=1000
    )

def test_initialization():
    """Test LLM initialization."""
    llm = OpenAILLM(api_key="test-key")
    assert llm.api_key == "test-key"
    assert "gpt-4" in llm.get_available_models()

class TestOpenAILLM:
    @pytest.mark.asynciods
    async def test_generate(self, mock_openai_response):
        """Test OpenAI LLM generation."""
        with patch('openai.AsyncClient') as MockClient:
            # Mock the client instance
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            MockClient.return_value = mock_client
            
            # Create LLM instance after mocking the client
            llm = OpenAILLM("test_key")
            llm.client = mock_client  # Replace the client with our mock
            
            response = await llm.generate(
                prompt="Test prompt",
                model="gpt-4",
                temperature=0.7,
                max_tokens=100
            )
            
            assert isinstance(response, LLMResponse)
            assert response.text == "Test response"
            assert response.tokens_used == 10
            assert response.model == "gpt-4"
            assert isinstance(response.latency, float)

    def test_get_available_models(self):
        """Test getting available OpenAI models."""
        llm = OpenAILLM("test_key")
        models = llm.get_available_models()
        assert isinstance(models, dict)
        assert len(models) > 0
        assert 'gpt-4' in models

class TestAnthropicLLM:
    @pytest.mark.asyncio
    async def test_generate(self, mock_anthropic_response):
        """Test Anthropic LLM generation."""
        with patch('anthropic.AsyncAnthropic') as MockClient:
            # Mock the client instance
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_anthropic_response)
            MockClient.return_value = mock_client
            
            # Create LLM instance after mocking the client
            llm = AnthropicLLM("test_key")
            llm.client = mock_client  # Replace the client with our mock
            
            response = await llm.generate(
                prompt="Test prompt",
                model="claude-3-opus-20240229",
                temperature=0.7,
                max_tokens=100
            )
            
            assert isinstance(response, LLMResponse)
            assert response.text == "Test response"
            assert response.tokens_used == 10
            assert response.model == "claude-3-opus-20240229"
            assert isinstance(response.latency, float)

    def test_get_available_models(self):
        """Test getting available Anthropic models."""
        llm = AnthropicLLM("test_key")
        models = llm.get_available_models()
        assert isinstance(models, dict)
        assert len(models) > 0
        assert 'claude-3-opus-20240229' in models

class TestMistralLLM:
    @pytest.mark.asyncio
    async def test_generate(self, mock_mistral_response):
        """Test Mistral LLM generation."""
        with patch('mistralai.client.MistralClient') as MockClient:
            # Mock the client instance
            mock_client = MagicMock()
            mock_client.chat = MagicMock(return_value=mock_mistral_response)  # Return the response directly, not in a list
            MockClient.return_value = mock_client
            
            # Create LLM instance after mocking the client
            llm = MistralLLM("test_key")
            llm.client = mock_client  # Replace the client with our mock
            
            response = await llm.generate(
                prompt="Test prompt",
                model="mistral-small-latest",
                temperature=0.7,
                max_tokens=100
            )
            
            assert isinstance(response, LLMResponse)
            assert response.text == "Test response"
            assert response.tokens_used == 10
            assert response.model == "mistral-small-latest"
            assert isinstance(response.latency, float)

    def test_get_available_models(self):
        """Test getting available Mistral models."""
        llm = MistralLLM("test_key")
        models = llm.get_available_models()
        assert isinstance(models, dict)
        assert len(models) > 0
        assert 'mistral-small-latest' in models 