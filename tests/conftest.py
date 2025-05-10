import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    response = MagicMock()
    response.choices = [
        MagicMock(message=MagicMock(content="Test response"))
    ]
    response.usage = MagicMock(total_tokens=10)
    response.model_dump = MagicMock(return_value={"id": "test-id"})
    return response

@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response."""
    return {
        'content': [{'text': 'Test response'}],
        'usage': {'input_tokens': 5, 'output_tokens': 5},
        'model': 'claude-3-opus-20240229'
    }

@pytest.fixture
def mock_mistral_response():
    """Create a mock Mistral API response."""
    return {
        'choices': [{'message': {'content': 'Test response'}}],
        'usage': {'input_tokens': 5, 'output_tokens': 5},
        'model': 'mistral-small-latest'
    }

@pytest.fixture
def mock_session_state():
    """Create a mock Streamlit session state."""
    return {
        'test_cases': [],
        'results': [],
        'selected_providers': {
            'openai': {'enabled': False, 'model': 'gpt-4'},
            'anthropic': {'enabled': False, 'model': 'claude-3-opus-20240229'},
            'mistral': {'enabled': False, 'model': 'mistral-small-latest'},
        }
    }

@pytest.fixture
def mock_st():
    """Mock Streamlit components."""
    with patch('streamlit.set_page_config'), \
         patch('streamlit.markdown'), \
         patch('streamlit.columns'), \
         patch('streamlit.checkbox'), \
         patch('streamlit.selectbox'), \
         patch('streamlit.slider'), \
         patch('streamlit.text_area'), \
         patch('streamlit.button'), \
         patch('streamlit.download_button'), \
         patch('streamlit.spinner'):
        yield 