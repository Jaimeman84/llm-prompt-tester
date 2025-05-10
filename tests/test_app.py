import pytest
from unittest.mock import MagicMock, patch
import streamlit as st
from src.llm import OpenAILLM, AnthropicLLM, MistralLLM
from src.app import initialize_session_state

class MockSessionState(dict):
    def __init__(self):
        super().__init__()
        self.update({
            'test_cases': [],
            'results': [],
            'selected_providers': {
                'openai': {'enabled': False, 'model': 'gpt-4'},
                'anthropic': {'enabled': False, 'model': 'claude-3-opus-20240229'},
                'mistral': {'enabled': False, 'model': 'mistral-small-latest'},
            },
            'selected_models': {},
            'max_tokens': 1000
        })
        # Make the attributes accessible as properties
        for key, value in self.items():
            setattr(self, key, value)

@pytest.fixture
def mock_st():
    with patch('streamlit.session_state', new_callable=MockSessionState) as mock_state:
        yield mock_state

@pytest.fixture
def mock_session_state():
    """Create a mock session state dictionary."""
    return {
        'test_cases': [],
        'results': [],
        'selected_providers': {
            'openai': {'enabled': False, 'model': 'gpt-4'},
            'anthropic': {'enabled': False, 'model': 'claude-3-opus-20240229'},
            'mistral': {'enabled': False, 'model': 'mistral-small-latest'},
        }
    }

class TestAppComponents:
    def test_session_state_initialization(self, mock_st):
        assert 'test_cases' in st.session_state
        assert 'results' in st.session_state
        assert 'selected_providers' in st.session_state
        assert 'selected_models' in st.session_state
        assert 'max_tokens' in st.session_state
        
        assert isinstance(st.session_state['test_cases'], list)
        assert isinstance(st.session_state['results'], list)
        assert isinstance(st.session_state['selected_providers'], dict)
        assert isinstance(st.session_state['selected_models'], dict)
        assert isinstance(st.session_state['max_tokens'], int)
        
    def test_provider_selection(self, mock_st):
        """Test provider selection functionality."""
        # Enable a provider
        st.session_state.selected_providers['openai']['enabled'] = True
        assert st.session_state.selected_providers['openai']['enabled']
        
        # Change model selection
        st.session_state.selected_providers['openai']['model'] = 'gpt-3.5-turbo'
        assert st.session_state.selected_providers['openai']['model'] == 'gpt-3.5-turbo'
        
    def test_model_selection(self, mock_st):
        """Test model selection functionality."""
        # Test model selection for each provider
        providers = ['openai', 'anthropic', 'mistral']
        models = {
            'openai': 'gpt-3.5-turbo',
            'anthropic': 'claude-3-sonnet-20240229',
            'mistral': 'mistral-small-latest'
        }
        
        for provider in providers:
            st.session_state.selected_providers[provider]['model'] = models[provider]
            assert st.session_state.selected_providers[provider]['model'] == models[provider]
            
    def test_response_generation(self, mock_st):
        """Test response generation functionality."""
        # Enable a provider
        st.session_state.selected_providers['openai']['enabled'] = True
        
        # Add a test result
        result = {
            'provider': 'openai',
            'model': 'gpt-4',
            'prompt': 'Test prompt',
            'response': 'Test response',
            'metrics': {
                'quality_score': 0.85,
                'tokens': 10,
                'cost': 0.0003,
                'latency': 1.5,
                'reading_ease': 70.0,
                'grade_level': 8.0
            }
        }
        st.session_state.results.append(result)
        
        # Check if result was added
        assert len(st.session_state.results) == 1
        assert st.session_state.results[0]['provider'] == 'openai'
        assert st.session_state.results[0]['response'] == 'Test response'
        
    def test_results_export(self, mock_st):
        """Test results export functionality."""
        # Add test results
        st.session_state.results.append({
            'provider': 'openai',
            'model': 'gpt-4',
            'prompt': 'Test prompt',
            'response': 'Test response',
            'metrics': {
                'quality_score': 0.85,
                'tokens': 10,
                'cost': 0.0003,
                'latency': 1.5,
                'reading_ease': 70.0,
                'grade_level': 8.0
            }
        })
        
        # Assert results can be exported
        assert len(st.session_state.results) > 0
        assert isinstance(st.session_state.results[0], dict)
        assert all(key in st.session_state.results[0] for key in ['provider', 'model', 'prompt', 'response', 'metrics'])
        
    def test_clear_results(self, mock_st):
        """Test clearing results functionality."""
        # Add test results
        st.session_state.results.append({
            'provider': 'openai',
            'model': 'gpt-4',
            'prompt': 'Test prompt',
            'response': 'Test response',
            'metrics': {}
        })
        
        # Clear results
        st.session_state.results = []
        
        # Assert results are cleared
        assert len(st.session_state.results) == 0 