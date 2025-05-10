import os
import asyncio
import yaml
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from src.llm.openai import OpenAILLM
from src.llm.anthropic import AnthropicLLM
from src.llm.mistral import MistralLLM
from src.evaluation.metrics import OutputEvaluator

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize Streamlit's session state with default values."""
    if 'test_cases' not in st.session_state:
        st.session_state.test_cases = []
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'selected_providers' not in st.session_state:
        st.session_state.selected_providers = {
            'openai': {'enabled': False, 'model': 'gpt-4'},
            'anthropic': {'enabled': False, 'model': 'claude-3-opus-20240229'},
            'mistral': {'enabled': False, 'model': 'mistral-small-latest'},
        }

# Initialize session state
initialize_session_state()

# Initialize LLM clients and evaluator
openai_llm = OpenAILLM(api_key=os.getenv('OPENAI_API_KEY', ''))
anthropic_llm = AnthropicLLM(api_key=os.getenv('ANTHROPIC_API_KEY', ''))
mistral_llm = MistralLLM(api_key=os.getenv('MISTRAL_API_KEY', ''))
evaluator = OutputEvaluator()

# Set dark theme and wide layout
st.set_page_config(layout="wide")

# Add logo and title
st.markdown("""
<div style="text-align: center;">
    <h1 style="font-size: 3em; margin-bottom: 0.5rem; font-weight: 600; color: #FFFFFF;">
        🤖 LLM TESTER
    </h1>
    <p style="font-size: 1.2em; color: #A0A0A0; margin: 0;">
        Compare, Analyze, and Evaluate LLM Responses
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        padding: 0.5rem 1rem !important;
    }
    
    /* Model info styling */
    .model-info {
        font-family: "Source Code Pro", monospace !important;
        font-size: 0.75em !important;
        line-height: 1.2 !important;
        background-color: #1A1A1A !important;
        border-radius: 4px !important;
        padding: 0.35rem 0.5rem !important;
        margin-top: 0.25rem !important;
        color: #A0A0A0 !important;
    }
    
    .model-info code {
        white-space: pre !important;
        color: #A0A0A0 !important;
        background: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Override Streamlit's default spacing */
    .st-emotion-cache-vlxhtx {
        gap: 0.25rem !important;
    }
    
    /* Title and header styles */
    .stTitle {
        font-size: 2em !important;
        margin: 1rem 0 !important;
        text-align: center !important;
        padding: 0 !important;
    }
    
    .stSubheader {
        font-size: 1.5em !important;
        margin: 0.75rem 0 0.5rem !important;
        padding: 0 !important;
        color: #E0E0E0 !important;
    }
    
    /* Provider section container */
    .provider-container {
        background-color: #1E1E1E;
        padding: 1rem !important;
        border-radius: 8px;
        margin: 0.5rem 0 !important;
        width: 100%;
        border: 1px solid #2E2E2E;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        display: flex;
        flex-direction: column;
    }
    
    .provider-container h3 {
        margin: 0 0 0.5rem 0 !important;
        padding: 0 !important;
        font-size: 1.2em !important;
    }
    
    /* Response container styling */
    .stMarkdown {
        margin: 0 !important;
    }
    
    .stMarkdown > div {
        margin: 0 !important;
    }
    
    /* Output text container */
    .element-container:has(pre) {
        height: 200px !important;
        overflow-y: auto !important;
        margin: 0.5rem 0 !important;
        padding-right: 8px !important;
    }
    
    /* Response section */
    .response-container {
        background-color: #1E1E1E;
        padding: 0.5rem !important;
        border-radius: 8px;
        margin: 0.25rem 0 !important;
        width: 100%;
        border: 1px solid #2E2E2E;
    }
    
    .response-container p {
        margin: 0 0 0.25rem 0 !important;
        padding: 0 !important;
    }
    
    /* Table styling */
    .dataframe {
        width: 100% !important;
        margin: 0 !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    
    .dataframe thead th {
        background-color: #2E2E2E !important;
        padding: 0.75rem !important;
        font-weight: 600 !important;
        text-align: left !important;
        border-bottom: 2px solid #3E3E3E !important;
        color: #E0E0E0 !important;
        font-size: 0.9em !important;
    }
    
    .dataframe tbody td {
        padding: 0.75rem !important;
        border-bottom: 1px solid #2E2E2E !important;
        font-size: 0.9em !important;
        line-height: 1.3 !important;
        vertical-align: middle !important;
    }
    
    .dataframe tbody td:first-child {
        font-weight: 500 !important;
        color: #E0E0E0 !important;
    }
    
    .dataframe tbody td:last-child {
        text-align: right !important;
    }
    
    .dataframe tbody tr:last-child td {
        border-bottom: none !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #2A2A2A !important;
    }
    
    /* Input fields and controls */
    .stTextArea {
        margin: 0.5rem 0 !important;
    }
    
    .stTextArea textarea {
        background-color: #1E1E1E !important;
        border: 1px solid #2E2E2E !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        font-size: 1em !important;
        color: #FAFAFA !important;
        min-height: 80px !important;
    }
    
    .stSlider {
        padding: 1rem 0.5rem !important;
        margin: 0.25rem 0 !important;
    }
    
    /* Code block styles */
    .stCodeBlock {
        background-color: #161B22 !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        margin: 0 !important;
        border: 1px solid #30363D !important;
    }
    
    .stCodeBlock code {
        font-family: 'Consolas', monospace !important;
        font-size: 0.9em !important;
        line-height: 1.4 !important;
        white-space: pre-wrap !important;
    }
    
    /* Make columns equal width with proper spacing */
    .row-widget.stHorizontal {
        display: flex !important;
        gap: 1rem !important;
        margin: 0.75rem 0 !important;
        padding: 0 0.5rem !important;
        align-items: stretch !important;
    }
    
    .row-widget.stHorizontal > div {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        padding: 1rem !important;
        background-color: #1E1E1E !important;
        border: 1px solid #2E2E2E !important;
        border-radius: 8px !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* Section headers */
    .section-header {
        margin: 1rem 0 0.5rem 0 !important;
        padding: 0.75rem !important;
        background-color: #1E1E1E !important;
        border-radius: 8px !important;
        border: 1px solid #2E2E2E !important;
        font-size: 1.2em !important;
    }
    
    /* Ensure text wrapping in code blocks */
    .element-container pre {
        margin: 0 !important;
        height: 100% !important;
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    
    .element-container code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        display: block !important;
        width: 100% !important;
    }
    
    /* Custom scrollbar for output container */
    .element-container::-webkit-scrollbar {
        width: 8px !important;
    }
    
    .element-container::-webkit-scrollbar-track {
        background: #1E1E1E !important;
        border-radius: 4px !important;
    }
    
    .element-container::-webkit-scrollbar-thumb {
        background: #2E2E2E !important;
        border-radius: 4px !important;
    }
    
    .element-container::-webkit-scrollbar-thumb:hover {
        background: #3E3E3E !important;
    }
    
    /* Make columns equal height */
    .row-widget.stHorizontal {
        display: flex !important;
        gap: 1rem !important;
        margin: 0.75rem 0 !important;
        padding: 0 0.5rem !important;
        align-items: stretch !important;
    }
    
    .row-widget.stHorizontal > div {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        padding: 1rem !important;
        background-color: #1E1E1E !important;
        border: 1px solid #2E2E2E !important;
        border-radius: 8px !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* Ensure code blocks fill the container */
    pre {
        margin: 0 !important;
        height: 100% !important;
    }
    
    /* Download button styling */
    .stDownloadButton {
        margin: 1rem 0 !important;
        display: flex !important;
        justify-content: center !important;
    }
    
    .stDownloadButton > button {
        background-color: #2E2E2E !important;
        color: #FFFFFF !important;
        border: 1px solid #3E3E3E !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 1em !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        width: auto !important;
        min-width: 250px !important;
        margin: 0 auto !important;
    }
    
    .stDownloadButton > button:hover {
        background-color: #3E3E3E !important;
        border-color: #4E4E4E !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    
    /* Caption styling */
    .caption-text {
        color: #A0A0A0 !important;
        font-size: 0.8em !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.2 !important;
        font-family: "Source Code Pro", monospace !important;
    }
    
    /* Add spacing after the model dropdown */
    .stSelectbox {
        margin-bottom: 0.5rem !important;
    }
    
    /* Container for model info */
    div:has(> p.caption-text) {
        margin-top: 0.25rem !important;
        background-color: #1A1A1A !important;
        border-radius: 4px !important;
        padding: 0.25rem 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Add section headers with custom styling
st.markdown('<h2 class="section-header">Provider Selection</h2>', unsafe_allow_html=True)

# Create three columns for provider selection
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('OpenAI')
    openai_enabled = st.checkbox('Enable OpenAI', key='openai_enabled')
    if openai_enabled:
        st.session_state.selected_providers['openai']['enabled'] = True
        openai_model = st.selectbox(
            'Model',
            options=['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'],
            key='openai_model'
        )
        st.session_state.selected_providers['openai']['model'] = openai_model
        model_info = openai_llm.get_available_models()[openai_model]
        st.markdown(f"""
        <div class="model-info">
            <code>Cost: ${model_info.cost_per_1k_tokens}/1K tokens</code><br>
            <code>Max tokens: {model_info.max_tokens}</code><br>
            <code>Typical latency: {model_info.typical_latency}</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.session_state.selected_providers['openai']['enabled'] = False

with col2:
    st.subheader('Anthropic')
    anthropic_enabled = st.checkbox('Enable Anthropic', key='anthropic_enabled')
    if anthropic_enabled:
        st.session_state.selected_providers['anthropic']['enabled'] = True
        anthropic_model = st.selectbox(
            'Model',
            options=['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
            key='anthropic_model'
        )
        st.session_state.selected_providers['anthropic']['model'] = anthropic_model
        model_info = anthropic_llm.get_available_models()[anthropic_model]
        st.markdown(f"""
        <div class="model-info">
            <code>Cost: ${model_info.cost_per_1k_tokens}/1K tokens</code><br>
            <code>Max tokens: {model_info.max_tokens}</code><br>
            <code>Typical latency: {model_info.typical_latency}</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.session_state.selected_providers['anthropic']['enabled'] = False

with col3:
    st.subheader('Mistral')
    mistral_enabled = st.checkbox('Enable Mistral', key='mistral_enabled')
    if mistral_enabled:
        st.session_state.selected_providers['mistral']['enabled'] = True
        mistral_model = st.selectbox(
            'Model',
            options=['mistral-small-latest', 'pixtral-12b-2409', 'open-mistral-nemo'],
            key='mistral_model'
        )
        st.session_state.selected_providers['mistral']['model'] = mistral_model
        model_info = mistral_llm.get_available_models()[mistral_model]
        st.markdown(f"""
        <div class="model-info">
            <code>Cost: ${model_info.cost_per_1k_tokens}/1K tokens</code><br>
            <code>Max tokens: {model_info.max_tokens}</code><br>
            <code>Typical latency: {model_info.typical_latency}</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.session_state.selected_providers['mistral']['enabled'] = False

# Generation Parameters
st.title('Generation Parameters')
col1, col2 = st.columns(2)
with col1:
    temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.7,
                          help="Higher values make the output more random, lower values make it more focused")
with col2:
    max_tokens = st.slider('Max Tokens', min_value=100, max_value=2000, value=500,
                          help="Maximum number of tokens to generate")

# Create two columns for prompt and expected output
prompt_col, expected_col = st.columns(2)

with prompt_col:
    prompt = st.text_area('Enter your prompt:')

with expected_col:
    reference = st.text_area('Expected output (optional):')

# Generate button
enabled_providers = [p for p, v in st.session_state.selected_providers.items() if v['enabled']]
if st.button('Generate Responses', disabled=not enabled_providers or not prompt):
    if prompt:
        # Create columns for each enabled provider
        provider_cols = st.columns(len(enabled_providers))
        
        async def test_all_providers():
            tasks = []
            for provider in enabled_providers:
                if provider == 'openai':
                    model = st.session_state.selected_providers[provider]['model']
                    tasks.append(openai_llm.generate(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    ))
                elif provider == 'anthropic':
                    model = st.session_state.selected_providers[provider]['model']
                    tasks.append(anthropic_llm.generate(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    ))
                elif provider == 'mistral':
                    model = st.session_state.selected_providers[provider]['model']
                    tasks.append(mistral_llm.generate(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    ))
            return await asyncio.gather(*tasks)
        
        with st.spinner('Generating responses...'):
            responses = asyncio.run(test_all_providers())
            
            # Display results for each provider
            for idx, col in enumerate(provider_cols):
                provider = enabled_providers[idx]
                model = st.session_state.selected_providers[provider]['model']
                response = responses[idx]
                
                with col:
                    st.markdown(f"""
                    <div class="provider-container">
                        <h3 style='margin-top: 0;'>{provider.title()} ({model})</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Response
                    st.markdown("""
                    <div class="response-container">
                        <p style='margin: 0 0 10px 0;'><strong>Response:</strong></p>
                    """, unsafe_allow_html=True)
                    st.code(response.text, language='text')
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Calculate quality score and color
                    quality_score = evaluator.calculate_overall_score(
                        response.text,
                        reference if reference else None
                    )
                    score_color = 'green' if quality_score >= 0.8 else 'orange' if quality_score >= 0.6 else 'red'
                    
                    # Metrics table
                    metrics_df = pd.DataFrame([
                        ('Quality Score', f'<span style="color: {score_color}; font-weight: bold;">{quality_score:.2f}</span>'),
                        ('Tokens Used', f"{response.tokens_used:,}"),
                        ('Cost', f"${response.cost:.4f}"),
                        ('Latency', f"{response.latency:.2f}s"),
                        ('Reading Ease', f"{evaluator.calculate_readability(response.text)['flesch_reading_ease']:.1f}"),
                        ('Grade Level', f"{evaluator.calculate_readability(response.text)['flesch_kincaid_grade']:.1f}")
                    ], columns=['Metric', 'Value'])
                    
                    st.markdown(metrics_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    
                    # Save result
                    st.session_state.results.append({
                        'provider': provider,
                        'model': model,
                        'prompt': prompt,
                        'response': response.text,
                        'metrics': {
                            'quality_score': quality_score,
                            'tokens': response.tokens_used,
                            'cost': response.cost,
                            'latency': response.latency,
                            'reading_ease': evaluator.calculate_readability(response.text)['flesch_reading_ease'],
                            'grade_level': evaluator.calculate_readability(response.text)['flesch_kincaid_grade']
                        }
                    })
    else:
        st.error('Please enter a prompt.')

# Add download button at the bottom
if st.session_state.results:
    # Create DataFrame for export
    results_df = pd.DataFrame([
        {
            'Provider': r['provider'],
            'Model': r['model'],
            'Prompt': r['prompt'],
            'Response': r['response'],
            'Quality Score': r['metrics']['quality_score'],
            'Tokens': r['metrics']['tokens'],
            'Cost': r['metrics']['cost'],
            'Latency': r['metrics']['latency'],
            'Reading Ease': r['metrics']['reading_ease'],
            'Grade Level': r['metrics']['grade_level']
        }
        for r in st.session_state.results
    ])
    
    # Convert DataFrame to CSV
    csv = results_df.to_csv(index=False).encode('utf-8')
    
    # Center the download button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.download_button(
            label='📥 Download Results',
            data=csv,
            file_name='llm_comparison_results.csv',
            mime='text/csv',
            use_container_width=True,
            key='download_button'  # Add unique key to prevent recreation
        )
        
        # Add clear results button
        if st.button('Clear Results', use_container_width=True):
            st.session_state.results = []
            st.experimental_rerun() 