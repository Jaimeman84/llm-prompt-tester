# LLM Prompt Tester

A Streamlit application for testing and comparing responses from different Language Learning Models (LLMs) including OpenAI, Anthropic, and Mistral.

<div align="center">
  <p>
    <strong>Compare, Analyze, and Evaluate LLM Responses</strong>
  </p>
  <p>
    <a href="#features">Features</a> •
    <a href="#setup">Setup</a> •
    <a href="#usage">Usage</a> •
    <a href="#metrics-explained">Metrics</a> •
    <a href="#contributing">Contributing</a>
  </p>
</div>

---

## Features

- **Multiple LLM Provider Support**
  - OpenAI (GPT-4, GPT-4-Turbo, GPT-3.5-Turbo, GPT-3.5-Turbo-16k)
  - Anthropic (Claude-3-Opus, Claude-3-Sonnet, Claude-3-Haiku)
  - Mistral (Mistral-Small, Pixtral-12B, Open-Mistral-Nemo)

- **Generation Parameters**
  - Adjustable temperature (0.0 - 1.0)
  - Configurable max tokens (100 - 2000)

- **Side-by-Side Comparison**
  - Real-time response generation
  - Parallel processing of requests
  - Detailed metrics for each response:
    - Quality Score (with color indicators)
    - Tokens Used
    - Cost
    - Latency
    - Reading Ease
    - Grade Level

- **Results Management**
  - Export results to CSV
  - Persistent results storage
  - Clear results option
  - Download comparison data

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
MISTRAL_API_KEY=your_mistral_key
```

4. Run the application:
```bash
streamlit run src/app.py
```

## Usage

1. Select the LLM providers you want to test
2. Choose the specific model for each provider
3. Adjust the temperature and max tokens as needed
4. Enter your prompt in the left textarea
5. (Optional) Enter expected output in the right textarea for quality comparison
6. Click "Generate Responses" to see the results
7. Download results using the "Download Results" button
8. Clear results history with the "Clear Results" button when needed

## Features in Detail

### Provider Selection
- Each provider shows model-specific information:
  - Cost per 1K tokens
  - Maximum token limit
  - Typical latency

### Response Display
- Responses are shown in equal-width columns
- Each response includes:
  - Model name and provider
  - Generated text with proper formatting
  - Comprehensive metrics table

### Quality Assessment
- Quality scores are color-coded:
  - Green: ≥ 0.8 (Excellent)
  - Orange: ≥ 0.6 (Good)
  - Red: < 0.6 (Needs Improvement)

### Metrics Explained

#### Quality Score
- A composite score (0.0 - 1.0) that evaluates the response quality
- When reference text is provided:
  - Semantic similarity between response and reference (using embeddings)
  - Content overlap and key point coverage
  - Structural similarity
- Without reference text:
  - Response coherence and completeness
  - Grammar and formatting quality
  - Task relevance

#### Tokens Used
- Number of tokens consumed in the response
- Tokens are subword units used by LLMs
- Calculated using model-specific tokenizers:
  - OpenAI: tiktoken
  - Anthropic: Claude tokenizer
  - Mistral: SentencePiece

#### Cost
- Actual cost in USD for the API call
- Calculated as: (Input tokens + Output tokens) × (Cost per 1K tokens ÷ 1000)
- Different rates for each model:
  - GPT-4: $0.03/1K tokens input, $0.06/1K tokens output
  - Claude-3-Opus: $0.015/1K tokens input, $0.075/1K tokens output
  - Mistral-Small: $0.0002/1K tokens input+output

#### Latency
- Time taken (in seconds) from request start to response completion
- Includes:
  - API request transmission time
  - Model processing time
  - Network latency
  - Response reception time

#### Reading Ease (Flesch Reading Ease)
- Score from 0-100 indicating text readability
- Higher scores mean easier to read
- Formula: 206.835 - 1.015(total words/total sentences) - 84.6(total syllables/total words)
- Score interpretation:
  - 90-100: Very easy to read
  - 60-70: Standard/conversational English
  - 0-30: Very difficult to read

#### Grade Level (Flesch-Kincaid)
- Estimates the US grade level needed to understand the text
- Formula: 0.39(total words/total sentences) + 11.8(total syllables/total words) - 15.59
- Examples:
  - Score 8.0: Eighth grade reading level
  - Score 12.0: High school graduate level
  - Score 16.0: College graduate level

### Data Export
- CSV export includes:
  - Provider and model information
  - Original prompt and response
  - All metrics and scores
  - Timestamps for analysis

## Contributing

Feel free to submit issues and enhancement requests! 