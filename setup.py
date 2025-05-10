from setuptools import setup, find_packages

setup(
    name="llm-prompt-tester",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "openai",
        "anthropic",
        "mistralai",
        "python-dotenv",
        "pandas",
        "pytest",
        "pytest-asyncio",
    ],
    python_requires=">=3.8",
) 