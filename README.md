# My Notebook Project

This project is designed to train control vectors, generate text with personality traits, collect responses, enrich them with analysis, and visualize the results.

## Setup Instructions

1. **Install Dependencies:**
   Ensure you have `uv` installed, then sync the project dependencies:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync
   ```

2. Get nltk.words:
   ```bash
   uv run setup.py
   ```

3. Authenticate with huggingface token, alternatively set your "HF_TOKEN", environment variable

   ```bash
   huggingface-cli login --token <token>
   ```

4. Set your OPENAI_API_KEY environment variable
   ```bash
   # This is how it is used in code, this is used for judging llm resopnses for metrics like clarity and engagement
   OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```
