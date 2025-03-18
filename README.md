# My Notebook Project

This project is designed to train control vectors, generate text with personality traits, collect responses, enrich them with analysis, and visualize the results. It supports a full workflow—from training control vectors and generating text, to analyzing and visualizing the outputs.

## Setup Instructions

1.  **Install Dependencies:**

    Ensure you have `uv` installed, then sync the project dependencies:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv sync
    ```

2.  **Download NLTK Words Corpus:**

    Run the setup script to download the NLTK words corpus:

    ```bash
    uv run setup.py
    ```

3.  **Authenticate with Hugging Face:**

    Either set your `HF_TOKEN` environment variable or run:

    ```bash
    huggingface-cli login --token <token>
    ```

4.  **Set OpenAI API Key:**

    Set your `OPENAI_API_KEY` environment variable (used for judging LLM responses):

    ```bash
    export OPENAI_API_KEY=<your_openai_api_key>
    ```

5.  **Configure Model and Vectors:**

    Edit the `constants.py` file to select your model and configure control vectors. For example:

    ```python
    MODEL_CONFIGS = {
        "hermes": {
            "name": "NousResearch/Hermes-3-Llama-3.2-3B",
            # other model settings...
        },
        # add more models if needed
    }
    STRENGTH_RANGES = (-2, 2, 0.5)
    ```

## Commands & Usage

The project’s main functionality is accessed via the `main.py` script. The available command-line arguments allow you to train control vectors, generate text, collect multiple responses, enrich analysis, and plot results.

1.  **Training a Control Vector**

    Train a specific control vector (choose either `ai` or `introvert`):

    ```bash
    uv run python main.py --model hermes --train ai
    ```

    or

    ```bash
    uv run python main.py --model hermes --train introvert
    ```

    This command uses configurations (like suffixes and templates) defined in `constants.py`.

2.  **Generating a Single Response**

    Generate text using a control vector. This mode requires you to specify the vector to use with the `--vector` flag:

    ```bash
    uv run python main.py --model hermes --generate --vector introvert --input "How do you feel in social settings?"
    ```

    Replace `introvert` with `ai` to use the AI personality control vector.

3.  **Collecting Responses (CSV Generation)**

    Collect responses for every combination of control vector strengths and save them to a CSV file. This is useful for comparing how different strengths affect the generated text.

    For a single vector:

    ```bash
    uv run python main.py --model hermes --collect --vectors ai
    ```

    For multiple vectors (e.g., both `ai` and `introvert`):

    ```bash
    uv run python main.py --model hermes --collect --vectors ai introvert --csv my_responses.csv
    ```

    The CSV filename defaults to `responses.csv` if not specified.

4.  **Enriching Collected Responses**

    Enhance your collected CSV with personality analysis and LLM scores. This step processes the responses and adds additional metrics:

    ```bash
    uv run python main.py --enrich responses.csv
    ```

    You can also specify an output prefix for the resulting files:

    ```bash
    uv run python main.py --enrich responses.csv --output-prefix my_project_
    ```

    This process produces enriched files such as `personality_results.csv` and `llm_scores.csv` (or with your specified prefix).

5.  **Plotting Results**

    Generate visualizations based on the enriched CSV file. The plotting command creates various plots (heatmaps, metrics grids, sentiment trends, etc.):

    ```bash
    uv run python main.py --plot llm_scores.csv
    ```

    To specify an output prefix for the plots:

    ```bash
    uv run python main.py --plot llm_scores.csv --output-prefix my_project_
    ```

## Command-Line Arguments Overview

*   `--model`
    Choose the model to use. Options are defined in `constants.py` (default is the first key in `MODEL_CONFIGS`).

*   `--vectors`
    Specify which control vectors to use for collection. Options: `ai`, `introvert`, or both (default: both).

*   `--train`
    Train a specific control vector. Accepted values: `ai` or `introvert`.

*   `--generate`
    Generate a single response using a control vector. Requires `--vector`.

*   `--vector`
    Specify which control vector to use for generation (accepted values: `ai` or `introvert`).

*   `--collect`
    Collect responses across all combinations of vector strengths and save to a CSV file.

*   `--input`
    Provide an input prompt for generation or collection. Defaults to `BASE_QUESTION` from `constants.py`.

*   `--csv`
    Specify the filename for saving collected responses (default: `responses.csv`).

*   `--enrich`
    Enrich an existing CSV file with personality analysis and LLM scores. Provide the CSV filename.

*   `--plot`
    Generate visualizations based on an enriched CSV file. Provide the CSV filename.

*   `--output-prefix`
    Optionally, set a prefix for all output files (e.g., `my_project_`).

## Project Structure

*   `main.py`
    Main script for training, generating, collecting, enriching, and plotting.
*   `constants.py`
    Configuration file for models, control vectors, and default parameters.
*   `control_vectors.py`
    Contains functions for training and generating text using control vectors.
*   `graph_utils.py`
    Utility functions for processing CSV results and generating plots.
*   `setup.py`
    Setup script to download necessary resources (e.g., the NLTK words corpus).

