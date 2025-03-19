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

5.  **Convert control vector to be used with lm_eval_harness**
    Converts the ai control vector to steer_configs/ai_steer_config.pt for use with lm-evaluation-harness.

    ```bash
    uv run python main.py --model hermes --convert ai

    #use it like - 
    lm_eval --model steered \
    --model_args pretrained=NousResearch/Hermes-3-Llama-3.2-3B,steer_path=steer_configs/ai_steer_config.pt \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --batch_size 8

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


# How to train your own control vectors - 

Here’s a step-by-step guide:
Step 1: Open constants.py

Step 2: Define Your Suffixes
Suffixes are partial sentences that the model will complete differently based on the persona (e.g., "happy" or "sad"). Create a list of these prompts that are neutral enough to work with both personas. Add this suffix list to the config file

Step 3: Create a Template Function
The template function combines a persona (e.g., "happy" or "sad") with a suffix to form a complete prompt for training. You’ll define this in control_vectors.py. Open that file and add a new function:
```python
def happiness_template(persona, suffix):
    return f"As a {persona} person, {suffix}"
```
Step 4: Add Your Config to CONTROL_VECTOR_CONFIGS
Back in constants.py, scroll to the CONTROL_VECTOR_CONFIGS dictionary. Add a new entry for your vector. The key (e.g., "happiness") is what you’ll use in commands like --train happiness. Here’s how to add it:
```python
from control_vectors import happiness_template  # Add this import

CONTROL_VECTOR_CONFIGS = {
    "ai": {
        "suffixes": AI_SUFFIXES,
        "template": ai_template,
        "positive_persona": "AI_Optimist",
        "negative_persona": "AI_Doomer",
        "vector_name": "AI_Optimist_vs_AI_Doomer",
    },
    "introvert": {
        "suffixes": INTROVERSION_SUFFIXES,
        "template": social_template,
        "positive_persona": "introvert",
        "negative_persona": "extrovert",
        "vector_name": "introvert_vs_extrovert",
    },
    "happiness": {  # Your new vector
        "suffixes": HAPPINESS_SUFFIXES,
        "template": happiness_template,
        "positive_persona": "happy",
        "negative_persona": "sad",
        "vector_name": "happy_vs_sad"
    },
}
```

Step 5: Train Your Vector
Run the training command with your new vector key:
```bash
uv run python main.py --model hermes --train happiness
```

You can also run an chat interface based on your control vector - 
```bash
uv run python chat_interface.py
```

pick the vectors to be used in the same file - 
these are the defaults - 
```python
control_vectors = {
    "ai_optimism": load_control_vector(
        model_key,
        CONTROL_VECTOR_CONFIGS["ai"]["positive_persona"],
        CONTROL_VECTOR_CONFIGS["ai"]["negative_persona"],
    ),
    "introversion": load_control_vector(
        model_key,
        CONTROL_VECTOR_CONFIGS["introvert"]["positive_persona"],
        CONTROL_VECTOR_CONFIGS["introvert"]["negative_persona"],
    ),
}

```
