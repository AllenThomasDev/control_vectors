from nltk.corpus import words
from openai import OpenAI
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import argparse
from models import load_model
from control_vectors import train_control_vector, load_control_vector
from constants import AI_SUFFIXES, INTROVERSION_SUFFIXES, OPENAI_API_KEY, MODEL_CONFIGS

# Load the NLTK words corpus (assumes it's available via setup.py)
english_words = set(words.words())

# Initialize objects
analyzer = SentimentIntensityAnalyzer()
client = OpenAI(api_key=OPENAI_API_KEY)


def main():
    parser = argparse.ArgumentParser(description="Run the project with a chosen model.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["hermes", "deepseek"],
        default="hermes",
        help="Model to use: 'hermes' or 'deepseek' (default: hermes)",
    )
    args = parser.parse_args()

    # Load the model and tokenizer
    model, tokenizer = load_model(args.model)
    model_key = args.model.lower()

    # Train control vectors
    train_control_vector(
        model,
        tokenizer,
        AI_SUFFIXES,
        ai_template,
        "AI_Optimist",
        "AI_Doomer",
        model_key,
    )
    train_control_vector(
        model,
        tokenizer,
        INTROVERSION_SUFFIXES,
        social_template,
        "introvert",
        "extrovert",
        model_key,
    )

    # Load the trained vectors
    ai_control_vector = load_control_vector(model_key, "AI_Optimist", "AI_Doomer")
    introvert_control_vector = load_control_vector(model_key, "introvert", "extrovert")

    print(f"Setup and training complete with model: {MODEL_CONFIGS[model_key]['name']}")


if __name__ == "__main__":
    # Delayed import to avoid circular issues
    from control_vectors import ai_template, social_template

    main()
