# main.py
import nltk
from nltk.corpus import words
from openai import OpenAI
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import argparse
from models import load_model
from control_vectors import (
    train_control_vector,
    load_control_vector,
    generate_with_vector,
    collect_responses_with_vectors,
)
from constants import (
    AI_SUFFIXES,
    INTROVERSION_SUFFIXES,
    OPENAI_API_KEY,
    STRENGTH_RANGES,
    MODEL_CONFIGS,
    BASE_QUESTION,
)

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
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate text using a control vector instead of training (requires --vector)",
    )
    parser.add_argument(
        "--vector",
        type=str,
        choices=["ai", "introvert"],
        help="Vector to use for generation: 'ai' (AI_Optimist vs. AI_Doomer) or 'introvert' (introvert vs. extrovert)",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect responses with all vector strength combinations and save to CSV",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=BASE_QUESTION,
        help="Input prompt for generation or collection (default: BASE_QUESTION from constants.py)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="responses.csv",
        help="Filename for saving collected responses (default: responses.csv)",
    )
    args = parser.parse_args()

    # Load the model and tokenizer
    model, tokenizer = load_model(args.model)
    model_key = args.model.lower()

    if args.generate:
        # Generation mode
        if not args.vector:
            parser.error("--generate requires --vector to specify which vector to use.")

        vector_map = {
            "ai": ("AI_Optimist", "AI_Doomer"),
            "introvert": ("introvert", "extrovert"),
        }
        positive_persona, negative_persona = vector_map[args.vector]
        control_vector = load_control_vector(
            model_key, positive_persona, negative_persona
        )

        if control_vector:
            print(f"\nGenerating with {args.vector} control vector:")
            output = generate_with_vector(
                model, tokenizer, args.input, control_vector * 1
            )
            print(output)
        else:
            print(
                f"Cannot generate: No vector found for {args.vector} with model {model_key}"
            )

        print(f"Generation complete with model: {MODEL_CONFIGS[model_key]['name']}")

    elif args.collect:
        # Collection mode
        ai_control_vector = load_control_vector(model_key, "AI_Optimist", "AI_Doomer")
        introvert_control_vector = load_control_vector(
            model_key, "introvert", "extrovert"
        )

        if not ai_control_vector or not introvert_control_vector:
            print("Cannot collect responses: Both vectors must be available.")
        else:
            vectors = [ai_control_vector, introvert_control_vector]
            vector_names = ["AI_Optimist_vs_AI_Doomer", "introvert_vs_extrovert"]
            collect_responses_with_vectors(
                args.input,
                model,
                tokenizer,
                vectors,
                vector_names,
                strength_range=STRENGTH_RANGES,
                max_new_tokens=256,
                save_to_csv=args.csv,
            )
        print(f"Collection complete with model: {MODEL_CONFIGS[model_key]['name']}")

    else:
        # Training and default data collection mode
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

        ai_control_vector = load_control_vector(model_key, "AI_Optimist", "AI_Doomer")
        introvert_control_vector = load_control_vector(
            model_key, "introvert", "extrovert"
        )

        if ai_control_vector:
            print("\nGenerating with AI Optimist control vector:")
            output = generate_with_vector(
                model, tokenizer, BASE_QUESTION, ai_control_vector * 1
            )
            print(output)

        print(
            f"Setup, training, and data collection complete with model: {MODEL_CONFIGS[model_key]['name']}"
        )


if __name__ == "__main__":
    # Delayed import to avoid circular issues
    from control_vectors import ai_template, social_template

    main()
