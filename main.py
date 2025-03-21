from steering_utils import convert_control_vector_to_steer_config
import os
import pandas as pd
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
    OPENAI_API_KEY,
    STRENGTH_RANGES,
    MODEL_CONFIGS,
    BASE_QUESTION,
    CONTROL_VECTOR_CONFIGS,  # Add this, remove AI_SUFFIXES and INTROVERSION_SUFFIXES
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
        choices=list(MODEL_CONFIGS.keys()),
        default=list(MODEL_CONFIGS.keys())[0],
        help="Model to use",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect responses with all vector strength combinations and save to CSV",
    )
    parser.add_argument(
        "--vectors",
        nargs="+",
        choices=list(CONTROL_VECTOR_CONFIGS.keys()),  # Use config keys
        default=list(CONTROL_VECTOR_CONFIGS.keys()),
        help="Vectors to use for collection",
    )
    parser.add_argument(
        "--train",
        type=str,
        choices=list(CONTROL_VECTOR_CONFIGS.keys()),  # Use config keys
        help="Train a specific control vector",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate text using a control vector instead of training (requires --vector)",
    )
    parser.add_argument(
        "--vector",
        type=str,
        choices=list(CONTROL_VECTOR_CONFIGS.keys()),
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
    parser.add_argument(
        "--enrich",
        type=str,
        help="Enrich the CSV file with personality analysis and LLM scores",
    )
    parser.add_argument(
        "--plot",
        type=str,
        help="Generate plots based on the enriched CSV file",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Prefix for output files (default: empty string)",
    )
    args = parser.parse_args()

    # New enrichment function
    if args.enrich:
        from graph_utils import process_personality_results
        from graph_utils import judge_responses_multiple_metrics

        print(f"Loading data from {args.enrich}...")
        results = pd.read_csv(args.enrich)

        print("Processing personality results...")
        df = process_personality_results(results, args.input or BASE_QUESTION)

        # Save intermediate results
        personality_output = (
            f"{args.output_prefix}personality_results.csv"
            if args.output_prefix
            else "personality_results.csv"
        )
        df.to_csv(personality_output)
        print(f"Saved personality results to {personality_output}")

        print("Judging responses with LLM metrics...")
        judged_df = judge_responses_multiple_metrics(df)

        # Save final enriched results
        llm_output = (
            f"{args.output_prefix}llm_scores.csv"
            if args.output_prefix
            else "llm_scores.csv"
        )
        judged_df.to_csv(llm_output)
        print(f"Saved LLM scores to {llm_output}")

        print("Enrichment complete!")
        return

    # New plotting function
    if args.plot:
        from graph_utils import (
            create_heatmap,
            response_length,
            short_word_ratio,
            unigram_repetition,
            plot_strength_vs_metrics_grid,
            plot_sentiment_area_trends,
            create_sentiment_heatmaps,
            plot_control_vs_llm_scores,
        )

        print(f"Loading data from {args.plot}...")
        df = pd.read_csv(args.plot)

        print("Generating plots...")

        # Basic filtering
        subset_df = df[
            (df["vector1_strength"].between(-1, 1, inclusive="both"))
            & (df["vector2_strength"].between(-1, 1, inclusive="both"))
        ]

        # Create heatmaps
        print("Creating personality trait heatmaps...")
        heatmaps = create_heatmap(subset_df)
        for trait, fig in heatmaps.items():
            output_file = (
                f"{args.output_prefix}heatmap_{trait}.html"
                if args.output_prefix
                else f"heatmap_{trait}.html"
            )
            fig.write_html(output_file)
            print(f"Saved {trait} heatmap to {output_file}")

        # Create metrics grid
        print("Creating metrics grid...")
        metric_funcs = [
            (response_length, "Response Length"),
            (short_word_ratio, "Short Word Ratio"),
            (unigram_repetition, "Unigram Repetition"),
        ]
        vectors = [("vector1_strength", "AI"), ("vector2_strength", "Introversion")]
        metrics_fig = plot_strength_vs_metrics_grid(
            df=df,
            metric_funcs=metric_funcs,
            vectors=vectors,
            response_col="cleaned_response",
        )
        metrics_output = (
            f"{args.output_prefix}metrics_grid.html"
            if args.output_prefix
            else "metrics_grid.html"
        )
        metrics_fig.write_html(metrics_output)
        print(f"Saved metrics grid to {metrics_output}")

        # Create sentiment area trends
        print("Creating sentiment area trends...")
        # Example for introversion (ai = 0)
        fig_intro = plot_sentiment_area_trends(
            df,
            target_vector="vector2_strength",
            fixed_vector="vector1_strength",
            fixed_value=0,
            title_prefix="Introversion",
        )
        if fig_intro:
            intro_output = (
                f"{args.output_prefix}introversion_sentiment.html"
                if args.output_prefix
                else "introversion_sentiment.html"
            )
            fig_intro.write_html(intro_output)
            print(f"Saved introversion sentiment plot to {intro_output}")

        # Example for ai (introversion = 0)
        fig_ai = plot_sentiment_area_trends(
            df,
            target_vector="vector1_strength",
            fixed_vector="vector2_strength",
            fixed_value=0,
            title_prefix="AI",
        )
        if fig_ai:
            ai_output = (
                f"{args.output_prefix}ai_sentiment.html"
                if args.output_prefix
                else "ai_sentiment.html"
            )
            fig_ai.write_html(ai_output)
            print(f"Saved AI sentiment plot to {ai_output}")

        # Create sentiment heatmaps
        print("Creating sentiment heatmaps...")
        fig = create_sentiment_heatmaps(df)
        sentiment_output = (
            f"{args.output_prefix}sentiment_heatmaps.html"
            if args.output_prefix
            else "sentiment_heatmaps.html"
        )
        fig.write_html(sentiment_output)
        print(f"Saved sentiment heatmaps to {sentiment_output}")

        # Plot control vs LLM scores
        if "llm_clarity_score" in df.columns:
            print("Creating control vs LLM scores plot...")
            llm_fig = plot_control_vs_llm_scores(
                df,
                control_vectors=["vector1_strength", "vector2_strength"],
                llm_scores=[
                    "llm_clarity_score",
                    "llm_engagement_score",
                    "llm_relevance_score",
                ],
                width=1200,
                height=800,
            )
            llm_output = (
                f"{args.output_prefix}llm_scores.html"
                if args.output_prefix
                else "llm_scores.html"
            )
            llm_fig.write_html(llm_output)
            print(f"Saved LLM scores plot to {llm_output}")

        print("Plotting complete!")
        return

    # Load the model and tokenizer
    model, tokenizer = load_model(args.model)
    model_key = args.model.lower()

    if args.generate:
        if not args.vector:
            parser.error("--generate requires --vector to specify which vector to use.")
        config = CONTROL_VECTOR_CONFIGS[args.vector]
        control_vector = load_control_vector(
            model_key, config["positive_persona"], config["negative_persona"]
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
    elif args.convert:
        config = CONTROL_VECTOR_CONFIGS[args.convert]
        control_vector = load_control_vector(
            model_key, config["positive_persona"], config["negative_persona"]
        )
        if control_vector:
            output_path = f"steer_configs/{args.convert}_steer_config.pt"
            os.makedirs("steer_configs", exist_ok=True)
            convert_control_vector_to_steer_config(
                control_vector.directions, output_path
            )
            print(f"Converted {args.convert} control vector to {output_path}")
        else:
            print(
                f"Cannot convert: No vector found for {args.convert} with model {model_key}"
            )
        return
    elif args.collect:
        vectors = {}
        for vec in args.vectors:
            config = CONTROL_VECTOR_CONFIGS[vec]
            control_vector = load_control_vector(
                model_key, config["positive_persona"], config["negative_persona"]
            )
            if control_vector:
                vectors[config["vector_name"]] = control_vector
            else:
                print(
                    f"Warning: No vector found for {vec} with model {model_key}. Skipping."
                )

        if not vectors:
            print("Cannot collect responses: No valid vectors available.")
        else:
            print(f"Collecting responses with vectors: {', '.join(args.vectors)}")
            collect_responses_with_vectors(
                args.input,
                model,
                tokenizer,
                vectors,
                strength_range=STRENGTH_RANGES,
                max_new_tokens=256,
                save_to_csv=args.csv,
            )
        print(f"Collection complete with model: {MODEL_CONFIGS[model_key]['name']}")
    elif args.collect:
        vectors = {}
        for vec in args.vectors:
            config = CONTROL_VECTOR_CONFIGS[vec]
            control_vector = load_control_vector(
                model_key, config["positive_persona"], config["negative_persona"]
            )
            if control_vector:
                vectors[config["vector_name"]] = control_vector
            else:
                print(
                    f"Warning: No vector found for {vec} with model {model_key}. Skipping."
                )

        if not vectors:
            print("Cannot collect responses: No valid vectors available.")
        else:
            print(f"Collecting responses with vectors: {', '.join(args.vectors)}")
            collect_responses_with_vectors(
                args.input,
                model,
                tokenizer,
                vectors,
                strength_range=STRENGTH_RANGES,
                max_new_tokens=256,
                save_to_csv=args.csv,
            )
        print(f"Collection complete with model: {MODEL_CONFIGS[model_key]['name']}")

    elif args.train:  # Changed from if to elif
        config = CONTROL_VECTOR_CONFIGS[args.train]
        print(f"Training {args.train} control vector...")
        train_control_vector(
            model,
            tokenizer,
            config["suffixes"],
            config["template"],
            config["positive_persona"],
            config["negative_persona"],
            model_key,
        )
        print(
            f"Training complete for {args.train} with model: {MODEL_CONFIGS[model_key]['name']}"
        )
        return


if __name__ == "__main__":
    main()
