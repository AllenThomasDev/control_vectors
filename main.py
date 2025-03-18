import nltk
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
        from analysis.personality_analysis import process_personality_results
        from analysis.llm_evaluation import judge_responses_multiple_metrics

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
        from analysis.visualization import (
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
