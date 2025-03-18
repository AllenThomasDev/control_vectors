import re
from collections import Counter
from constants import ENGLISH_WORDS, OPENAI_API_KEY
from tqdm import tqdm
import openai
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


def sentiment_compound(text):
    """
    Compute the compound sentiment score for a given text using VADER.

    Parameters:
    text (str): The input text for sentiment analysis.

    Returns:
    float: Compound sentiment score.
    """
    return analyzer.polarity_scores(text)["compound"]


def sentiment_proportions(text):
    """
    Compute the proportions of positive, neutral, and negative sentiment for a given text.

    Parameters:
    text (str): The input text for sentiment analysis.

    Returns:
    dict: Dictionary containing positive, neutral, and negative sentiment proportions.
    """
    scores = analyzer.polarity_scores(text)
    return {
        "Positive": scores["pos"],
        "Neutral": scores["neu"],
        "Negative": scores["neg"],
    }


def create_heatmap(df, traits=None):
    """
    Parameters:
    df (DataFrame): DataFrame containing personality traits and control variables
    traits (list): List of trait names to visualize. If None, uses default traits.

    Returns:
    dict: Dictionary of plotly figures, keyed by trait name
    """
    if traits is None:
        traits = [
            "Extroversion",
            "Neuroticism",
            "Conscientiousness",
            "Agreeableness",
            "Openness",
        ]

    figs = {}
    for trait in traits:
        # Create a pivot table
        pivot = df.pivot_table(
            index="vector1_strength", columns="vector2_strength", values=trait
        )

        # Create heatmap
        fig = px.imshow(
            pivot,
            x=pivot.columns,
            y=pivot.index,
            color_continuous_scale="RdBu_r",
            labels=dict(
                x="vector2_strength (introversion)",
                y="vector1_strength (ai)",
                color=trait,
            ),
            title=f"{trait} Change by Control Variables",
        )

        fig.update_layout(
            width=700,
            height=600,
            xaxis_title="Introversion Vector Strength",
            yaxis_title="AI Optimism Vector Strength",
        )

        # Customize hover information
        fig.update_traces(
            hovertemplate="AI strength: %{y}<br>Introversion strength: %{x}<br>Value: %{z:.4f}<extra></extra>"
        )

        # Add text annotations
        for i, row in enumerate(pivot.index):
            for j, col in enumerate(pivot.columns):
                fig.add_annotation(
                    x=col,
                    y=row,
                    text=f"{pivot.iloc[i, j]:.3f}",
                    showarrow=False,
                    font=dict(
                        color="white" if abs(pivot.iloc[i, j]) > 0.2 else "black"
                    ),
                )

        figs[trait] = fig
    return figs


def plot_sentiment_area_trends(
    df,
    target_vector,
    fixed_vector,
    fixed_value=0,
    response_col="cleaned_response",
    title_prefix="",
):
    """
    Plot an enhanced stacked area chart of raw sentiment proportions for a single control vector,
    emphasizing Positive and Negative changes.

    Parameters:
    df (DataFrame): DataFrame with vector1_strength, vector2_strength, and response_col
    target_vector (str): 'vector1_strength' (ai) or 'vector2_strength' (introversion) to vary
    fixed_vector (str): The other vector to hold at fixed_value
    fixed_value (float): Value to fix the non-target vector (default 0)
    response_col (str): Column name for text responses
    title_prefix (str): Prefix for plot title (e.g., "AI" or "Introversion")

    Returns:
    plotly.graph_objs._figure.Figure: Enhanced stacked area plot of sentiment proportions
    """
    # Calculate sentiment proportions
    proportions = df[response_col].apply(sentiment_proportions)
    df["positive"] = proportions.apply(lambda x: x["Positive"])
    df["neutral"] = proportions.apply(lambda x: x["Neutral"])
    df["negative"] = proportions.apply(lambda x: x["Negative"])

    # Sentiment metrics
    metrics = ["positive", "neutral", "negative"]
    metric_names = ["Positive", "Neutral", "Negative"]

    # Filter data where fixed_vector is at fixed_value
    subset = df[df[fixed_vector] == fixed_value].sort_values(target_vector)
    if subset.empty:
        print(f"No data found where {fixed_vector} = {fixed_value}")
        return None

    # Create figure
    fig = go.Figure()

    # Enhanced colors with slight transparency for overlap visibility
    colors = [
        "rgba(0, 200, 0, 0.7)",  # Vibrant green for Positive
        "rgba(150, 150, 150, 0.5)",  # Subtle gray for Neutral
        "rgba(255, 0, 0, 0.7)",  # Vibrant red for Negative
    ]

    # Plot each sentiment metric as a stacked area
    for metric, name, color in zip(metrics, metric_names, colors):
        fig.add_trace(
            go.Scatter(
                x=subset[target_vector],
                y=subset[metric],
                mode="lines",  # Lines for smooth stacking
                name=name,
                stackgroup="one",
                line=dict(width=0),  # No outline
                fillcolor=color,
                hovertemplate=f"{name}: %{{y:.3f}}<br>{title_prefix} Strength: %{{x:.2f}}",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Sentiment Proportions by {title_prefix} Vector ({fixed_vector} = {fixed_value})",
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        xaxis_title=f"{title_prefix} Vector Strength",
        yaxis_title="Sentiment Proportion",
        legend=dict(
            title="Sentiment",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        height=600,
        width=1000,
        hovermode="x unified",
        yaxis_range=[0, 1],  # Fixed range for proportions
        xaxis=dict(
            gridcolor="lightgray", zeroline=True, zerolinecolor="black", zerolinewidth=1
        ),
        yaxis=dict(gridcolor="lightgray", tickvals=[0, 0.25, 0.5, 0.75, 1]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=50, t=100, b=50),
    )

    return fig


def create_sentiment_heatmaps(df, response_col="cleaned_response"):
    """
    Create horizontal heatmaps showing normalized sentiment changes with control vector strengths.

    Parameters:
    df (DataFrame): DataFrame with vector1_strength, vector2_strength, and response_col
    response_col (str): Column name containing text responses for sentiment analysis

    Returns:
    plotly.graph_objs._figure.Figure: Figure with horizontal sentiment heatmaps
    """
    # Calculate sentiment scores
    df["compound"] = df[response_col].apply(sentiment_compound)
    proportions = df[response_col].apply(sentiment_proportions)
    df["positive"] = proportions.apply(lambda x: x["Positive"])
    df["neutral"] = proportions.apply(lambda x: x["Neutral"])
    df["negative"] = proportions.apply(lambda x: x["Negative"])

    # Sentiment metrics and titles
    metrics = ["compound", "positive", "neutral", "negative"]
    titles = [
        "Compound Sentiment",
        "Positive Proportion",
        "Neutral Proportion",
        "Negative Proportion",
    ]

    # Find baseline values at (0, 0)
    baseline = df[(df["vector1_strength"] == 0) & (df["vector2_strength"] == 0)]
    if baseline.empty:
        print("Warning: No baseline data at (0, 0). Using raw values instead.")
        for metric in metrics:
            df[f"{metric}_norm"] = df[metric]
    else:
        baseline_values = {
            "compound": baseline["compound"].iloc[0],
            "positive": baseline["positive"].iloc[0],
            "neutral": baseline["neutral"].iloc[0],
            "negative": baseline["negative"].iloc[0],
        }
        # Normalize by subtracting baseline
        for metric in metrics:
            df[f"{metric}_norm"] = df[metric] - baseline_values[metric]

    # Create subplots horizontally (1 row, 4 columns)
    fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=[f"{title} (Normalized)" for title in titles],
        horizontal_spacing=0.05,
    )

    # Add heatmaps
    for i, metric in enumerate(metrics):
        pivot = df.pivot_table(
            index="vector1_strength",
            columns="vector2_strength",
            values=f"{metric}_norm",
        )
        heatmap = go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Viridis",
            showscale=False,  # Each heatmap gets its own colorbar
            colorbar=dict(
                title=titles[i],
                len=0.8,  # Adjust colorbar length
                x=1 + i * 0.25,  # Position colorbars to avoid overlap
            ),
            hovertemplate=f"AI strength: %{{y}}<br>Introversion strength: %{{x}}<br>{titles[i]}: %{{z:.4f}}",
        )
        fig.add_trace(heatmap, row=1, col=i + 1)

        # Axes titles (only on first and last for clarity)
        if i == 0:
            fig.update_yaxes(title_text="vector1_strength (ai)", row=1, col=i + 1)
        fig.update_xaxes(title_text="vector2_strength (introversion)", row=1, col=i + 1)

        # Enforce square aspect ratio
        fig.update_yaxes(scaleanchor=f"x{i + 1}", scaleratio=1, row=1, col=i + 1)

    # Update layout
    subplot_size = 400  # Smaller size for horizontal layout
    fig.update_layout(
        height=subplot_size + 100,  # Height for one row plus padding
        width=subplot_size * 4 + 200,  # Width for 4 subplots plus padding
        title_text="Normalized Sentiment Analysis by Control Vector Strengths",
    )

    return fig


def plot_strength_vs_metrics_grid(
    df, metric_funcs, vectors, response_col="response", trend_degree=1
):
    """
    Plots control vector strengths vs. computed metrics in a 2x3 subplot grid using box plots.

    Parameters:
    - df: Pandas DataFrame containing vector strength and response columns.
    - metric_funcs: List of tuples [(func, name), ...] where func takes a response string and returns a numeric value, and name is the metric name.
    - vectors: List of tuples [(vector_col, vector_name), ...] specifying vector column names and their human-readable names.
    - response_col: String, column name containing responses (default: "response").
    """
    # Define subplot structure: 2 rows (vectors), 3 columns (metrics)
    n_rows = len(vectors)
    n_cols = len(metric_funcs)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[
            f"{vector[1]} vs. {metric[1]}"
            for vector in vectors
            for metric in metric_funcs
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    # Colors for consistency
    colorscale = "Viridis"

    # Populate subplots
    for i, (vector, vector_name) in enumerate(vectors, start=1):  # Rows
        strengths = df[vector].to_numpy()
        trend_x = np.linspace(min(strengths), max(strengths), 100)

        for j, (metric_func, metric_name) in enumerate(
            metric_funcs, start=1
        ):  # Columns
            metric_values = df[response_col].apply(metric_func).to_numpy()

            # Fit trend line
            trend_poly = np.polyfit(strengths, metric_values, deg=trend_degree)
            trend_line = np.poly1d(trend_poly)
            trend_y = trend_line(trend_x)

            # Add scatter
            fig.add_trace(
                go.Scatter(
                    x=strengths,
                    y=metric_values,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=metric_values,
                        colorscale=colorscale,
                        showscale=False,  # No colorbar to reduce clutter
                    ),
                    name=metric_name,
                    hovertemplate=f"{vector_name} Strength: %{{x}}<br>{metric_name}: %{{y:.3f}}",
                    # Legend only on first subplot
                    showlegend=(i == 1 and j == 1),
                ),
                row=i,
                col=j,
            )

            # Add trend line
            fig.add_trace(
                go.Scatter(
                    x=trend_x,
                    y=trend_y,
                    mode="lines",
                    line=dict(color="black", width=2, dash="dash"),
                    name=f"Trend (degree {trend_degree})",
                    hoverinfo="skip",
                    # Legend only on first subplot
                    showlegend=(i == 1 and j == 1),
                ),
                row=i,
                col=j,
            )

            # Update axes
            fig.update_xaxes(
                title_text=f"{vector_name} Vector Strength" if i == n_rows else "",
                tickmode="array",
                tickvals=np.unique(strengths),
                gridcolor="lightgray",
                row=i,
                col=j,
            )
            fig.update_yaxes(
                title_text=metric_name if i == 1 else "",
                zeroline=True,
                zerolinecolor="black",
                zerolinewidth=1,
                gridcolor="lightgray",
                row=i,
                col=j,
            )

    # Update layout
    fig.update_layout(
        title_text="Control Vector Strengths vs. Response Metrics",
        height=1000,  # Adjusted for 2 rows
        width=1500,  # Adjusted for 3 columns
        legend=dict(x=1.05, y=1, xanchor="left", yanchor="top"),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.show()


def response_length(response):
    # Ensure response is string before measuring length
    return len(str(response))


def short_word_ratio(response):
    """Computes the ratio of short words (<3 chars) in the response."""
    tokens = re.findall(r"\b\w+\b", response)  # Extract words robustly
    if not tokens:
        return 0
    short_words = [word for word in tokens if len(word) < 3]
    return len(short_words) / len(tokens)


def unigram_repetition(response):
    """Computes the fraction of the response covered by the top 3 most frequent words."""
    tokens = response.split()
    if not tokens:
        return 0
    word_counts = Counter(tokens)
    most_common_words = word_counts.most_common(3)
    return sum(count for _, count in most_common_words) / len(tokens)


def non_dict_ratio(response):
    """Computes the ratio of non-dictionary words in the response."""
    tokens = response.split()
    if not tokens:
        return 0
    non_dict_words = [word for word in tokens if word.lower() not in ENGLISH_WORDS]
    return len(non_dict_words) / len(tokens)


def plot_control_vs_llm_scores(
    df,
    control_vectors=["vector1_strength", "vector2_strength"],
    llm_scores=["llm_clarity_score", "llm_engagement_score", "llm_relevance_score"],
    width=1500,
    height=1000,
):
    """
    Generate a 2x3 subplot grid plotting control vector strengths against LLM scores using Plotly.

    Parameters:
    - df: pandas DataFrame containing the data
    - control_vectors: list of str, column names for control vectors (default: ['vector1_strength', 'vector2_strength'])
    - llm_scores: list of str, column names for LLM scores (default: ['llm_clarity_score', 'llm_engagement_score', 'llm_relevance_score'])
    - width: int, width of the figure in pixels (default: 1500)
    - height: int, height of the figure in pixels (default: 1000)

    Returns:
    - None (displays the Plotly figure)
    """
    # Create a 2x3 subplot figure
    df_copy = df.copy()

    # Wrap the 'cleaned_response' text and replace newlines with <br> for HTML hover rendering
    df_copy.cleaned_response = df_copy.cleaned_response.str.wrap(30)
    df_copy.cleaned_response = df_copy.cleaned_response.apply(
        lambda x: x.replace("\n", "<br>")
    )
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            f"{vector} vs. {llm_score}"
            for vector in control_vectors
            for llm_score in llm_scores
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Generate plots for each control vector
    for i, vector in enumerate(control_vectors):
        for j, llm_score in enumerate(llm_scores):
            row = i + 1  # Row 1 for vector1, Row 2 for vector2
            col = j + 1  # Columns 1, 2, 3 for clarity, engagement, relevance

            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=df_copy[vector],
                    y=df_copy[llm_score],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="blue" if vector == "vector1_strength" else "green",
                        opacity=0.6,
                    ),
                    name=f"{vector} vs. {llm_score}",
                    # Hover shows response text
                    text=df_copy["cleaned_response"],
                    hovertemplate="<b>Response</b>: %{text}<br>"
                    + f"<b>{vector}</b>: %{{x}}<br>"
                    + f"<b>{llm_score}</b>: %{{y}}<extra></extra>",
                ),
                row=row,
                col=col,
            )

            # Update axes labels
            fig.update_xaxes(title_text=vector, row=row, col=col)
            fig.update_yaxes(title_text=llm_score, row=row, col=col)

    # Update layout
    fig.update_layout(
        width=width,
        height=height,
        title_text="Control Vector Strengths vs. LLM Scores",
        showlegend=False,  # Legend not needed since each subplot is labeled
        template="plotly_white",  # Clean white background
    )

    # Show the plot
    fig.show()


results = pd.read_csv("multi_vector_responses_deepseek(2).csv")


def personality_detection(text):
    """Detect personality traits from text using a pre-trained BERT model."""
    tokenizer = BertTokenizer.from_pretrained("Minej/bert-base-personality")
    model = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality")

    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.squeeze().detach().numpy()

    label_names = [
        "Extroversion",
        "Neuroticism",
        "Agreeableness",
        "Conscientiousness",
        "Openness",
    ]
    return {label_names[i]: predictions[i] for i in range(len(label_names))}


def clean_text(text, input_question=""):
    """Remove tags and input question from response text."""
    tags = [
        "<|im_start|>user",
        "<|im_start|>assistant",
        "<|im_end|>",
        "<|im_start|>",
        "<|begin_of_text|>",
        "user\n",
        "assistant\n",
        "\n",
        "<｜User｜>",
        "<｜Assistant｜>",
        "<think>",
        "</think>",
    ]
    for tag in tags:
        text = text.replace(tag, "")
    text = text.replace(input_question, "")
    return text.strip()


def process_personality_results(results, base_question, output_csv="normalized.csv"):
    """
    Process responses to detect personality traits and normalize against baseline.
    Args:
        results (pd.DataFrame): DataFrame with 'response' column and vector strengths.
        base_question (str): Question to remove from responses.
        output_csv (str): Path to save the final normalized DataFrame.
    Returns:
        pd.DataFrame: Processed DataFrame with normalized personality scores.
    """
    # Clean the response text
    df = results.copy()
    df["cleaned_response"] = df["response"].apply(
        lambda text: clean_text(text, base_question)
    )

    # Detect personality traits for each response
    df["personality"] = df["cleaned_response"].apply(personality_detection)

    # Extract personality traits into separate columns
    traits = [
        "Extroversion",
        "Neuroticism",
        "Agreeableness",
        "Conscientiousness",
        "Openness",
    ]
    for trait in traits:
        df[trait] = df["personality"].apply(lambda x: x.get(trait, np.nan))

    # Find baseline (where both vector strengths are 0)
    baseline_mask = (df["vector1_strength"] == 0) & (df["vector2_strength"] == 0)
    if baseline_mask.any():
        baseline_values = df.loc[baseline_mask, traits].iloc[0].values
    else:
        # Default to zeros if no baseline
        baseline_values = np.zeros(len(traits))

    # Normalize by subtracting baseline values
    df[traits] = df[traits] - baseline_values

    # Save to CSV
    df.to_csv(output_csv, index=False)
    return df


def judge_responses_multiple_metrics(
    df, response_col="cleaned_response", metrics=None, score_range=(0, 10)
):
    """
    Use OpenAI LLM to judge responses in a DataFrame across multiple metrics and assign quantitative scores.

    Parameters:
    - df: DataFrame containing the responses
    - response_col: str, column name with responses to judge (default: "cleaned_response")
    - metrics: list of str, metrics to judge (default: ["clarity", "engagement", "relevance"])
    - score_range: tuple, (min, max) score range (default: (0, 10))

    Returns:
    - DataFrame with new columns containing LLM-assigned scores for each metric
    """
    # Default metrics if none provided
    if metrics is None:
        metrics = ["clarity", "engagement", "relevance"]

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Copy the DataFrame to avoid modifying the original
    results_df = df.copy()

    # Initialize new columns for each metric
    for metric in metrics:
        score_col = f"llm_{metric}_score"
        results_df[score_col] = pd.NA  # Use NA for missing values

    # Define the prompt template for multiple metrics
    prompt_template = f"""
    Evaluate the following response and assign it a score from {score_range[0]} to {score_range[1]} for each metric below.
    Return ONLY a comma-separated list of integers in the order of the metrics, nothing else.
    Metrics: {", ".join(metrics)}

    Response:
    "{{response}}"
    """

    # Process each response
    for idx, row in tqdm(
        results_df.iterrows(), total=len(results_df), desc="Judging responses"
    ):
        response = row[response_col]

        # Skip if response is empty or not a string
        if not isinstance(response, str) or len(response.strip()) == 0:
            for metric in metrics:
                results_df.at[idx, f"llm_{metric}_score"] = None
            continue

        try:
            prompt = prompt_template.format(response=response)

            # Call OpenAI API
            completion = client.chat.completions.create(
                model="o3-mini",  # Switch to "gpt-4-turbo" if desired
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator returning only numerical scores.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            # Extract and parse the scores
            scores = completion.choices[0].message.content.strip().split(",")
            scores = [int(score.strip()) for score in scores]

            # Validate and assign scores
            if len(scores) == len(metrics):
                for metric, score in zip(metrics, scores):
                    score_col = f"llm_{metric}_score"
                    if score_range[0] <= score <= score_range[1]:
                        results_df.at[idx, score_col] = score
                    else:
                        results_df.at[idx, score_col] = None  # Invalid score
            else:
                print(
                    f"Warning: Mismatched number of scores at index {idx}. Expected {len(metrics)}, got {len(scores)}"
                )
                for metric in metrics:
                    results_df.at[idx, f"llm_{metric}_score"] = None

        except Exception as e:
            print(f"Error judging response at index {idx}: {e}")
            for metric in metrics:
                results_df.at[idx, f"llm_{metric}_score"] = None

    return results_df
