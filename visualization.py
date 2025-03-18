import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from doomer.evaluation import (
    sentiment_compound,
    sentiment_proportions,
)  # Import directly from evaluation.py


def create_heatmap(df, traits=None):
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
    proportions = df[response_col].apply(sentiment_proportions)
    df["positive"] = proportions.apply(lambda x: x["Positive"])
    df["neutral"] = proportions.apply(lambda x: x["Neutral"])
    df["negative"] = proportions.apply(lambda x: x["Negative"])

    metrics = ["positive", "neutral", "negative"]
    metric_names = ["Positive", "Neutral", "Negative"]

    subset = df[df[fixed_vector] == fixed_value].sort_values(target_vector)
    if subset.empty:
        print(f"No data found where {fixed_vector} = {fixed_value}")
        return None

    fig = go.Figure()

    colors = [
        "rgba(0, 200, 0, 0.7)",  # Vibrant green for Positive
        "rgba(150, 150, 150, 0.5)",  # Subtle gray for Neutral
        "rgba(255, 0, 0, 0.7)",  # Vibrant red for Negative
    ]

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
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1,
        ),
        yaxis=dict(gridcolor="lightgray", tickvals=[0, 0.25, 0.5, 0.75, 1]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=50, t=100, b=50),
    )

    return fig


def create_sentiment_heatmaps(df, response_col="cleaned_response"):
    df["compound"] = df[response_col].apply(sentiment_compound)
    proportions = df[response_col].apply(sentiment_proportions)
    df["positive"] = proportions.apply(lambda x: x["Positive"])
    df["neutral"] = proportions.apply(lambda x: x["Neutral"])
    df["negative"] = proportions.apply(lambda x: x["Negative"])

    metrics = ["compound", "positive", "neutral", "negative"]
    titles = [
        "Compound Sentiment",
        "Positive Proportion",
        "Neutral Proportion",
        "Negative Proportion",
    ]

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
        for metric in metrics:
            df[f"{metric}_norm"] = df[metric] - baseline_values[metric]

    fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=[f"{title} (Normalized)" for title in titles],
        horizontal_spacing=0.05,
    )

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

        if i == 0:
            fig.update_yaxes(title_text="vector1_strength (ai)", row=1, col=i + 1)
        fig.update_xaxes(title_text="vector2_strength (introversion)", row=1, col=i + 1)

        fig.update_yaxes(scaleanchor=f"x{i + 1}", scaleratio=1, row=1, col=i + 1)

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
    colorscale = "Viridis"

    for i, (vector, vector_name) in enumerate(vectors, start=1):  # Rows
        strengths = df[vector].to_numpy()
        trend_x = np.linspace(min(strengths), max(strengths), 100)

        for j, (metric_func, metric_name) in enumerate(
            metric_funcs, start=1
        ):  # Columns
            metric_values = df[response_col].apply(metric_func).to_numpy()

            trend_poly = np.polyfit(strengths, metric_values, deg=trend_degree)
            trend_line = np.poly1d(trend_poly)
            trend_y = trend_line(trend_x)

            fig.add_trace(
                go.Scatter(
                    x=strengths,
                    y=metric_values,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=metric_values,
                        colorscale=colorscale,
                        showscale=False,
                    ),
                    name=metric_name,
                    hovertemplate=f"{vector_name} Strength: %{{x}}<br>{metric_name}: %{{y:.3f}}",
                    showlegend=(i == 1 and j == 1),
                ),
                row=i,
                col=j,
            )

            fig.add_trace(
                go.Scatter(
                    x=trend_x,
                    y=trend_y,
                    mode="lines",
                    line=dict(color="black", width=2, dash="dash"),
                    name=f"Trend (degree {trend_degree})",
                    hoverinfo="skip",
                    showlegend=(i == 1 and j == 1),
                ),
                row=i,
                col=j,
            )
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


def plot_control_vs_llm_scores(
    df,
    control_vectors=["vector1_strength", "vector2_strength"],
    llm_scores=["llm_clarity_score", "llm_engagement_score", "llm_relevance_score"],
    width=1500,
    height=1000,
):
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
    fig.update_layout(
        width=width,
        height=height,
        title_text="Control Vector Strengths vs. LLM Scores",
        showlegend=False,  # Legend not needed since each subplot is labeled
        template="plotly_white",  # Clean white background
    )
    fig.show()
