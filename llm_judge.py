import pandas as pd
import openai
from tqdm import tqdm
from doomer.config import (
    OPENAI_API_KEY,
    LLM_JUDGE_MODEL,
    LLM_JUDGE_SCORE_RANGE,
    LLM_JUDGE_METRICS,
)


def judge_responses_multiple_metrics(
    df, response_col="cleaned_response", metrics=None, score_range=None
):
    """
    Use OpenAI LLM to judge responses in a DataFrame across multiple metrics and assign quantitative scores.

    Parameters:
    - df: DataFrame containing the responses
    - response_col: str, column name with responses to judge (default: "cleaned_response")
    - metrics: list of str, metrics to judge (default: LLM_JUDGE_METRICS from config)
    - score_range: tuple, (min, max) score range (default: LLM_JUDGE_SCORE_RANGE from config)

    Returns:
    - DataFrame with new columns containing LLM-assigned scores for each metric
    """
    if metrics is None:
        metrics = LLM_JUDGE_METRICS
    if score_range is None:
        score_range = LLM_JUDGE_SCORE_RANGE

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    results_df = df.copy()

    for metric in metrics:
        score_col = f"llm_{metric}_score"
        results_df[score_col] = pd.NA  # Use NA for missing values

    prompt_template = f"""
    Evaluate the following response and assign it a score from {score_range[0]} to {score_range[1]} for each metric below.
    Return ONLY a comma-separated list of integers in the order of the metrics, nothing else.
    Metrics: {", ".join(metrics)}

    Response:
    "{{response}}"
    """

    for idx, row in tqdm(
        results_df.iterrows(), total=len(results_df), desc="Judging responses"
    ):
        response = row[response_col]

        if not isinstance(response, str) or len(response.strip()) == 0:
            for metric in metrics:
                results_df.at[idx, f"llm_{metric}_score"] = None
            continue

        try:
            prompt = prompt_template.format(response=response)

            completion = client.chat.completions.create(
                model=LLM_JUDGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator returning only numerical scores.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            scores_str = completion.choices[0].message.content.strip()
            scores = [
                int(score.strip())
                for score in scores_str.split(",")
                if score.strip().isdigit()
            ]

            if len(scores) == len(metrics):
                for metric, score in zip(metrics, scores):
                    score_col = f"llm_{metric}_score"
                    if score_range[0] <= score <= score_range[1]:
                        results_df.at[idx, score_col] = score
                    else:
                        results_df.at[idx, score_col] = None  # Invalid score
            else:
                print(
                    f"Warning: Mismatched number of scores at index {idx}. Expected {len(metrics)}, got {len(scores)}.  Response: {scores_str}"
                )
                for metric in metrics:
                    results_df.at[idx, f"llm_{metric}_score"] = None

        except Exception as e:
            print(f"Error judging response at index {idx}: {e}")
            for metric in metrics:
                results_df.at[idx, f"llm_{metric}_score"] = None

    return results_df
