import pandas as pd
import numpy as np
from doomer.model_utils import generate_with_vector
from tqdm import tqdm


from itertools import product


def collect_responses_with_vectors(
    input_text: str,
    model,
    tokenizer,
    vectors: dict,
    strength_range=(-2, 2, 0.5),
    max_new_tokens: int = 256,
    save_to_csv: str = None,
):
    """
    Collect responses by applying control vectors with varying strengths.

    Args:
        input_text (str): The input prompt for generation.
        model: The model object for text generation.
        tokenizer: The tokenizer for the model.
        vectors (dict): Dictionary of vector names to control vector objects (e.g., {"AI_Optimist_vs_AI_Doomer": vector1, ...}).
        strength_range (tuple): Tuple of (start, end, step) for strength values (default: (-2, 2, 0.5)).
        max_new_tokens (int): Maximum tokens to generate (default: 256).
        save_to_csv (str): Optional filename to save results as CSV.

    Returns:
        list: List of result dictionaries with vector names, strengths, and responses.
    """
    start, end, step = strength_range
    strengths = np.arange(start, end + step, step)
    vector_names = list(vectors.keys())
    num_vectors = len(vector_names)

    # Generate all combinations of strengths for the given number of vectors
    strength_combinations = list(product(strengths, repeat=num_vectors))

    results = []
    for combo in tqdm(strength_combinations, desc="Generating responses"):
        # Combine vectors with their respective strengths
        combined_vector = sum(
            vectors[name] * strength for name, strength in zip(vector_names, combo)
        )
        response = generate_with_vector(
            model, tokenizer, input_text, combined_vector, max_new_tokens
        )

        # Build result dictionary dynamically
        result = {f"vector{i + 1}_name": name for i, name in enumerate(vector_names)}
        result.update(
            {
                f"vector{i + 1}_strength": float(strength)
                for i, strength in enumerate(combo)
            }
        )
        result["response"] = response
        results.append(result)

    if save_to_csv:
        df = pd.DataFrame(results)
        df.to_csv(save_to_csv, index=False)
        print(f"Responses saved to {save_to_csv}")

    return results
