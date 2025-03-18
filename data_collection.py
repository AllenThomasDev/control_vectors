import pandas as pd
import numpy as np
from doomer.model_utils import generate_with_vector
from tqdm import tqdm


def collect_responses_with_vectors(
    input_text: str,
    model,
    tokenizer,
    vectors: dict,
    strength_range=(-2, 2, 0.5),
    max_new_tokens: int = 256,
    save_to_csv: str = None,
):
    start, end, step = strength_range
    strengths = np.arange(start, end + step, step)
    results = []
    vector_names = list(vectors.keys())

    for strength1 in tqdm(strengths, desc="Outer loop (Strength 1)"):
        for strength2 in strengths:
            combined_vector = (
                vectors[vector_names[0]] * strength1
                + vectors[vector_names[1]] * strength2
            )
            response = generate_with_vector(
                model, tokenizer, input_text, combined_vector, max_new_tokens
            )

            result = {
                "vector1_name": vector_names[0],
                "vector2_name": vector_names[1],
                "vector1_strength": strength1,
                "vector2_strength": strength2,
                "response": response,
            }
            results.append(result)

    if save_to_csv:
        df = pd.DataFrame(results)
        df.to_csv(save_to_csv, index=False)
        print(f"Responses saved to {save_to_csv}")

    return results
