from constants import (
    STRENGTH_RANGES,
)
import torch
from tqdm import tqdm
from itertools import product
import pandas as pd
import numpy as np
import os
from repeng import ControlVector, DatasetEntry


def ai_template(persona: str, suffix: str) -> str:
    user_tag, asst_tag = "<|im_start|>user", "<|im_start|>assistant"
    return f"{user_tag} Pretend you're an {persona} expressing your opinions about AI progress. <|im_end|> {asst_tag} {suffix}"


def social_template(persona: str, suffix: str) -> str:
    user_tag, asst_tag = "<|im_start|>user", "<|im_start|>assistant"
    return f"{user_tag} Pretend you're an {persona} talking about how you behave in social settings. <|im_end|> {asst_tag} {suffix}"


def train_control_vector(
    model,
    tokenizer,
    suffixes,
    template_function,
    positive_persona,
    negative_persona,
    model_name,
):
    """Train and save a control vector with model-specific naming."""
    vector_filename = (
        f"vectors/{model_name}_{positive_persona}_{negative_persona}_control_vector.pt"
    )

    if os.path.exists(vector_filename):
        print(f"Control vector already exists at {vector_filename}. Skipping training.")
        return

    dataset = []
    for suffix in suffixes:
        tokens = tokenizer.tokenize(suffix)
        for i in range(1, len(tokens) - 1):
            truncated = tokenizer.convert_tokens_to_string(tokens[:i])
            dataset.append(
                DatasetEntry(
                    positive=template_function(positive_persona, truncated),
                    negative=template_function(negative_persona, truncated),
                )
            )

    print(f"Training control vector for {positive_persona} vs. {negative_persona}...")
    model.reset()
    control_vector = ControlVector.train(model, tokenizer, dataset)

    os.makedirs("vectors", exist_ok=True)
    torch.save(control_vector, vector_filename)
    print(f"Saved control vector to {vector_filename}")
    model.reset()


def load_control_vector(model_name, positive_persona, negative_persona):
    """Load a control vector if it exists."""
    vector_path = (
        f"vectors/{model_name}_{positive_persona}_{negative_persona}_control_vector.pt"
    )
    if os.path.exists(vector_path):
        vector = torch.load(vector_path, weights_only=False)
        print(f"Loaded control vector from {vector_path}")
        return vector
    print(f"No control vector found at {vector_path}")
    return None


def generate_with_vector(
    model,
    tokenizer,
    input_str: str,
    *vectors,
    max_new_tokens=256,
):
    """Generate text with optional control vectors applied."""
    from constants import GENERATION_SETTINGS

    settings = GENERATION_SETTINGS.copy()
    settings["pad_token_id"] = tokenizer.eos_token_id
    settings["eos_token_id"] = tokenizer.eos_token_id
    settings["max_new_tokens"] = max_new_tokens

    input_ids = tokenizer(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": input_str}],
            add_generation_prompt=True,
            tokenize=False,
        ),
        return_tensors="pt",
    ).to(model.device)

    def gen():
        output_ids = model.generate(**input_ids, **settings)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    for vector in vectors:
        model.set_control(vector)
        output = gen()
        model.reset()
        return output

    model.reset()
    return None


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

    strength_combinations = list(product(strengths, repeat=num_vectors))
    results = []
    for combo in tqdm(strength_combinations, desc="Generating responses"):
        # Scale each vector by its strength and sum them
        scaled_vectors = [
            vectors[name] * strength for name, strength in zip(vector_names, combo)
        ]
        # Handle empty case, though unlikely here
        combined_vector = sum(scaled_vectors) if scaled_vectors else None

        if combined_vector is not None:
            response = generate_with_vector(
                model, tokenizer, input_text, combined_vector, max_new_tokens
            )
        else:
            response = "No vectors provided"

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
