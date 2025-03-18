# control_vectors.py
import torch
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
