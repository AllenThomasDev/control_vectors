from transformers import AutoTokenizer, AutoModelForCausalLM
from repeng import ControlModel
import torch
from constants import MODEL_CONFIGS, HF_TOKEN


def load_model(model_choice):
    """Load the specified model and tokenizer."""
    config = MODEL_CONFIGS.get(model_choice.lower())
    if not config:
        raise ValueError(f"Invalid model choice. Use 'hermes' or 'deepseek'.")

    model_name = config["name"]
    control_layers = config["control_layers"]

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    tokenizer.pad_token_id = 0
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model = ControlModel(model, control_layers)
    return model, tokenizer
