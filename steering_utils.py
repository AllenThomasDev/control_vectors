import torch


def convert_control_vector_to_steer_config(directions, output_path="steer_config.pt"):
    """
    Converts a ControlVector object into a steer configuration and saves it as a PyTorch file.

    Args:
    directions (dict): The directions attribute from a ControlVector object (layer_num -> direction_array).
    output_path (str): Path to save the steering config file (default: "steer_config.pt").

    Returns:
    dict: The steering configuration dictionary.
    """
    steer_config = {}
    for layer_num, direction_array in directions.items():
        layer_key = f"layers.{layer_num}"
    # Convert numpy array to PyTorch tensor and reshape to (1, dim)
    steering_vector = torch.tensor(direction_array).unsqueeze(0)

    steer_config[layer_key] = {
        "steering_vector": steering_vector,
        "bias": torch.zeros_like(steering_vector),  # No bias by default
        "steering_coefficient": 1.0,  # Default coefficient
        "action": "add",  # Default action
    }

    torch.save(steer_config, output_path)
    print(f"Saved steering config to {output_path}")
    return steer_config
