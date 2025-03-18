# chat_interface.py
import gradio as gr
import torch
from models import load_model
from control_vectors import generate_with_vector, load_control_vector
from constants import MODEL_CONFIGS, CONTROL_VECTOR_CONFIGS

# Load model and tokenizer (default to hermes)
model, tokenizer = load_model("hermes")
model_key = "hermes"

# Load control vectors
control_vectors = {
    "ai_optimism": load_control_vector(
        model_key,
        CONTROL_VECTOR_CONFIGS["ai"]["positive_persona"],
        CONTROL_VECTOR_CONFIGS["ai"]["negative_persona"],
    ),
    "introversion": load_control_vector(
        model_key,
        CONTROL_VECTOR_CONFIGS["introvert"]["positive_persona"],
        CONTROL_VECTOR_CONFIGS["introvert"]["negative_persona"],
    ),
}


def chat_interface(
    input_text, ai_optimism_strength, introversion_strength, show_baseline, max_tokens
):
    if not input_text:
        return "Please enter some text."

    # Prepare vectors based on strengths
    vectors = []
    if ai_optimism_strength != 0 and control_vectors["ai_optimism"]:
        scaled_ai_vector = control_vectors["ai_optimism"] * ai_optimism_strength
    vectors.append(scaled_ai_vector)
    if introversion_strength != 0 and control_vectors["introversion"]:
        scaled_introvert_vector = (
            control_vectors["introversion"] * introversion_strength
        )
    vectors.append(scaled_introvert_vector)

    # Generate the response
    response = generate_with_vector(
        input_text,
        *vectors,
        max_new_tokens=max_tokens,
        show_baseline=show_baseline,
        render=False,
    )

    return response


# Create Gradio interface
with gr.Blocks(title="Vector-Controlled Chatbot") as demo:
    gr.Markdown("# Vector-Controlled Chatbot")
    gr.Markdown(
        "Enter a prompt, adjust the strengths of the control vectors, and generate a response."
    )

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Your Message", placeholder="Type your prompt here..."
            )
    ai_optimism_slider = gr.Slider(
        minimum=-5, maximum=5, step=0.1, value=0, label="AI Optimism Strength"
    )
    introversion_slider = gr.Slider(
        minimum=-5, maximum=5, step=0.1, value=0, label="Introversion Strength"
    )
    max_tokens = gr.Slider(
        minimum=50, maximum=500, step=10, value=256, label="Max New Tokens"
    )
    show_baseline = gr.Checkbox(label="Show Baseline Response", value=False)
    submit_btn = gr.Button("Generate")

    with gr.Column():
        output_text = gr.Textbox(label="Response", lines=10, interactive=False)

    submit_btn.click(
        fn=chat_interface,
        inputs=[
            input_text,
            ai_optimism_slider,
            introversion_slider,
            show_baseline,
            max_tokens,
        ],
        outputs=output_text,
    )

if __name__ == "__main__":
    demo.launch(share=True)
