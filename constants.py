import os

MODEL_CONFIGS = {
    "hermes": {
        "name": "NousResearch/Hermes-3-Llama-3.2-3B",
        "control_layers": list(range(-5, -18, -1)),
    },
    "deepseek": {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "control_layers": list(range(20, 60)),
    },
}

# AI-related suffixes
AI_SUFFIXES = [
    "AI is rapidly changing...",
    "The development of AI raises questions about...",
    "Considering the implications of AI technology...",
    "AI has the potential to impact...",
    "Many people are discussing the future of AI and...",
    "AI algorithms are increasingly used to...",
    "The ability of AI to learn and adapt creates a...",
    "The societal impact of AI is something to...",
    "AI systems are becoming more capable of...",
    "It's important to understand how AI works to...",
    "The role of humans in an AI-driven world is...",
    "Thinking about the ethics of AI can help us...",
    "The capabilities of AI are advancing at a...",
    "AI technology is being applied in a variety of...",
    "There is much discussion about the power and limitations of...",
    "It's important to remember that AI is a...",
    "The future of work may involve a closer collaboration with...",
    "AI systems are designed to perform...",
    "The debate continues about the future benefits and potential harm of...",
    "We should think carefully about ways that people might...",
    "The technology behind AI allows computers to...",
    "AI has created opportunities and also...",
    "Many are interested in how AI will be used to...",
    "AI is something that many people have very strong opinions about.",
]

# Introversion-related suffixes
INTROVERSION_SUFFIXES = [
    "When I'm in a group setting, I tend to...",
    "In social situations, I usually prefer to...",
    "My ideal way to spend free time is...",
    "When meeting new people, I typically...",
    "I feel most energized when...",
    "During gatherings, I often find myself...",
    "The best way for me to unwind is...",
    "I’m most comfortable when...",
    "I like to recharge by...",
    "When there’s a lot of noise around me, I...",
    "I feel most confident when...",
    "In large crowds, I usually...",
    "When I’m faced with a new situation, I tend to...",
    "I usually enjoy being around people who...",
    "When I’m in a social setting, I prefer to...",
    "I’m happiest when I’m alone with my thoughts, but I also...",
    "If I have to make a decision in a group, I often...",
    "In conversations, I tend to...",
    "When I’m in a new place, I usually...",
    "I enjoy sharing my ideas with others, especially when...",
    "I feel most comfortable when others are around to...",
    "I often find myself retreating to a quiet space when...",
    "I think I express myself best when...",
    "If there’s a lot of people talking, I usually...",
    "When I go out, I prefer to go to places where...",
]

# API Keys (hardcoded for now, consider environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GENERATION_SETTINGS = {
    "pad_token_id": None,  # Will be set dynamically with tokenizer.eos_token_id
    "do_sample": False,  # Equivalent to temperature=0
    "max_new_tokens": 256,
    "repetition_penalty": 1.3,
    "eos_token_id": None,  # Will be set dynamically with tokenizer.eos_token_id
}

# Base question for data collection
BASE_QUESTION = "What are your thoughts on the future of AI and how it may impact social interactions?"
