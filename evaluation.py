import re
from collections import Counter
import nltk
from nltk.corpus import words
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure that necessary NLTK data is downloaded
try:
    english_words = set(words.words())
    analyzer = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("words")
    nltk.download("vader_lexicon")
    english_words = set(words.words())
    analyzer = SentimentIntensityAnalyzer()


def sentiment_compound(text):
    return analyzer.polarity_scores(text)["compound"]


def sentiment_proportions(text):
    scores = analyzer.polarity_scores(text)
    return {
        "Positive": scores["pos"],
        "Neutral": scores["neu"],
        "Negative": scores["neg"],
    }


def response_length(response):
    return len(str(response))


def short_word_ratio(response):
    tokens = re.findall(r"\b\w+\b", response)  # Improved regex
    if not tokens:
        return 0
    short_words = [word for word in tokens if len(word) < 3]
    return len(short_words) / len(tokens) if tokens else 0


def unigram_repetition(response):
    tokens = response.split()
    if not tokens:
        return 0
    word_counts = Counter(tokens)
    most_common_words = word_counts.most_common(3)
    return sum(count for _, count in most_common_words) / len(tokens)


def non_dict_ratio(response):
    tokens = response.split()
    if not tokens:
        return 0
    non_dict_words = [w for w in tokens if w.lower() not in english_words]
    return len(non_dict_words) / len(tokens)


def clean_text(text, input_question=""):
    """Remove tags and input question from response text."""
    tags = [
        "<|im_start|>user",
        "<|im_start|>assistant",
        "<|im_end|>",
        "<|im_start|>",
        "<|begin_of_text|>",
        "user\n",
        "assistant\n",
        "\n",
        "<｜User｜>",
        "<｜Assistant｜>",
        "<think>",
        "</think>",
    ]
    for tag in tags:
        text = text.replace(tag, "")
    text = text.replace(input_question, "")
    return text.strip()
