import nltk
import os


def download_nltk_data():
    nltk_data_path = nltk.data.path[0]  # First path in nltk.data.path
    words_path = os.path.join(nltk_data_path, "corpora", "words")
    if os.path.exists(words_path):
        print("NLTK 'words' corpus already exists at:", words_path)
    else:
        print("Downloading NLTK 'words' corpus to:", nltk_data_path)
        nltk.download("words", quiet=True)
        print("NLTK 'words' corpus downloaded successfully.")


if __name__ == "__main__":
    download_nltk_data()
