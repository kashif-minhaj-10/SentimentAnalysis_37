"""
preprocessor.py
---------------
Handles all text cleaning and preprocessing before sentiment analysis.
"""

import re
import string


def clean_text(text: str) -> str:
    """
    Master cleaning function — runs all steps in the correct order.
    Returns a cleaned version of the input text.
    """
    if not isinstance(text, str):
        text = str(text)

    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_mentions_and_hashtags(text)
    text = remove_special_characters(text)
    text = normalize_whitespace(text)

    return text.strip()


def remove_urls(text: str) -> str:
    """Removes http/https links and www addresses."""
    pattern = r"http[s]?://\S+|www\.\S+"
    return re.sub(pattern, "", text)


def remove_html_tags(text: str) -> str:
    """Strips HTML tags like <br>, <p>, <b> etc."""
    pattern = r"<[^>]+>"
    return re.sub(pattern, "", text)


def remove_mentions_and_hashtags(text: str) -> str:
    """Removes @mentions and #hashtags common in social media text."""
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    return text


def remove_special_characters(text: str) -> str:
    """
    Removes punctuation and special characters.
    Keeps letters, digits, and spaces only.
    Note: we do NOT lowercase here — TextBlob handles casing internally.
    """
    allowed = string.ascii_letters + string.digits + " "
    return "".join(char for char in text if char in allowed)


def normalize_whitespace(text: str) -> str:
    """Collapses multiple spaces/newlines/tabs into a single space."""
    return re.sub(r"\s+", " ", text)


def is_valid_text(text: str) -> bool:
    """
    Checks if the text has enough content to be worth analyzing.
    Returns False for empty strings or very short inputs.
    """
    if not isinstance(text, str):
        return False
    cleaned = text.strip()
    return len(cleaned) >= 3


def get_word_count(text: str) -> int:
    """Returns the number of words in a text string."""
    return len(text.split())


def truncate_text(text: str, max_length: int = 80) -> str:
    """
    Shortens text for display purposes only (not for analysis).
    Adds '...' if the text was cut.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."