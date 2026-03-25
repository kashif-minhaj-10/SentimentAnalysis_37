"""
analyzer.py
-----------
Core sentiment analysis engine using TextBlob.
Handles single text analysis, batch analysis, and result interpretation.
"""

from textblob import TextBlob
from preprocessor import clean_text, is_valid_text, get_word_count


# ─────────────────────────────────────────────
#  Classification thresholds
#  You can adjust these values if needed
# ─────────────────────────────────────────────
POSITIVE_THRESHOLD  =  0.1
NEGATIVE_THRESHOLD  = -0.1
HIGH_SUBJECTIVITY   =  0.6
LOW_SUBJECTIVITY    =  0.3


def analyze_sentiment(text: str) -> dict:
    """
    Main analysis function.
    Accepts raw text, cleans it, runs TextBlob, and returns a full result dict.

    Returns a dictionary with:
        - original_text    : the text as provided by the user
        - cleaned_text     : text after preprocessing
        - polarity         : float from -1.0 (negative) to +1.0 (positive)
        - subjectivity     : float from 0.0 (objective) to 1.0 (subjective)
        - sentiment        : 'Positive', 'Negative', or 'Neutral'
        - intensity        : 'Strong', 'Moderate', or 'Mild'
        - subjectivity_label : 'Subjective', 'Objective', or 'Balanced'
        - interpretation   : human-readable explanation string
        - word_count       : number of words in the original text
        - is_valid         : whether the text had enough content to analyze
    """

    # Validate input
    if not is_valid_text(text):
        return _invalid_result(text)

    cleaned = clean_text(text)

    # Run TextBlob analysis
    blob = TextBlob(cleaned)
    polarity    = round(blob.sentiment.polarity,    4)
    subjectivity = round(blob.sentiment.subjectivity, 4)

    # Classify the results
    sentiment         = _classify_sentiment(polarity)
    intensity         = _classify_intensity(polarity)
    subjectivity_label = _classify_subjectivity(subjectivity)
    interpretation    = _generate_interpretation(
                            polarity, subjectivity,
                            sentiment, intensity, subjectivity_label
                        )

    return {
        "original_text"     : text,
        "cleaned_text"      : cleaned,
        "polarity"          : polarity,
        "subjectivity"      : subjectivity,
        "sentiment"         : sentiment,
        "intensity"         : intensity,
        "subjectivity_label": subjectivity_label,
        "interpretation"    : interpretation,
        "word_count"        : get_word_count(text),
        "is_valid"          : True,
    }


def analyze_batch(texts: list) -> list:
    """
    Analyzes a list of text strings one by one.
    Returns a list of result dictionaries (same format as analyze_sentiment).
    Skips empty or None entries gracefully.
    """
    results = []
    for index, text in enumerate(texts, start=1):
        if text is None or str(text).strip() == "":
            continue
        result = analyze_sentiment(str(text))
        result["index"] = index
        results.append(result)
    return results


def get_summary_statistics(results: list) -> dict:
    """
    Takes a list of result dicts from analyze_batch and computes overall stats.

    Returns:
        - total          : total number of texts analyzed
        - positive_count : number of positive results
        - negative_count : number of negative results
        - neutral_count  : number of neutral results
        - positive_pct   : percentage positive
        - negative_pct   : percentage negative
        - neutral_pct    : percentage neutral
        - avg_polarity   : average polarity score
        - avg_subjectivity : average subjectivity score
        - most_positive  : the result dict with the highest polarity
        - most_negative  : the result dict with the lowest polarity
        - overall_sentiment : dominant sentiment across all texts
    """
    if not results:
        return {}

    valid_results = [r for r in results if r.get("is_valid")]
    total = len(valid_results)

    if total == 0:
        return {"total": 0}

    positive = [r for r in valid_results if r["sentiment"] == "Positive"]
    negative = [r for r in valid_results if r["sentiment"] == "Negative"]
    neutral  = [r for r in valid_results if r["sentiment"] == "Neutral"]

    avg_polarity     = round(sum(r["polarity"]     for r in valid_results) / total, 4)
    avg_subjectivity = round(sum(r["subjectivity"] for r in valid_results) / total, 4)

    most_positive = max(valid_results, key=lambda r: r["polarity"])
    most_negative = min(valid_results, key=lambda r: r["polarity"])

    counts = {
        "Positive": len(positive),
        "Negative": len(negative),
        "Neutral" : len(neutral),
    }
    overall_sentiment = max(counts, key=counts.get)

    return {
        "total"            : total,
        "positive_count"   : len(positive),
        "negative_count"   : len(negative),
        "neutral_count"    : len(neutral),
        "positive_pct"     : round(len(positive) / total * 100, 1),
        "negative_pct"     : round(len(negative) / total * 100, 1),
        "neutral_pct"      : round(len(neutral)  / total * 100, 1),
        "avg_polarity"     : avg_polarity,
        "avg_subjectivity" : avg_subjectivity,
        "most_positive"    : most_positive,
        "most_negative"    : most_negative,
        "overall_sentiment": overall_sentiment,
    }


# ─────────────────────────────────────────────
#  Internal helper functions
# ─────────────────────────────────────────────

def _classify_sentiment(polarity: float) -> str:
    """Returns 'Positive', 'Negative', or 'Neutral' based on polarity score."""
    if polarity > POSITIVE_THRESHOLD:
        return "Positive"
    elif polarity < NEGATIVE_THRESHOLD:
        return "Negative"
    else:
        return "Neutral"


def _classify_intensity(polarity: float) -> str:
    """
    Returns how strongly positive or negative the sentiment is.
    Works on the absolute value so it applies to both directions.
    """
    abs_polarity = abs(polarity)
    if abs_polarity >= 0.5:
        return "Strong"
    elif abs_polarity >= 0.2:
        return "Moderate"
    else:
        return "Mild"


def _classify_subjectivity(subjectivity: float) -> str:
    """
    Returns whether the text is mostly opinion-based or factual.
        > 0.6  -> Subjective  (personal opinion, emotions)
        < 0.3  -> Objective   (facts, data, neutral statements)
        else   -> Balanced
    """
    if subjectivity > HIGH_SUBJECTIVITY:
        return "Subjective"
    elif subjectivity < LOW_SUBJECTIVITY:
        return "Objective"
    else:
        return "Balanced"


def _generate_interpretation(
    polarity: float,
    subjectivity: float,
    sentiment: str,
    intensity: str,
    subjectivity_label: str,
) -> str:
    """
    Builds a plain-English explanation of what the scores mean.
    This is what gets displayed to the user as the final readable result.
    """
    if sentiment == "Positive":
        pol_desc = f"The text expresses a {intensity.lower()} positive sentiment (polarity: {polarity:+.4f})."
    elif sentiment == "Negative":
        pol_desc = f"The text expresses a {intensity.lower()} negative sentiment (polarity: {polarity:+.4f})."
    else:
        pol_desc = f"The text expresses a neutral sentiment (polarity: {polarity:+.4f}), showing neither clear positivity nor negativity."

    if subjectivity_label == "Subjective":
        sub_desc = f"It is highly subjective (subjectivity: {subjectivity:.4f}), meaning it is opinion-based rather than factual."
    elif subjectivity_label == "Objective":
        sub_desc = f"It is largely objective (subjectivity: {subjectivity:.4f}), suggesting factual or neutral language."
    else:
        sub_desc = f"It has a balanced mix of opinion and fact (subjectivity: {subjectivity:.4f})."

    return f"{pol_desc} {sub_desc}"


def _invalid_result(text: str) -> dict:
    """Returns a standard result dict for text that cannot be analyzed."""
    return {
        "original_text"     : text,
        "cleaned_text"      : "",
        "polarity"          : 0.0,
        "subjectivity"      : 0.0,
        "sentiment"         : "Neutral",
        "intensity"         : "Mild",
        "subjectivity_label": "Balanced",
        "interpretation"    : "Text was too short or empty to analyze.",
        "word_count"        : 0,
        "is_valid"          : False,
    }