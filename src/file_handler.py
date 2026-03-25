"""
file_handler.py
---------------
Handles all file input and output operations.
Reads CSV files containing text data and writes analysis results back to CSV.
"""

import os
import csv
from datetime import datetime


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
SUPPORTED_EXTENSIONS = [".csv", ".txt"]
DEFAULT_OUTPUT_DIR   = "outputs"
DEFAULT_DATA_DIR     = "data"


# ─────────────────────────────────────────────
#  CSV Reading
# ─────────────────────────────────────────────

def read_csv(filepath: str, text_column: str = None) -> list:
    """
    Reads a CSV file and returns a list of text strings for analysis.

    Parameters:
        filepath    : full or relative path to the CSV file
        text_column : name of the column containing the text to analyze.
                      If None, the function auto-detects the best column.

    Returns:
        A list of strings, one per row.
        Skips empty rows and header rows automatically.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: '{filepath}'")

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Use .csv or .txt")

    if ext == ".txt":
        return read_txt(filepath)

    texts = []
    with open(filepath, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        headers = reader.fieldnames

        if not headers:
            raise ValueError("CSV file has no headers. Please add a header row.")

        # Auto-detect which column contains the text
        column = text_column or _detect_text_column(headers)

        if column not in headers:
            raise ValueError(
                f"Column '{column}' not found.\n"
                f"Available columns: {', '.join(headers)}"
            )

        for row in reader:
            value = row.get(column, "").strip()
            if value:
                texts.append(value)

    if not texts:
        raise ValueError("No text data found in the file. Check the column name.")

    return texts


def read_txt(filepath: str) -> list:
    """
    Reads a plain .txt file where each line is one text entry.
    Skips blank lines automatically.
    """
    with open(filepath, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def get_csv_headers(filepath: str) -> list:
    """
    Returns the list of column headers from a CSV file.
    Useful for letting the user pick which column to analyze.
    """
    with open(filepath, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        return reader.fieldnames or []


def get_file_info(filepath: str) -> dict:
    """
    Returns basic metadata about a file without reading all its content.
    Used to show the user a preview before running analysis.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: '{filepath}'")

    size_bytes = os.path.getsize(filepath)
    size_kb    = round(size_bytes / 1024, 2)

    # Count rows quickly
    row_count = 0
    with open(filepath, encoding="utf-8-sig") as f:
        for _ in f:
            row_count += 1

    headers = []
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        headers = get_csv_headers(filepath)
        row_count = max(0, row_count - 1)   # subtract header row

    return {
        "filepath"  : filepath,
        "filename"  : os.path.basename(filepath),
        "extension" : ext,
        "size_kb"   : size_kb,
        "row_count" : row_count,
        "headers"   : headers,
    }


# ─────────────────────────────────────────────
#  CSV Writing / Saving Results
# ─────────────────────────────────────────────

def save_results_to_csv(results: list, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """
    Saves a list of analysis result dicts to a timestamped CSV file.

    Each row in the output CSV contains:
        index, original_text, sentiment, polarity,
        subjectivity, intensity, subjectivity_label, word_count, interpretation

    Returns the full filepath of the saved file.
    """
    _ensure_directory(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"sentiment_results_{timestamp}.csv"
    filepath  = os.path.join(output_dir, filename)

    fieldnames = [
        "index",
        "original_text",
        "sentiment",
        "polarity",
        "subjectivity",
        "intensity",
        "subjectivity_label",
        "word_count",
        "interpretation",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, result in enumerate(results, start=1):
            writer.writerow({
                "index"             : result.get("index", i),
                "original_text"     : result.get("original_text", ""),
                "sentiment"         : result.get("sentiment", ""),
                "polarity"          : result.get("polarity", 0.0),
                "subjectivity"      : result.get("subjectivity", 0.0),
                "intensity"         : result.get("intensity", ""),
                "subjectivity_label": result.get("subjectivity_label", ""),
                "word_count"        : result.get("word_count", 0),
                "interpretation"    : result.get("interpretation", ""),
            })

    return filepath


def save_summary_to_txt(summary: dict, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """
    Saves the summary statistics dictionary to a human-readable .txt file.
    Returns the full filepath of the saved file.
    """
    _ensure_directory(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"sentiment_summary_{timestamp}.txt"
    filepath  = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 55 + "\n")
        f.write("        SENTIMENT ANALYSIS — SUMMARY REPORT\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Generated on : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("OVERVIEW\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total texts analyzed : {summary.get('total', 0)}\n")
        f.write(f"Overall sentiment    : {summary.get('overall_sentiment', 'N/A')}\n\n")

        f.write("SENTIMENT BREAKDOWN\n")
        f.write("-" * 30 + "\n")
        f.write(f"Positive : {summary.get('positive_count', 0)}  ({summary.get('positive_pct', 0)}%)\n")
        f.write(f"Negative : {summary.get('negative_count', 0)}  ({summary.get('negative_pct', 0)}%)\n")
        f.write(f"Neutral  : {summary.get('neutral_count',  0)}  ({summary.get('neutral_pct',  0)}%)\n\n")

        f.write("SCORE AVERAGES\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average polarity     : {summary.get('avg_polarity', 0.0):+.4f}\n")
        f.write(f"Average subjectivity : {summary.get('avg_subjectivity', 0.0):.4f}\n\n")

        most_pos = summary.get("most_positive", {})
        most_neg = summary.get("most_negative", {})

        f.write("HIGHLIGHTS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Most positive text (polarity {most_pos.get('polarity', 0):+.4f}):\n")
        f.write(f"  \"{most_pos.get('original_text', '')[:80]}\"\n\n")
        f.write(f"Most negative text (polarity {most_neg.get('polarity', 0):+.4f}):\n")
        f.write(f"  \"{most_neg.get('original_text', '')[:80]}\"\n\n")

        f.write("=" * 55 + "\n")
        f.write("End of report\n")

    return filepath


# ─────────────────────────────────────────────
#  Sample Data Generator
# ─────────────────────────────────────────────

def create_sample_csv(filepath: str = None) -> str:
    """
    Creates a sample CSV file with 25 diverse review texts.
    Used to populate data/sample_reviews.csv for demo and testing purposes.
    Returns the filepath of the created file.
    """
    if filepath is None:
        filepath = os.path.join(DEFAULT_DATA_DIR, "sample_reviews.csv")

    _ensure_directory(os.path.dirname(filepath))

    sample_data = [
        # Positive reviews
        ("This product is absolutely fantastic! I love everything about it.", "Electronics"),
        ("Best purchase I have ever made. Highly recommend to everyone.", "Electronics"),
        ("The quality is outstanding and delivery was super fast.", "Clothing"),
        ("Amazing customer service. They went above and beyond to help me.", "Service"),
        ("I am so happy with this product. It exceeded all my expectations.", "Home"),
        ("Wonderful experience from start to finish. Will definitely buy again.", "Service"),
        ("The food was delicious and the atmosphere was very welcoming.", "Restaurant"),
        ("Excellent build quality. Worth every penny spent on it.", "Electronics"),

        # Negative reviews
        ("Terrible product. Broke after just two days of use.", "Electronics"),
        ("Worst customer service I have ever experienced. Very disappointing.", "Service"),
        ("The item arrived damaged and smelled awful. Requesting a refund.", "Clothing"),
        ("Complete waste of money. Does not work as advertised at all.", "Electronics"),
        ("Very poor quality. I expected much better for this price.", "Home"),
        ("Awful experience. The staff was rude and unhelpful throughout.", "Restaurant"),
        ("Do not buy this. It stopped working after one week.", "Electronics"),
        ("Extremely disappointed. The product looks nothing like the pictures.", "Clothing"),

        # Neutral reviews
        ("The product arrived on time. It is okay for the price.", "Home"),
        ("Decent quality. Nothing special but it does the job.", "Electronics"),
        ("The package was delivered. Product matches the description.", "Clothing"),
        ("It is a standard product. Works as expected nothing more.", "Home"),
        ("Average experience. The food was neither good nor bad.", "Restaurant"),

        # Mixed or complex
        ("The product is good but the shipping took way too long.", "Electronics"),
        ("Great price but the quality could definitely be improved.", "Clothing"),
        ("Customer service was excellent but the product itself disappointed me.", "Service"),
        ("Good overall but I wish the instructions were clearer.", "Home"),
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["review", "category"])
        writer.writerows(sample_data)

    return filepath


# ─────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────

def _ensure_directory(directory: str) -> None:
    """Creates a directory if it does not already exist."""
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def _detect_text_column(headers: list) -> str:
    """
    Tries to automatically find the most likely text column in a CSV.
    Checks for common column names used in review datasets.
    Falls back to the first column if nothing matches.
    """
    priority_names = [
        "review", "text", "comment", "feedback",
        "description", "message", "content", "tweet",
        "post", "sentence", "input", "data",
    ]
    headers_lower = [h.lower() for h in headers]

    for name in priority_names:
        if name in headers_lower:
            index = headers_lower.index(name)
            return headers[index]

    # Nothing matched — just use the first column
    return headers[0]