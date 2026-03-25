"""
main.py
-------
Entry point for the Sentiment Analysis System.
Provides a menu-driven CLI interface to run all features of the project.

Run this file from the project root directory:
    python main.py
"""

import os
import sys

# ─────────────────────────────────────────────
#  Add src/ to path so imports work correctly
# ─────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from analyzer      import analyze_sentiment, analyze_batch, get_summary_statistics
from file_handler  import (read_csv, save_results_to_csv,
                           save_summary_to_txt, create_sample_csv,
                           get_file_info)
from visualizer    import generate_all_charts
from preprocessor  import truncate_text


# ─────────────────────────────────────────────
#  Display helpers
# ─────────────────────────────────────────────

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_banner():
    print()
    print("  " + "=" * 55)
    print("       SENTIMENT ANALYSIS SYSTEM")
    print("       Powered by TextBlob & NLTK")
    print("  " + "=" * 55)
    print()


def print_divider(char="-", width=55):
    print("  " + char * width)


def print_menu():
    print("  MAIN MENU")
    print_divider()
    print("  [1]  Analyze a single text")
    print("  [2]  Analyze multiple texts manually")
    print("  [3]  Analyze from CSV file")
    print("  [4]  Generate sample CSV and analyze it")
    print("  [5]  View summary statistics")
    print("  [6]  Generate charts and visualizations")
    print("  [7]  Save results to file")
    print("  [8]  How to interpret results")
    print("  [0]  Exit")
    print_divider()


def print_result(result: dict, show_index: int = None):
    """Prints a single analysis result in a clean formatted block."""
    print()
    print_divider()

    if show_index is not None:
        print(f"  Result #{show_index}")
        print_divider()

    print(f"  Text     : {truncate_text(result['original_text'], 60)}")
    print(f"  Words    : {result['word_count']}")
    print_divider("·")

    # Sentiment label with simple inline indicator
    sentiment = result["sentiment"]
    indicator = {"Positive": "(+)", "Negative": "(-)", "Neutral": "( )"}
    print(f"  Sentiment    : {indicator.get(sentiment, '')}  {sentiment}")
    print(f"  Intensity    : {result['intensity']}")
    print(f"  Subjectivity : {result['subjectivity_label']}")
    print_divider("·")
    print(f"  Polarity     : {result['polarity']:+.4f}   (range -1.0 to +1.0)")
    print(f"  Subjectivity : {result['subjectivity']:.4f}    (range  0.0 to  1.0)")
    print_divider("·")
    print(f"  Interpretation:")
    print(f"  {result['interpretation']}")
    print_divider()


def print_summary(summary: dict):
    """Prints the batch summary statistics in a formatted block."""
    if not summary or summary.get("total", 0) == 0:
        print("\n  No summary available. Run an analysis first.")
        return

    print()
    print("  " + "=" * 45)
    print("  SUMMARY STATISTICS")
    print("  " + "=" * 45)
    print(f"  Total texts analyzed : {summary['total']}")
    print(f"  Overall sentiment    : {summary['overall_sentiment']}")
    print_divider()
    print(f"  Positive : {summary['positive_count']:>3}  ({summary['positive_pct']}%)")
    print(f"  Negative : {summary['negative_count']:>3}  ({summary['negative_pct']}%)")
    print(f"  Neutral  : {summary['neutral_count']:>3}  ({summary['neutral_pct']}%)")
    print_divider()
    print(f"  Avg polarity     : {summary['avg_polarity']:+.4f}")
    print(f"  Avg subjectivity : {summary['avg_subjectivity']:.4f}")
    print_divider()

    most_pos = summary.get("most_positive", {})
    most_neg = summary.get("most_negative", {})

    print("  Most positive text:")
    print(f"  Polarity {most_pos.get('polarity', 0):+.4f} — "
          f"\"{truncate_text(most_pos.get('original_text', ''), 50)}\"")
    print()
    print("  Most negative text:")
    print(f"  Polarity {most_neg.get('polarity', 0):+.4f} — "
          f"\"{truncate_text(most_neg.get('original_text', ''), 50)}\"")
    print("  " + "=" * 45)


def print_guide():
    """Prints the interpretation guide for understanding scores."""
    print()
    print("  " + "=" * 55)
    print("  HOW TO INTERPRET RESULTS")
    print("  " + "=" * 55)
    print()
    print("  POLARITY SCORE  (range: -1.0 to +1.0)")
    print_divider("·")
    print("  +0.6 to +1.0  →  Strong positive sentiment")
    print("  +0.2 to +0.5  →  Moderate positive sentiment")
    print("  +0.1 to +0.2  →  Mild positive sentiment")
    print("   0.0          →  Completely neutral")
    print("  -0.1 to -0.2  →  Mild negative sentiment")
    print("  -0.2 to -0.5  →  Moderate negative sentiment")
    print("  -0.6 to -1.0  →  Strong negative sentiment")
    print()
    print("  SUBJECTIVITY SCORE  (range: 0.0 to 1.0)")
    print_divider("·")
    print("  0.0 to 0.3  →  Objective  (factual, data-driven)")
    print("  0.3 to 0.6  →  Balanced   (mix of fact and opinion)")
    print("  0.6 to 1.0  →  Subjective (personal opinion)")
    print()
    print("  CLASSIFICATION THRESHOLDS")
    print_divider("·")
    print("  Polarity >  0.1  →  Classified as POSITIVE")
    print("  Polarity < -0.1  →  Classified as NEGATIVE")
    print("  Otherwise        →  Classified as NEUTRAL")
    print()
    print("  NOTE: TextBlob works best on English text.")
    print("  Sarcasm, slang, and very short texts may")
    print("  produce less accurate results.")
    print("  " + "=" * 55)


# ─────────────────────────────────────────────
#  Menu option handlers
# ─────────────────────────────────────────────

def handle_single_analysis() -> list:
    """Option 1 — Analyze one text entered by the user."""
    print("\n  SINGLE TEXT ANALYSIS")
    print_divider()
    print("  Enter the text you want to analyze.")
    print("  (Press Enter twice or type 'done' to finish)\n")

    user_input = input("  > ").strip()

    if not user_input or user_input.lower() == "done":
        print("  No text entered.")
        return []

    result = analyze_sentiment(user_input)
    print_result(result)
    return [result]


def handle_multi_text_analysis() -> list:
    """Option 2 — Analyze several texts entered one at a time."""
    print("\n  MULTI-TEXT ANALYSIS")
    print_divider()
    print("  Enter one text per line.")
    print("  Type 'done' on a new line when finished.\n")

    texts   = []
    counter = 1

    while True:
        line = input(f"  Text {counter}: ").strip()
        if line.lower() == "done" or line == "":
            if not texts:
                print("  No texts entered.")
                return []
            break
        texts.append(line)
        counter += 1

    print(f"\n  Analyzing {len(texts)} text(s)...")
    results = analyze_batch(texts)

    for i, result in enumerate(results, start=1):
        print_result(result, show_index=i)

    return results


def handle_csv_analysis() -> list:
    """Option 3 — Read texts from a CSV file and analyze them all."""
    print("\n  CSV FILE ANALYSIS")
    print_divider()
    print("  Enter the path to your CSV file.")
    print("  Example: data/sample_reviews.csv\n")

    filepath = input("  File path: ").strip()

    if not filepath:
        print("  No path entered.")
        return []

    # Normalize path separators for Windows
    filepath = filepath.replace("/", os.sep)

    try:
        info = get_file_info(filepath)
        print()
        print(f"  File     : {info['filename']}")
        print(f"  Rows     : {info['row_count']}")
        print(f"  Columns  : {', '.join(info['headers']) if info['headers'] else 'N/A'}")
        print()

        # Let user pick a column if there are multiple
        column = None
        if len(info["headers"]) > 1:
            print("  Which column contains the text to analyze?")
            for idx, h in enumerate(info["headers"], 1):
                print(f"  [{idx}] {h}")
            choice = input("\n  Enter column name or press Enter to auto-detect: ").strip()
            column = choice if choice else None

        texts   = read_csv(filepath, text_column=column)
        print(f"\n  Loaded {len(texts)} text entries.")
        print("  Analyzing...")

        results = analyze_batch(texts)

        # Show first 5 results as a preview
        preview_count = min(5, len(results))
        print(f"\n  Showing first {preview_count} results:\n")
        for i in range(preview_count):
            print_result(results[i], show_index=i + 1)

        if len(results) > preview_count:
            print(f"\n  ... and {len(results) - preview_count} more.")
            print("  Use option [5] to see summary, or [7] to save all results.")

        return results

    except FileNotFoundError as e:
        print(f"\n  Error: {e}")
    except ValueError as e:
        print(f"\n  Error: {e}")
    except Exception as e:
        print(f"\n  Unexpected error: {e}")

    return []


def handle_sample_analysis() -> list:
    """Option 4 — Generate sample CSV then analyze it automatically."""
    print("\n  SAMPLE DATA ANALYSIS")
    print_divider()
    print("  Generating sample_reviews.csv with 25 reviews...")

    filepath = create_sample_csv()
    print(f"  Created: {filepath}")
    print("  Analyzing all 25 sample reviews...")

    texts   = read_csv(filepath)
    results = analyze_batch(texts)

    preview_count = min(5, len(results))
    print(f"\n  Showing first {preview_count} results:\n")
    for i in range(preview_count):
        print_result(results[i], show_index=i + 1)

    print(f"\n  ... {len(results) - preview_count} more results available.")
    print("  Use option [5] to view full summary statistics.")
    return results


def handle_summary(results: list):
    """Option 5 — Compute and display summary statistics."""
    if not results:
        print("\n  No results loaded. Please run an analysis first (options 1-4).")
        return

    summary = get_summary_statistics(results)
    print_summary(summary)


def handle_charts(results: list):
    """Option 6 — Generate all charts from current results."""
    if not results:
        print("\n  No results loaded. Please run an analysis first (options 1-4).")
        return

    print("\n  CHART GENERATION")
    print_divider()

    if len(results) < 3:
        print("  Note: Some charts need at least 3 results to display properly.")

    generate_all_charts(results, output_dir="outputs")
    print("\n  Open the 'outputs/' folder to view your charts.")


def handle_save(results: list):
    """Option 7 — Save current results and summary to files."""
    if not results:
        print("\n  No results to save. Run an analysis first.")
        return

    print("\n  SAVE RESULTS")
    print_divider()

    # Save detailed results CSV
    csv_path = save_results_to_csv(results, output_dir="outputs")
    print(f"  Results saved  -> {csv_path}")

    # Save summary report TXT
    summary  = get_summary_statistics(results)
    txt_path = save_summary_to_txt(summary, output_dir="outputs")
    print(f"  Summary saved  -> {txt_path}")

    print("\n  Both files are in the 'outputs/' folder.")


# ─────────────────────────────────────────────
#  Main application loop
# ─────────────────────────────────────────────

def main():
    clear_screen()
    print_banner()

    # Session state — holds the last batch of results
    current_results = []

    while True:
        print()
        print_menu()

        choice = input("  Enter your choice: ").strip()

        # ── Option 1: Single text ──────────────────
        if choice == "1":
            current_results = handle_single_analysis()

        # ── Option 2: Multiple texts manually ──────
        elif choice == "2":
            current_results = handle_multi_text_analysis()

        # ── Option 3: CSV file ─────────────────────
        elif choice == "3":
            current_results = handle_csv_analysis()

        # ── Option 4: Sample data ──────────────────
        elif choice == "4":
            current_results = handle_sample_analysis()

        # ── Option 5: Summary statistics ──────────
        elif choice == "5":
            handle_summary(current_results)

        # ── Option 6: Charts ──────────────────────
        elif choice == "6":
            handle_charts(current_results)

        # ── Option 7: Save to file ─────────────────
        elif choice == "7":
            handle_save(current_results)

        # ── Option 8: Guide ───────────────────────
        elif choice == "8":
            print_guide()

        # ── Option 0: Exit ────────────────────────
        elif choice == "0":
            print()
            print_divider()
            print("  Thank you for using the Sentiment Analysis System.")
            print("  Exiting...")
            print_divider()
            print()
            break

        # ── Invalid input ─────────────────────────
        else:
            print("\n  Invalid choice. Please enter a number from 0 to 8.")

        # Pause before returning to menu
        print()
        input("  Press Enter to return to the menu...")
        clear_screen()
        print_banner()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()