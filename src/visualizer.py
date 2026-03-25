"""
visualizer.py
-------------
Generates all charts and visual outputs for the sentiment analysis results.
Saves every chart as a PNG file inside the outputs/ folder.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime


# ─────────────────────────────────────────────
#  Global style configuration
# ─────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = "outputs"

# Consistent color scheme used across all charts
COLORS = {
    "Positive": "#2ecc71",   # green
    "Negative": "#e74c3c",   # red
    "Neutral" : "#95a5a6",   # gray
}

PALETTE = [COLORS["Positive"], COLORS["Negative"], COLORS["Neutral"]]


def _setup_style():
    """Applies a clean consistent style to all matplotlib charts."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family"       : "DejaVu Sans",
        "font.size"         : 11,
        "axes.titlesize"    : 14,
        "axes.titleweight"  : "bold",
        "axes.labelsize"    : 11,
        "figure.dpi"        : 120,
        "savefig.bbox"      : "tight",
        "savefig.dpi"       : 150,
    })


def _save_figure(fig, name: str, output_dir: str) -> str:
    """
    Saves a matplotlib figure as a PNG file with a timestamp in the name.
    Returns the full filepath of the saved image.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{name}_{timestamp}.png"
    filepath  = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)
    return filepath


def _ensure_results(results: list, min_count: int = 1) -> bool:
    """Returns True if there are enough valid results to plot."""
    valid = [r for r in results if r.get("is_valid")]
    return len(valid) >= min_count


# ─────────────────────────────────────────────
#  Chart 1 — Sentiment Distribution Bar Chart
# ─────────────────────────────────────────────

def plot_sentiment_bar(results: list, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """
    Bar chart showing how many texts fall into each sentiment category.
    Best for quickly seeing the overall distribution at a glance.
    """
    if not _ensure_results(results):
        print("Not enough data to generate bar chart.")
        return ""

    _setup_style()

    # Count each category
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for r in results:
        if r.get("is_valid"):
            counts[r["sentiment"]] += 1

    labels = list(counts.keys())
    values = list(counts.values())
    colors = [COLORS[label] for label in labels]
    total  = sum(values)

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, values, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)

    # Add count + percentage labels on top of each bar
    for bar, value in zip(bars, values):
        pct = f"{value / total * 100:.1f}%" if total > 0 else "0%"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{value}\n({pct})",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )

    ax.set_title("Sentiment Distribution", pad=15)
    ax.set_xlabel("Sentiment Category")
    ax.set_ylabel("Number of Texts")
    ax.set_ylim(0, max(values) * 1.25 if values else 1)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Subtle background grid on y-axis only
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return _save_figure(fig, "sentiment_bar", output_dir)


# ─────────────────────────────────────────────
#  Chart 2 — Sentiment Distribution Pie Chart
# ─────────────────────────────────────────────

def plot_sentiment_pie(results: list, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """
    Pie chart showing percentage share of each sentiment category.
    Only slices with at least one entry are shown.
    """
    if not _ensure_results(results):
        print("Not enough data to generate pie chart.")
        return ""

    _setup_style()

    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for r in results:
        if r.get("is_valid"):
            counts[r["sentiment"]] += 1

    # Remove zero-count categories so pie doesn't show empty slices
    filtered = {k: v for k, v in counts.items() if v > 0}
    labels   = list(filtered.keys())
    values   = list(filtered.values())
    colors   = [COLORS[label] for label in labels]

    fig, ax = plt.subplots(figsize=(7, 6))

    wedges, texts, autotexts = ax.pie(
        values,
        labels      = labels,
        colors      = colors,
        autopct     = "%1.1f%%",
        startangle  = 140,
        pctdistance = 0.82,
        wedgeprops  = {"edgecolor": "white", "linewidth": 2},
    )

    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight("bold")
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    ax.set_title("Sentiment Share", pad=20)

    # Legend with exact counts
    legend_labels = [f"{label}: {val}" for label, val in zip(labels, values)]
    patches = [mpatches.Patch(color=COLORS[l], label=ll)
               for l, ll in zip(labels, legend_labels)]
    ax.legend(handles=patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)

    fig.tight_layout()
    return _save_figure(fig, "sentiment_pie", output_dir)


# ─────────────────────────────────────────────
#  Chart 3 — Polarity Score Histogram
# ─────────────────────────────────────────────

def plot_polarity_histogram(results: list, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """
    Histogram of polarity scores across all analyzed texts.
    Shows the full distribution from -1.0 to +1.0 with threshold lines.
    """
    if not _ensure_results(results, min_count=3):
        print("Need at least 3 results to generate histogram.")
        return ""

    _setup_style()

    polarities = [r["polarity"] for r in results if r.get("is_valid")]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(polarities, bins=20, range=(-1, 1),
            color="#3498db", edgecolor="white",
            linewidth=0.8, alpha=0.85)

    # Threshold lines
    ax.axvline(x= 0.1, color=COLORS["Positive"], linestyle="--",
               linewidth=1.5, label="Positive threshold (+0.1)")
    ax.axvline(x=-0.1, color=COLORS["Negative"], linestyle="--",
               linewidth=1.5, label="Negative threshold (−0.1)")
    ax.axvline(x=sum(polarities) / len(polarities),
               color="#e67e22", linestyle="-",
               linewidth=2, label=f"Mean ({sum(polarities)/len(polarities):+.3f})")

    # Shade the three sentiment regions lightly
    ax.axvspan(-1.0, -0.1, alpha=0.05, color=COLORS["Negative"])
    ax.axvspan(-0.1,  0.1, alpha=0.05, color=COLORS["Neutral"])
    ax.axvspan( 0.1,  1.0, alpha=0.05, color=COLORS["Positive"])

    ax.set_title("Polarity Score Distribution")
    ax.set_xlabel("Polarity Score  (−1 = Very Negative  →  +1 = Very Positive)")
    ax.set_ylabel("Number of Texts")
    ax.set_xlim(-1.1, 1.1)
    ax.legend(fontsize=9, framealpha=0.7)

    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return _save_figure(fig, "polarity_histogram", output_dir)


# ─────────────────────────────────────────────
#  Chart 4 — Polarity vs Subjectivity Scatter
# ─────────────────────────────────────────────

def plot_polarity_vs_subjectivity(results: list, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """
    Scatter plot with polarity on the X axis and subjectivity on the Y axis.
    Each dot is one text, colored by its sentiment classification.
    Reveals whether opinionated text tends to be more positive or negative.
    """
    if not _ensure_results(results, min_count=3):
        print("Need at least 3 results to generate scatter plot.")
        return ""

    _setup_style()

    valid = [r for r in results if r.get("is_valid")]

    polarities    = [r["polarity"]     for r in valid]
    subjectivities = [r["subjectivity"] for r in valid]
    sentiments    = [r["sentiment"]    for r in valid]
    dot_colors    = [COLORS[s]         for s in sentiments]

    fig, ax = plt.subplots(figsize=(9, 6))

    scatter = ax.scatter(
        polarities, subjectivities,
        c=dot_colors, s=80,
        alpha=0.75, edgecolors="white", linewidths=0.8
    )

    # Quadrant dividers
    ax.axvline(x=0,   color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Quadrant labels (subtle)
    quad_style = dict(fontsize=8, color="gray", alpha=0.7)
    ax.text( 0.85,  0.92, "Positive\nOpinion",  ha="center", transform=ax.transAxes, **quad_style)
    ax.text( 0.15,  0.92, "Negative\nOpinion",  ha="center", transform=ax.transAxes, **quad_style)
    ax.text( 0.85,  0.08, "Positive\nFact",     ha="center", transform=ax.transAxes, **quad_style)
    ax.text( 0.15,  0.08, "Negative\nFact",     ha="center", transform=ax.transAxes, **quad_style)

    ax.set_title("Polarity vs Subjectivity")
    ax.set_xlabel("Polarity  (−1 = Negative  →  +1 = Positive)")
    ax.set_ylabel("Subjectivity  (0 = Objective  →  1 = Subjective)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 1.05)

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLORS["Positive"], label="Positive"),
        mpatches.Patch(color=COLORS["Negative"], label="Negative"),
        mpatches.Patch(color=COLORS["Neutral"],  label="Neutral"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              framealpha=0.8, fontsize=10)

    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return _save_figure(fig, "polarity_vs_subjectivity", output_dir)


# ─────────────────────────────────────────────
#  Chart 5 — Intensity Breakdown Bar Chart
# ─────────────────────────────────────────────

def plot_intensity_breakdown(results: list, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """
    Grouped bar chart showing Strong / Moderate / Mild counts
    broken down by sentiment category (Positive and Negative).
    Neutral texts are excluded as intensity is less meaningful for them.
    """
    if not _ensure_results(results, min_count=2):
        print("Not enough data to generate intensity chart.")
        return ""

    _setup_style()

    intensities = ["Strong", "Moderate", "Mild"]
    pos_counts  = {i: 0 for i in intensities}
    neg_counts  = {i: 0 for i in intensities}

    for r in results:
        if not r.get("is_valid"):
            continue
        if r["sentiment"] == "Positive":
            pos_counts[r["intensity"]] += 1
        elif r["sentiment"] == "Negative":
            neg_counts[r["intensity"]] += 1

    x      = range(len(intensities))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_pos = ax.bar(
        [i - width / 2 for i in x],
        [pos_counts[i] for i in intensities],
        width, label="Positive",
        color=COLORS["Positive"], edgecolor="white", linewidth=1
    )
    bars_neg = ax.bar(
        [i + width / 2 for i in x],
        [neg_counts[i] for i in intensities],
        width, label="Negative",
        color=COLORS["Negative"], edgecolor="white", linewidth=1
    )

    # Count labels on bars
    for bar in list(bars_pos) + list(bars_neg):
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.1,
                str(int(h)), ha="center", va="bottom", fontsize=9
            )

    ax.set_title("Sentiment Intensity Breakdown")
    ax.set_xlabel("Intensity Level")
    ax.set_ylabel("Number of Texts")
    ax.set_xticks(list(x))
    ax.set_xticklabels(intensities)
    ax.set_ylim(0, max(max(pos_counts.values()), max(neg_counts.values())) * 1.3 + 1)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(framealpha=0.8)

    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return _save_figure(fig, "intensity_breakdown", output_dir)


# ─────────────────────────────────────────────
#  Master function — generate all charts at once
# ─────────────────────────────────────────────

def generate_all_charts(results: list, output_dir: str = DEFAULT_OUTPUT_DIR) -> list:
    """
    Runs all five chart functions in sequence.
    Returns a list of filepaths for every chart that was successfully saved.
    Prints a status line for each chart.
    """
    saved_files = []

    chart_functions = [
        ("Sentiment Bar Chart",          plot_sentiment_bar),
        ("Sentiment Pie Chart",          plot_sentiment_pie),
        ("Polarity Histogram",           plot_polarity_histogram),
        ("Polarity vs Subjectivity",     plot_polarity_vs_subjectivity),
        ("Intensity Breakdown",          plot_intensity_breakdown),
    ]

    print("\n  Generating charts...")
    print("  " + "-" * 35)

    for name, func in chart_functions:
        try:
            filepath = func(results, output_dir)
            if filepath:
                saved_files.append(filepath)
                print(f"  [OK] {name}")
                print(f"       Saved -> {filepath}")
        except Exception as e:
            print(f"  [SKIP] {name} — {e}")

    print("  " + "-" * 35)
    print(f"  {len(saved_files)} chart(s) saved to '{output_dir}/'")

    return saved_files