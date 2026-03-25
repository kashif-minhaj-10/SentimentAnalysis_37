# Sentiment Analysis System

A Python-based Sentiment Analysis System that classifies textual data as
Positive, Negative, or Neutral using TextBlob and NLTK.

---

## Problem Statement

Manually reading and classifying large volumes of text such as customer
reviews, social media posts, or feedback forms is time-consuming and
inconsistent. This system automates that process using Natural Language
Processing techniques to provide fast, consistent, and quantified sentiment
classification.

---

## Features

- Analyze a single text entry interactively
- Analyze multiple texts in one session
- Load and analyze data from CSV files
- Polarity score (-1.0 to +1.0) and subjectivity score (0.0 to 1.0)
- Classifies sentiment as Positive, Negative, or Neutral
- Intensity classification — Strong, Moderate, or Mild
- Subjectivity classification — Subjective, Balanced, or Objective
- Summary statistics with percentages and averages
- Five chart types saved as PNG files
- Export results to CSV and summary to TXT

---

## Tools and Technologies

| Tool / Library | Purpose                                          |
| -------------- | ------------------------------------------------ |
| Python 3.x     | Core programming language                        |
| TextBlob       | Sentiment polarity and subjectivity analysis     |
| NLTK           | Natural language processing backend for TextBlob |
| Matplotlib     | Chart and graph generation                       |
| Seaborn        | Enhanced chart styling                           |
| Pandas         | Data handling for CSV operations                 |
| VS Code        | Development environment                          |

---

## Project Structure

```
SentimentAnalysis/
├── src/
│   ├── __init__.py
│   ├── preprocessor.py      # Text cleaning and validation
│   ├── analyzer.py          # Core sentiment analysis engine
│   ├── file_handler.py      # CSV reading and result saving
│   └── visualizer.py        # Chart generation
├── data/
│   └── sample_reviews.csv   # Sample dataset (25 reviews)
├── outputs/                 # Generated charts and result files
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Installation

**Step 1 — Clone the repository**

```
git clone https://github.com/YOUR_USERNAME/SentimentAnalysis_YOURROLLNUMBER.git
cd SentimentAnalysis_YOURROLLNUMBER
```

**Step 2 — Create and activate virtual environment**

```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

**Step 3 — Install dependencies**

```
pip install -r requirements.txt
python -m textblob.download_corpora
```

---

## How to Run

```
python main.py
```

---

## Menu Options

| Option | Description                        |
| ------ | ---------------------------------- |
| 1      | Analyze a single text              |
| 2      | Analyze multiple texts manually    |
| 3      | Analyze texts from a CSV file      |
| 4      | Generate sample CSV and analyze it |
| 5      | View summary statistics            |
| 6      | Generate charts and visualizations |
| 7      | Save results to output files       |
| 8      | View score interpretation guide    |
| 0      | Exit                               |

---

## How Sentiment is Classified

| Polarity Score        | Classification |
| --------------------- | -------------- |
| Greater than +0.1     | Positive       |
| Between -0.1 and +0.1 | Neutral        |
| Less than -0.1        | Negative       |

| Subjectivity Score  | Classification |
| ------------------- | -------------- |
| Greater than 0.6    | Subjective     |
| Between 0.3 and 0.6 | Balanced       |
| Less than 0.3       | Objective      |

---

## Sample Output

```
  ═══════════════════════════════════════
  Text     : This product is absolutely fantastic!
  Words    : 6
  ·······································
  Sentiment    : (+)  Positive
  Intensity    : Strong
  Subjectivity : Subjective
  ·······································
  Polarity     : +0.6250
  Subjectivity : 0.8000
  ·······································
  Interpretation:
  The text expresses a strong positive sentiment
  (polarity: +0.6250). It is highly subjective
  (subjectivity: 0.8000), meaning it is opinion-based.
  ═══════════════════════════════════════
```

---

## Output Files

All outputs are saved to the `outputs/` folder automatically:

- `sentiment_results_TIMESTAMP.csv` — full results for every text analyzed
- `sentiment_summary_TIMESTAMP.txt` — overall statistics report
- `sentiment_bar_TIMESTAMP.png` — sentiment distribution bar chart
- `sentiment_pie_TIMESTAMP.png` — sentiment share pie chart
- `polarity_histogram_TIMESTAMP.png` — polarity score distribution
- `polarity_vs_subjectivity_TIMESTAMP.png` — scatter plot
- `intensity_breakdown_TIMESTAMP.png` — intensity grouped bar chart

---

## Conclusion

This system demonstrates a complete NLP pipeline from raw text input through
preprocessing, analysis, classification, visualization, and export. TextBlob
provides an accessible yet effective foundation for sentiment analysis
suitable for real-world applications such as product review monitoring,
social media tracking, and customer feedback evaluation.

---

## Author

Name : KASHIF MINHAJ PK  
Roll Number : 37
GitHub : https://github.com/YOUR_USERNAME/SentimentAnalysis_YOURROLLNUMBER
