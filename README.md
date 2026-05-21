# decodelabs_task
In this repository I will complete the tasks assigned by decode labs for better learning and growing






---



This repository contains three progressive AI engineering projects completed during the **DecodeLabs Industrial Training Program (Batch 2026)**. Each project builds on the previous one, forming a complete ML pipeline portfolio — from foundational classification to content-based recommendation systems.

| Project | Title | Algorithm | Dataset |
|:---:|---|---|---|
|  P1 | Rule-Based AI Logic | Heuristic / If-Then | Custom |
|  P2 | Data Classification Using AI | K-Nearest Neighbors | Iris (UCI) |
|  P3 | AI Recommendation Logic | TF-IDF + Cosine Similarity | Custom Job Roles |

---

## 🗂 Repository Structure

```
decodelabs-ai-kit/
│
├── project_2/
│   └── iris_knn_classifier.py       # KNN classification on Iris dataset
│
├── project_3/
│   └── tech_stack_recommender.py    # Content-based career recommender
│
└── README.md
```

---

## 🔵 Project 2 — Data Classification Using AI

### Goal
Build a basic classification model that learns to distinguish between three species of the Iris flower using the **K-Nearest Neighbors** algorithm.

### The Pipeline

```
Raw Data → Feature Scaling → Train/Test Split → KNN → Confusion Matrix + F1 Score
```

### Key Concepts Implemented

- **StandardScaler** — normalizes features to mean=0, variance=1, eliminating distance bias
- **Train-Test Split (80/20)** — shuffled to remove order bias before splitting
- **Elbow Method** — loops K from 1–30, plots error rate curve to find optimal K
- **KNN Classification** — proximity principle: similar things exist in close proximity
- **Confusion Matrix** — visualizes TP, FP, FN, TN per class
- **F1 Score** — harmonic mean of precision and recall; robust against imbalanced data

### Dataset

The **Iris Benchmark** — 150 balanced samples, 3 classes (Setosa, Versicolor, Virginica), 4 features (sepal length, sepal width, petal length, petal width).

### Run It

```bash
pip install scikit-learn matplotlib seaborn numpy
python project_2/iris_knn_classifier.py
```

### Sample Output

```
Dataset Shape: (150, 4)
Classes: ['setosa' 'versicolor' 'virginica']
Optimal K from elbow method: 7

F1 Score (weighted): 0.9667

              precision    recall  f1-score
    setosa       1.00      1.00      1.00
versicolor       0.92      1.00      0.96
 virginica       1.00      0.90      0.95
```

> Elbow plot saved as `k_tuning_elbow.png` · Confusion matrix saved as `confusion_matrix.png`

---

## 🟠 Project 3 — AI Recommendation Logic

### Goal
Build a **Tech Stack Recommender** — a content-based filtering engine that maps a user's raw skills to the most relevant career paths using TF-IDF weighting and Cosine Similarity scoring.

### The Pipeline (IPO Framework)

```
User Skills (Input) → TF-IDF Vectorization → Cosine Similarity Scoring → Top-N Ranked Roles (Output)
```

### Key Concepts Implemented

- **Content-Based Filtering** — driven by item attributes, independent of other users' behavior; no historical interaction data required
- **Vector Mapping** — qualitative skill tags transformed into numerical arrays in a shared vocabulary space
- **TF-IDF Weighting** — generic terms (like "software") penalized via IDF; specific, descriptive skills rewarded with higher weight
- **Cosine Similarity** — measures angular alignment between user vector and job role vectors; invariant to vector magnitude (solves the Euclidean distance flaw)
- **Cold Start Handling** — onboarding survey (minimum 3 inputs) bootstraps the user profile vector
- **Top-N Filtering** — sorts by score descending, truncates to Top 3 to prevent choice overload

### Why Cosine Over Euclidean?

Euclidean distance is sensitive to vector magnitude — a job role with more tags will always appear "farther" even if perfectly aligned in direction. Cosine similarity focuses purely on **orientation**, making it the industry standard for text-based recommendation.

### Run It

```bash
pip install pandas scikit-learn
python project_3/tech_stack_recommender.py
```

### Sample Output

```
Your skills: python, docker, aws

══════════════════════════════════════════════════════
   TOP 3 CAREER PATH RECOMMENDATIONS FOR YOU
══════════════════════════════════════════════════════

  Rank 1: DevOps Engineer          Match Score: 68.4%
  Rank 2: Cloud Architect          Match Score: 54.1%
  Rank 3: Data Engineer            Match Score: 41.7%

Full Similarity Scores:
  DevOps Engineer       [████████████████████------]  68.4%
  Cloud Architect       [████████████████-----------]  54.1%
  Data Engineer         [████████████---------------]  41.7%
  ...
```

---

## ⚙️ Installation

**Clone the repository:**
```bash
git clone https://github.com/your-username/decodelabs-ai-kit.git
cd decodelabs-ai-kit
```

**Install all dependencies at once:**
```bash
pip install scikit-learn pandas matplotlib seaborn numpy
```

**Requirements summary:**

| Library | Purpose |
|---|---|
| `scikit-learn` | KNN, TF-IDF, Cosine Similarity, StandardScaler |
| `pandas` | DataFrame management |
| `matplotlib` | Elbow curve & confusion matrix plots |
| `seaborn` | Heatmap visualization |
| `numpy` | Numerical operations |

---

## 🧠 Concepts at a Glance

```
Supervised Learning          →  You provide labeled data; machine derives the logic
KNN (K-Nearest Neighbors)    →  Classify by majority vote of K closest neighbors
StandardScaler               →  Mean = 0, Variance = 1 (removes feature bias)
F1 Score                     →  Harmonic mean of Precision & Recall
Confusion Matrix             →  TP / FP / FN / TN breakdown per class
Content-Based Filtering      →  Match user profile to item attributes directly
TF-IDF                       →  Term Frequency × Inverse Document Frequency weighting
Cosine Similarity            →  Angular distance between vectors (magnitude-invariant)
Cold Start Problem           →  Zero-vector issue for new users; solved via onboarding
Top-N Filtering              →  Truncate output to prevent choice overload
```

---

## 🗺 Learning Progression

```
Project 2                    Project 3                    Next →
─────────────────────────    ─────────────────────────    ──────────────
Passive Classification       Active Prediction            Deep Learning
Label existing data    →     Predict user needs     →     CNNs & NLP
KNN (distance)               Cosine (angle)               Neural Networks
Iris species                 Career paths                 Computer Vision
```

---



This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---

<div align="center">

Built with 🧡 during the **DecodeLabs Industrial Training Program**  
[www.decodelabs.tech](https://www.decodelabs.tech) · Greater Lucknow, India

</div
