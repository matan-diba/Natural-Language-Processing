# Automated Classification of Clinical Trial Outcomes using BioBERT & NLP

An end-to-end Natural Language Processing (NLP) pipeline designed to automatically classify clinical trial outcomes (positive/significant vs. negative/non-significant) directly from scientific abstracts extracted via the PubMed API.

## 📌 Project Overview
In medical and clinical research, thousands of controlled trials are published regularly, creating a massive information overload. Identifying whether a trial demonstrated statistically significant efficacy typically requires laborious manual literature review. This project automates the extraction and classification of clinical trial outcomes, comparing traditional Machine Learning approaches with domain-adapted Transformer architectures.

## 🔬 Methodology & Architecture
1. **Automated Data Retrieval:** Extracted ~400 rigorous randomized controlled trial abstracts via the NCBI Entrez PubMed API (`BioPython`).
2. **Semi-Automated Labeling & Text Preprocessing:** Implemented rule-based pattern matching (p-values, statistical significance, and outcome phrases) focused on conclusion sections; removed HTML/XML artifacts and specialized biomedical noise while preserving mathematical inequalities (`<`, `>`, `=`).
3. **Modeling Strategies:**
   * **Baseline Model:** TF-IDF Vectorizer (word & bi-gram representations, max 1,000 features) + Class-Weighted Logistic Regression.
   * **Transformer Model:** Domain-specific `dmis-lab/biobert-v1.1` fine-tuned for sequence classification with cross-entropy loss weighted for severe class imbalance.
   * **Hyperparameter Tuning:** Explored learning rates, weight decay regularization, and extended training epochs to optimize recall on the minority class.

## 📊 Key Results & Findings

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (TF-IDF + LogReg)** | 0.7000 | 0.3043 | 0.4667 | **0.3684** |
| **BioBERT (Generic / 3 Epochs)** | **0.6875** | **0.3214** | 0.6000 | **0.4186** |
| **BioBERT (Tuned / 5 Epochs)** | 0.3000 | 0.1746 | **0.7333** | 0.2821 |

* **Domain Insights:** The baseline linear model achieved a balanced F1-score due to direct lexical alignment with statistical keywords. In contrast, fine-tuned BioBERT significantly boosted recall (up to 73.3%), demonstrating high sensitivity in detecting positive medical breakthroughs—a crucial trait for clinical decision-support systems where false negatives are critical.
* **Qualitative Error Analysis:** Detailed analysis revealed cases where BioBERT correctly captured holistic paragraph semantics even when abstracts were truncated or contained complex hedging language, outperforming rigid regex rules.

## 🛠 Tech Stack
* **Language:** Python
* **Libraries:** PyTorch, Hugging Face (`transformers`, `datasets`, `accelerate`, `evaluate`), scikit-learn, BioPython, Pandas, NumPy, Matplotlib, Seaborn
