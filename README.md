### 🛍️ Product Matching: Jaccard, TF-IDF, and DeBERTa Approaches
This repository implements and compares three product matching methods on the WDC Product Matching Dataset: https://webdatacommons.org/largescaleproductcorpus/wdc-products/. The task is to determine whether two product listings refer to the same real-world item. We evaluate methods ranging from basic token similarity to contextual deep learning models.

## 📂 Repository Structure

| Notebook                             | Description                                                                 |
|--------------------------------------|-----------------------------------------------------------------------------|
| `Preprocessing.ipynb`               | Extracts, cleans, and transforms raw WDC data into model-ready format.     |
| `Feature_Similarity_Methods.ipynb`  | Implements basic feature similarity (Jaccard, sequence match, weighted).   |
| `CosineSimilarity+TFIDF+LogReg.ipynb` | Implements TF-IDF vectorization + cosine similarity + logistic regression. |
| `DeBerta.ipynb`                     | Fine-tunes the DeBERTa model for pairwise classification using Hugging Face Transformers. |

### 🧪 Approaches Compared
Jaccard + Feature Similarity
Lexical comparison using token overlap on title, brand, and description. Includes weighted ensemble scoring.

TF-IDF + Cosine Similarity + Logistic Regression
Transforms text to sparse vectors with TF-IDF and classifies similarity using logistic regression.

DeBERTa-based Deep Learning Classifier
Fine-tuned DeBERTa (v3-small) model trained on tokenized text pairs using curriculum learning across 20pair → 50pair → 80pair datasets.

## 📈 Key Results

| Model                                  | Accuracy | F1 Score |
|----------------------------------------|----------|----------|
| Weighted Feature Similarity (Jaccard)  | 0.68     | 0.55     |
| TF-IDF + Cosine Similarity + LogReg    | 0.83     | 0.62     |
| TF-IDF + Cosine + LogReg + Price Diff  | 0.83     | 0.62     |
| DeBERTa (Combined Dataset)             | 0.91     | 0.67     |
| DeBERTa (Curriculum Learning - Final)  | 0.93     | 0.75     |

### 📌 Highlights
Preprocessing: Includes price normalization, multilingual filtering, and formatting.

Curriculum Learning: DeBERTa performance improves by training progressively on simpler to harder datasets.

### 🧰 Dependencies
Python 3.9+

PyTorch

Hugging Face Transformers

Scikit-learn

Pandas, NumPy

langdetect
