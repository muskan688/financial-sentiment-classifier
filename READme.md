# 🏦 Financial Sentiment Classifier

Fine-tuned FinBERT model that classifies financial news sentences 
as Positive, Neutral, or Negative with 96.3% accuracy.

---

## 📊 Dataset
- **Source:** FinanceMTEB/financial_phrasebank (Hugging Face)
- **Size:** 1,264 train + 1,000 test sentences
- **Classes:** Negative (0), Neutral (1), Positive (2)
- **Challenge:** Imbalanced dataset (779 neutral vs 165 negative)
- **Fix:** Used class weights to handle imbalance

---

## 🛠️ Approach
1. **Data Cleaning** → lowercase, remove punctuation (vectorized pandas)
2. **Tokenization** → FinBERT tokenizer, max 128 tokens, padding + truncation
3. **Class Imbalance** → class weights (negative: 2.55x penalty)
4. **Fine-tuning** → 3 epochs, learning rate 2e-5, macro F1 evaluation

---

## 📈 Results

| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| DistilBERT | 95.1% | 0.930 |
| FinBERT | 96.3% | 0.944 |

FinBERT outperforms DistilBERT because it was
pre-trained specifically on financial text.

---

## 🔍 Example Predictions

| Sentence | Prediction | Confidence |
|----------|-----------|------------|
| Apple reported record breaking profits | POSITIVE 📈 | 97.7% |
| The company filed for bankruptcy | NEUTRAL 😐 | 98.9% |
| Operating costs fell sharply | NEGATIVE 📉 | 84.9% |

---

## 🚀 Future Improvements
1. More negative training examples
2. Larger financial news dataset
3. Try 5 epochs instead of 3
4. Deep dive error analysis on misclassified examples

---

## 🛠️ Tech Stack
- Python
- Hugging Face Transformers (FinBERT)
- PyTorch
- Google Colab (NVIDIA T4 GPU)
- Pandas, NumPy, Scikit-learn

---

## 📚 What I Learned
- Fine-tuning transformer models on domain-specific data
- Handling class imbalance with weighted loss
- Difference between DistilBERT vs FinBERT
- Vectorized data cleaning with pandas
