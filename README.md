# Financial Sentiment Classifier

Fine-tunes FinBERT to classify sentences from financial news as negative, neutral or positive, and compares it with a DistilBERT baseline trained with the same settings.

## Results

Test set: 1,000 sentences. Each model was trained 3 times (seeds 42, 43, 44). Values are mean ± standard deviation across the 3 runs.

| Model | Accuracy | Macro F1 |
|---|---|---|
| FinBERT (`ProsusAI/finbert`) | 0.967 ± 0.001 | 0.951 ± 0.001 |
| DistilBERT (`distilbert-base-uncased`) | 0.956 ± 0.000 | 0.939 ± 0.001 |

FinBERT is about 1 point ahead on accuracy and 1.2 points on macro F1, in all three seeds. The test set is small, so the gap is modest. A likely reason is that FinBERT was pre-trained on financial text, but I did not test that.

![Confusion matrices](confusion_matrices.png)

In the last FinBERT run, 33 of 1,000 test sentences were wrong. Most errors are positive sentences predicted as negative (12), negative predicted as positive (6) and neutral predicted as positive (8).

Raw scores for every run are in [`results.json`](results.json).

## Data

- Source: [`FinanceMTEB/financial_phrasebank`](https://huggingface.co/datasets/FinanceMTEB/financial_phrasebank) on Hugging Face.
- 1,264 training and 1,000 test sentences.
- Labels: negative (0), neutral (1), positive (2).
- The classes are imbalanced: 165 negative and 779 neutral sentences in the training set.

## Method

- Tokenizer and model: `ProsusAI/finbert` and `distilbert-base-uncased`, 3-class classification head.
- Class imbalance: weighted cross-entropy loss with balanced class weights (negative 2.55, neutral 0.54, positive 1.32).
- Settings (identical for both models): 4 epochs, learning rate 2e-5, batch size 16, weight decay 0.01, max length 128 tokens, mixed precision on a Colab T4 GPU.
- No hyperparameter tuning. The test set is used once, for the final scores. There is no separate validation set.

## Known limitations

- Small test set, one dataset. Results may not carry over to other financial text.
- Errors can be confident. In the original notebook (`sentiment.ipynb`), "The company filed for bankruptcy today" was predicted neutral with 98.9% confidence, and "Sales remained unchanged from last year" was predicted negative. I have not done a full error analysis.

## Files

| File | Content |
|---|---|
| `train.py` | Training and evaluation code (weighted loss, 3 seeds, summary, confusion matrices). Command-line entry point. |
| `tests/test_train.py` | Unit tests for class weights, weighted loss, scoring, summary and the committed `results.json`. No model download needed. |
| `requirements.txt` | Python dependencies. |
| `finbert_vs_distilbert.ipynb` | Notebook version of the same experiment. Source of the numbers above. |
| `sentiment.ipynb` | First exploration notebook: data checks, tokenisation, class weights, example predictions. |
| `results.json` | Accuracy and macro F1 for every run. |
| `confusion_matrices.png` | Confusion matrices from the last seed. |

## How to run

Requires Python 3.9 or newer. A GPU is strongly recommended (the reported numbers used a Colab T4); on CPU a full run takes hours.

```bash
git clone https://github.com/muskan688/financial-sentiment-classifier.git
cd financial-sentiment-classifier
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Full experiment (both models, seeds 42 to 44, about 10 to 20 minutes on a T4). It overwrites `results.json` and `confusion_matrices.png`:

```bash
python train.py
```

Quick smoke run (one model, one seed, one epoch):

```bash
python train.py --models FinBERT --seeds 42 --epochs 1 --results smoke.json --plot smoke.png
```

Run the tests (fast, no GPU or model download):

```bash
python -m pytest
```

You can also run the notebook `finbert_vs_distilbert.ipynb` in Google Colab with a T4 GPU runtime.

## Possible next steps

- Error analysis on the misclassified sentences.
- Test on a second financial dataset.
- Add a validation split and tune the learning rate and number of epochs.
