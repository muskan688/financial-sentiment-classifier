"""Fine-tune FinBERT and DistilBERT on Financial PhraseBank and compare them.

Usage:
    python train.py                      # both models, seeds 42-44, writes results.json
    python train.py --models FinBERT --seeds 42 --epochs 1   # quick smoke run
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

DATASET = "FinanceMTEB/financial_phrasebank"
NAMES = ["negative", "neutral", "positive"]
MODELS = {"FinBERT": "ProsusAI/finbert", "DistilBERT": "distilbert-base-uncased"}
SEEDS = [42, 43, 44]


def balanced_class_weights(labels):
    """Balanced class weights for labels in {0, 1, 2}, as a float tensor."""
    w = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=np.asarray(labels))
    return torch.tensor(w, dtype=torch.float)


def weighted_cross_entropy(logits, labels, weights):
    return torch.nn.functional.cross_entropy(logits, labels, weight=weights.to(logits.device))


def score(y_true, y_pred):
    return {"acc": accuracy_score(y_true, y_pred), "f1": f1_score(y_true, y_pred, average="macro")}


def summarise(results):
    """Mean and std of accuracy and macro F1 across runs, keeping the raw runs."""
    summary = {}
    for label, runs in results.items():
        a = np.array([r["acc"] for r in runs])
        f = np.array([r["f1"] for r in runs])
        summary[label] = {"acc_mean": a.mean(), "acc_std": a.std(),
                          "f1_mean": f.mean(), "f1_std": f.std(), "runs": runs}
    return summary


def run(model_name, seed, raw, weights, epochs=4):
    """Train one model with one seed and score it once on the test set."""
    # Imported here so the helpers above can be tested without transformers installed.
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments, set_seed)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = weighted_cross_entropy(outputs.logits, labels, weights)
            return (loss, outputs) if return_outputs else loss

    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = lambda b: tok(b["text"], truncation=True, padding="max_length", max_length=128)
    cols = ["input_ids", "attention_mask", "labels"]
    tr = raw["train"].map(enc, batched=True).rename_column("label", "labels")
    te = raw["test"].map(enc, batched=True).rename_column("label", "labels")
    tr.set_format("torch", columns=cols)
    te.set_format("torch", columns=cols)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3,
        id2label=dict(enumerate(NAMES)), label2id={n: i for i, n in enumerate(NAMES)})
    args = TrainingArguments(
        output_dir=f"out_{seed}", num_train_epochs=epochs, learning_rate=2e-5,
        per_device_train_batch_size=16, per_device_eval_batch_size=64,
        weight_decay=0.01, seed=seed, save_strategy="no", report_to="none",
        fp16=torch.cuda.is_available())
    trainer = WeightedTrainer(model=model, args=args, train_dataset=tr)
    trainer.train()
    preds = trainer.predict(te).predictions.argmax(-1)
    y = np.array(raw["test"]["label"])
    return score(y, preds), y, preds


def save_confusion_matrices(y, last_preds, path="confusion_matrices.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(last_preds), figsize=(4.5 * len(last_preds), 4), squeeze=False)
    for ax, (label, preds) in zip(axes[0], last_preds.items()):
        cm = confusion_matrix(y, preds)
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{label} (last seed)")
        ax.set_xticks(range(3)); ax.set_xticklabels(NAMES, rotation=30)
        ax.set_yticks(range(3)); ax.set_yticklabels(NAMES)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    plt.tight_layout()
    plt.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--results", default="results.json")
    p.add_argument("--plot", default="confusion_matrices.png")
    args = p.parse_args()

    from datasets import load_dataset
    raw = load_dataset(DATASET)
    weights = balanced_class_weights(raw["train"]["label"])
    print("class weights:", weights.tolist())

    results, last_preds, y = {}, {}, None
    for label in args.models:
        runs = []
        for seed in args.seeds:
            r, y, preds = run(MODELS[label], seed, raw, weights, args.epochs)
            runs.append(r)
            print(label, "seed", seed, r)
        results[label] = runs
        last_preds[label] = preds

    summary = summarise(results)
    for label, s in summary.items():
        print(f"{label}: acc {s['acc_mean']:.3f} +/- {s['acc_std']:.3f}, "
              f"macro F1 {s['f1_mean']:.3f} +/- {s['f1_std']:.3f}")
    with open(args.results, "w") as f:
        json.dump(summary, f, indent=2)
    save_confusion_matrices(y, last_preds, args.plot)


if __name__ == "__main__":
    main()
