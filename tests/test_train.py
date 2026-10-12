import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_class_weights_match_readme_for_training_distribution():
    # 165 negative, 779 neutral, 320 positive = 1,264 sentences (see README "Data").
    labels = [0] * 165 + [1] * 779 + [2] * 320
    w = train.balanced_class_weights(labels)
    assert w.shape == (3,)
    assert [round(x, 2) for x in w.tolist()] == [2.55, 0.54, 1.32]


def test_weighted_loss_equals_plain_loss_with_equal_weights():
    logits = torch.tensor([[2.0, 0.1, -1.0], [0.3, 0.2, 1.5]])
    labels = torch.tensor([0, 2])
    got = train.weighted_cross_entropy(logits, labels, torch.ones(3))
    want = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.isclose(got, want)


def test_weighted_loss_upweights_the_rare_class():
    # Sample 0 (class 0) is wrong, sample 1 (class 1) is right. Upweighting class 0
    # makes the wrong sample count for more, so the batch loss goes up.
    logits = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    labels = torch.tensor([0, 1])
    weighted = train.weighted_cross_entropy(logits, labels, torch.tensor([5.0, 1.0, 1.0]))
    unweighted = train.weighted_cross_entropy(logits, labels, torch.ones(3))
    assert weighted > unweighted


def test_score_perfect_and_imperfect():
    assert train.score([0, 1, 2], [0, 1, 2]) == {"acc": 1.0, "f1": 1.0}
    s = train.score([0, 1, 2, 1], [0, 1, 1, 1])
    assert s["acc"] == 0.75
    assert 0 < s["f1"] < 1


def test_summarise_mean_and_std():
    runs = {"M": [{"acc": 0.9, "f1": 0.8}, {"acc": 0.95, "f1": 0.9}]}
    s = train.summarise(runs)["M"]
    assert s["acc_mean"] == pytest.approx(0.925)
    assert s["acc_std"] == pytest.approx(np.std([0.9, 0.95]))
    assert s["f1_mean"] == pytest.approx(0.85)
    assert s["runs"] == runs["M"]


def test_committed_results_match_readme_numbers():
    with open(os.path.join(ROOT, "results.json")) as f:
        res = json.load(f)
    assert set(res) == set(train.MODELS)
    assert round(res["FinBERT"]["acc_mean"], 3) == 0.967
    assert round(res["DistilBERT"]["acc_mean"], 3) == 0.956
    for model in res.values():
        assert len(model["runs"]) == len(train.SEEDS)
        assert summarise_matches(model)


def summarise_matches(model):
    again = train.summarise({"x": model["runs"]})["x"]
    return abs(again["acc_mean"] - model["acc_mean"]) < 1e-9 and abs(again["f1_mean"] - model["f1_mean"]) < 1e-9
