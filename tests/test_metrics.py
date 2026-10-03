import pandas as pd

from src.metrics import (
    confusion_counts,
    precision,
    recall,
    f1_score,
    false_positive_rate,
)


def test_confusion_counts_basic_case():
    y_true = pd.Series([1, 1, 0, 0])
    y_pred = pd.Series([1, 0, 1, 0])

    counts = confusion_counts(y_true, y_pred)
    assert counts["tp"] == 1
    assert counts["fn"] == 1
    assert counts["fp"] == 1
    assert counts["tn"] == 1


def test_precision_recall_f1_fpr_values():
    tp, fp, tn, fn = 5, 3, 10, 2

    p = precision(tp, fp)            # 5/(5+3) = 0.625
    r = recall(tp, fn)               # 5/(5+2) = 0.714285...
    f1 = f1_score(p, r)
    fpr = false_positive_rate(fp, tn)  # 3/(3+10) = 0.230769...

    assert abs(p - 0.625) < 1e-9
    assert abs(r - (5 / 7)) < 1e-9
    assert abs(fpr - (3 / 13)) < 1e-9
    assert 0.0 <= f1 <= 1.0


def test_zero_divisions_return_zero():
    assert precision(0, 0) == 0.0
    assert recall(0, 0) == 0.0
    assert f1_score(0.0, 0.0) == 0.0
    assert false_positive_rate(0, 0) == 0.0