import pandas as pd


def confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Calcula TP, FP, TN, FN a partir de y_true e y_pred (booleano ou 0/1).

    Retorna:
        dict com: tp, fp, tn, fn
    """
    yt = pd.Series(y_true).astype(bool)
    yp = pd.Series(y_pred).astype(bool)

    tp = int(((yt == True) & (yp == True)).sum())
    fp = int(((yt == False) & (yp == True)).sum())
    tn = int(((yt == False) & (yp == False)).sum())
    fn = int(((yt == True) & (yp == False)).sum())

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision(tp: int, fp: int) -> float:
    denom = tp + fp
    if denom == 0:
        return 0.0
    return tp / denom


def recall(tp: int, fn: int) -> float:
    denom = tp + fn
    if denom == 0:
        return 0.0
    return tp / denom


def f1_score(p: float, r: float) -> float:
    denom = p + r
    if denom == 0:
        return 0.0
    return 2 * p * r / denom


def false_positive_rate(fp: int, tn: int) -> float:
    denom = fp + tn
    if denom == 0:
        return 0.0
    return fp / denom