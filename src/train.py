import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import os
import sys

# src 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess

# ── 경로 설정 ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LightGBM 파라미터 ────────────────────────────────────────
PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "learning_rate":    0.05,
    "num_leaves":       127,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "n_jobs":           -1,
    "verbose":          -1,
    "random_state":     42,
}

N_SPLITS    = 5
NUM_BOOST   = 2000
EARLY_STOP  = 50


def train_cv(X, y, X_test):
    """5-Fold Stratified CV 학습"""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_preds   = np.zeros(len(X))
    test_preds  = np.zeros(len(X_test))
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr,  X_val  = X.iloc[tr_idx],  X.iloc[val_idx]
        y_tr,  y_val  = y[tr_idx],       y[val_idx]

        dtrain = lgb.Dataset(X_tr,  label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(100),
            ],
        )

        val_pred            = model.predict(X_val)
        oof_preds[val_idx]  = val_pred
        test_preds         += model.predict(X_test) / N_SPLITS

        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_auc)
        print(f"  Fold {fold+1} AUC: {fold_auc:.4f}  (best iter: {model.best_iteration})")

    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\n{'='*45}")
    print(f"  OOF AUC : {oof_auc:.4f}")
    print(f"  Fold 평균: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"{'='*45}\n")

    return oof_preds, test_preds, oof_auc


def save_submission(sub, test_preds, oof_auc):
    """submission 파일 저장 — AUC 점수를 파일명에 포함"""
    score_str = f"{oof_auc:.4f}".replace(".", "_")
    filename  = f"submission_auc{score_str}.csv"
    filepath  = os.path.join(OUTPUT_DIR, filename)

    sub["probability"] = test_preds
    sub.to_csv(filepath, index=False)
    print(f"저장 완료 → {filepath}")
    return filepath


def main():
    print("=" * 45)
    print("  전처리 시작")
    print("=" * 45)
    train, test, target, sub = preprocess(save=False)

    X      = train
    y      = target
    X_test = test

    print("\n" + "=" * 45)
    print("  학습 시작 (LightGBM 5-Fold CV)")
    print("=" * 45 + "\n")
    oof_preds, test_preds, oof_auc = train_cv(X, y, X_test)

    save_submission(sub, test_preds, oof_auc)


if __name__ == "__main__":
    main()