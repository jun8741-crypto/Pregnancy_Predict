import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, SEED, N_SPLITS
from preprocess import preprocess

# ── LightGBM 파라미터 (Optuna 튜닝 결과) ────────────────────
LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "auc",
    "learning_rate":     0.01769,
    "num_leaves":        35,
    "max_depth":         8,
    "min_child_samples": 88,
    "feature_fraction":  0.5753,
    "bagging_fraction":  0.6406,
    "bagging_freq":      2,
    "reg_alpha":         0.04619,
    "reg_lambda":        0.00066,
    "n_jobs":            -1,
    "verbose":           -1,
    "random_state":      SEED,
}

# ── XGBoost 파라미터 ─────────────────────────────────────────
XGB_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "learning_rate":    0.05,
    "max_depth":        6,
    "min_child_weight": 50,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "n_jobs":           -1,
    "verbosity":        0,
    "random_state":     SEED,
    "tree_method":      "hist",
}


def train_lgb_cv(X, y, X_test):
    """LightGBM 5-Fold CV"""
    skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx],      y[val_idx]

        model = lgb.train(
            LGB_PARAMS,
            lgb.Dataset(X_tr, label=y_tr),
            num_boost_round=3000,
            valid_sets=[lgb.Dataset(X_val, label=y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(False),
            ],
        )
        oof_preds[val_idx] = model.predict(X_val)
        test_preds        += model.predict(X_test) / N_SPLITS
        print(f"  LGB Fold {fold+1}: {roc_auc_score(y_val, oof_preds[val_idx]):.4f}")

    lgb_auc = roc_auc_score(y, oof_preds)
    print(f"  LGB OOF AUC: {lgb_auc:.4f}")
    return oof_preds, test_preds


def train_xgb_cv(X, y, X_test):
    """XGBoost 5-Fold CV"""
    skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx],      y[val_idx]

        dtrain = xgb.DMatrix(X_tr,  label=y_tr)
        dval   = xgb.DMatrix(X_val, label=y_val)
        dtest  = xgb.DMatrix(X_test)

        model = xgb.train(
            XGB_PARAMS,
            dtrain,
            num_boost_round=3000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        oof_preds[val_idx] = model.predict(dval)
        test_preds        += model.predict(dtest) / N_SPLITS
        print(f"  XGB Fold {fold+1}: {roc_auc_score(y_val, oof_preds[val_idx]):.4f}")

    xgb_auc = roc_auc_score(y, oof_preds)
    print(f"  XGB OOF AUC: {xgb_auc:.4f}")
    return oof_preds, test_preds


def main():
    print("=" * 50)
    print("  전처리 시작")
    print("=" * 50)
    train, test, target, sub = preprocess(save=False)

    print("\n" + "=" * 50)
    print("  LightGBM 학습")
    print("=" * 50)
    lgb_oof, lgb_test = train_lgb_cv(train, target, test)

    print("\n" + "=" * 50)
    print("  XGBoost 학습")
    print("=" * 50)
    xgb_oof, xgb_test = train_xgb_cv(train, target, test)

    # 앙상블 (LGB 60% + XGB 40%)
    ens_oof  = 0.6 * lgb_oof  + 0.4 * xgb_oof
    ens_test = 0.6 * lgb_test + 0.4 * xgb_test
    ens_auc  = roc_auc_score(target, ens_oof)

    print("\n" + "=" * 50)
    print(f"  LGB OOF : {roc_auc_score(target, lgb_oof):.4f}")
    print(f"  XGB OOF : {roc_auc_score(target, xgb_oof):.4f}")
    print(f"  앙상블   : {ens_auc:.4f}")
    print("=" * 50)

    score_str = f"{ens_auc:.4f}".replace(".", "_")
    filepath  = os.path.join(OUTPUT_DIR, f"submission_lgb_xgb_auc{score_str}.csv")
    sub["probability"] = ens_test
    sub.to_csv(filepath, index=False)
    print(f"저장 완료 → {filepath}")


if __name__ == "__main__":
    main()