import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, SEED, N_SPLITS
from preprocess import preprocess

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

XGB_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "nthread":          -1,
    "verbosity":        0,
    "seed":             SEED,
}

CAT_PARAMS = {
    "iterations":            2000,
    "learning_rate":         0.05,
    "depth":                 6,
    "loss_function":         "Logloss",
    "eval_metric":           "AUC",
    "random_seed":           SEED,
    "early_stopping_rounds": 50,
    "verbose":               False,
    "task_type":             "CPU",
}


def main():
    print("=" * 55)
    print("  전처리 시작")
    print("=" * 55)
    train, test, target, sub = preprocess(save=False)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_lgb  = np.zeros(len(train))
    oof_xgb  = np.zeros(len(train))
    oof_cat  = np.zeros(len(train))
    test_lgb = np.zeros(len(test))
    test_xgb = np.zeros(len(test))
    test_cat = np.zeros(len(test))

    print("\n" + "=" * 55)
    print("  Level-1 학습 (LGB / XGB / CatBoost)")
    print("=" * 55)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train, target)):
        X_tr, X_val = train.iloc[tr_idx], train.iloc[val_idx]
        y_tr, y_val = target[tr_idx],      target[val_idx]
        print(f"\n── Fold {fold + 1}/{N_SPLITS} ──")

        # LightGBM
        lgb_model = lgb.train(
            LGB_PARAMS,
            lgb.Dataset(X_tr, label=y_tr),
            num_boost_round=3000,
            valid_sets=[lgb.Dataset(X_val, label=y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(False),
            ],
        )
        oof_lgb[val_idx]  = lgb_model.predict(X_val)
        test_lgb          += lgb_model.predict(test) / N_SPLITS
        print(f"  LGB : {roc_auc_score(y_val, oof_lgb[val_idx]):.4f}")

        # XGBoost
        dtrain    = xgb.DMatrix(X_tr,   label=y_tr)
        dval      = xgb.DMatrix(X_val,  label=y_val)
        dtest     = xgb.DMatrix(test)
        xgb_model = xgb.train(
            XGB_PARAMS, dtrain,
            num_boost_round=1000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        oof_xgb[val_idx]  = xgb_model.predict(dval)
        test_xgb          += xgb_model.predict(dtest) / N_SPLITS
        print(f"  XGB : {roc_auc_score(y_val, oof_xgb[val_idx]):.4f}")

        # CatBoost
        cat_model = CatBoostClassifier(**CAT_PARAMS)
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        oof_cat[val_idx]  = cat_model.predict_proba(X_val)[:, 1]
        test_cat          += cat_model.predict_proba(test)[:, 1] / N_SPLITS
        print(f"  CAT : {roc_auc_score(y_val, oof_cat[val_idx]):.4f}")

    print(f"\n{'=' * 55}")
    print(f"  L1 OOF AUC  LGB : {roc_auc_score(target, oof_lgb):.4f}")
    print(f"              XGB : {roc_auc_score(target, oof_xgb):.4f}")
    print(f"              CAT : {roc_auc_score(target, oof_cat):.4f}")

    # ── Level-2: LogisticRegression 메타러너 ──────────────────
    print(f"\n{'=' * 55}")
    print("  Level-2 메타러너 (LogisticRegression)")
    print(f"{'=' * 55}")

    meta_train = np.column_stack([oof_lgb, oof_xgb, oof_cat])
    meta_test  = np.column_stack([test_lgb, test_xgb, test_cat])

    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(meta_train, target)

    weights = dict(zip(["LGB", "XGB", "CAT"], meta.coef_[0]))
    print(f"  메타 가중치 : {weights}")

    oof_stack   = meta.predict_proba(meta_train)[:, 1]
    overall_auc = roc_auc_score(target, oof_stack)
    print(f"\n  Stack OOF AUC : {overall_auc:.4f}")
    print(f"{'=' * 55}")

    test_preds = meta.predict_proba(meta_test)[:, 1]
    score_str  = f"{overall_auc:.4f}".replace(".", "_")
    filepath   = os.path.join(OUTPUT_DIR, f"submission_stack_auc{score_str}.csv")
    sub["probability"] = test_preds
    sub.to_csv(filepath, index=False)
    print(f"저장 완료 → {filepath}")


if __name__ == "__main__":
    main()
