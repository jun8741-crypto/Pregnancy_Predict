import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
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

# ── CatBoost 파라미터 ────────────────────────────────────────
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

# ── IVF / DI 전용 피처 ──────────────────────────────────────
IVF_DROP_COLS = ["DI 시술 횟수", "DI 임신 횟수", "DI 출산 횟수"]
DI_DROP_COLS  = [
    "IVF 시술 횟수", "IVF 임신 횟수", "IVF 출산 횟수",
    "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수",
    "이식된 배아 수", "미세주입 배아 이식 수", "저장된 배아 수",
    "미세주입 후 저장된 배아 수", "해동된 배아 수", "해동 난자 수",
    "수집된 신선 난자 수", "저장된 신선 난자 수",
    "배아_활용률", "미세주입_이식비율", "신선난자_활용률",
]


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
        print(f"    LGB Fold {fold+1}: {roc_auc_score(y_val, oof_preds[val_idx]):.4f}")

    return oof_preds, test_preds


def train_cat_cv(X, y, X_test):
    """CatBoost 5-Fold CV"""
    skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx],      y[val_idx]

        model = CatBoostClassifier(**CAT_PARAMS)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds        += model.predict_proba(X_test)[:, 1] / N_SPLITS
        print(f"    CAT Fold {fold+1}: {roc_auc_score(y_val, oof_preds[val_idx]):.4f}")

    return oof_preds, test_preds


def train_on_subset(X, y, X_test, label):
    """IVF 또는 DI 서브셋에 대해 LGB + CAT 앙상블"""
    print(f"\n  [{label}] LightGBM 학습 중...")
    lgb_oof, lgb_test = train_lgb_cv(X, y, X_test)
    lgb_auc = roc_auc_score(y, lgb_oof)
    print(f"  [{label}] LGB OOF AUC: {lgb_auc:.4f}")

    print(f"\n  [{label}] CatBoost 학습 중...")
    cat_oof, cat_test = train_cat_cv(X, y, X_test)
    cat_auc = roc_auc_score(y, cat_oof)
    print(f"  [{label}] CAT OOF AUC: {cat_auc:.4f}")

    # 앙상블 (LGB 60% + CAT 40%)
    ens_oof  = 0.6 * lgb_oof  + 0.4 * cat_oof
    ens_test = 0.6 * lgb_test + 0.4 * cat_test
    ens_auc  = roc_auc_score(y, ens_oof)
    print(f"  [{label}] 앙상블 OOF AUC: {ens_auc:.4f}")

    return ens_oof, ens_test


def main():
    print("=" * 50)
    print("  전처리 시작")
    print("=" * 50)
    train, test, target, sub = preprocess(save=False)

    # 시술 유형 컬럼 찾기 (인코딩되어 있으므로 값으로 구분)
    # 원본에서 IVF=1, DI=0 또는 반대일 수 있으니 확인
    print(f"\n시술 유형 분포:\n{train['시술 유형'].value_counts()}")

    # IVF / DI 인덱스 분리
    ivf_train_idx = train[train["시술 유형"] == train["시술 유형"].max()].index
    di_train_idx  = train[train["시술 유형"] == train["시술 유형"].min()].index
    ivf_test_idx  = test[test["시술 유형"] == test["시술 유형"].max()].index
    di_test_idx   = test[test["시술 유형"] == test["시술 유형"].min()].index

    print(f"Train IVF: {len(ivf_train_idx)} | DI: {len(di_train_idx)}")
    print(f"Test  IVF: {len(ivf_test_idx)}  | DI: {len(di_test_idx)}")

    # IVF 서브셋
    X_ivf_train = train.loc[ivf_train_idx].drop(columns=[c for c in IVF_DROP_COLS if c in train.columns])
    y_ivf       = target[ivf_train_idx]
    X_ivf_test  = test.loc[ivf_test_idx].drop(columns=[c for c in IVF_DROP_COLS if c in test.columns])

    # DI 서브셋
    X_di_train  = train.loc[di_train_idx].drop(columns=[c for c in DI_DROP_COLS if c in train.columns])
    y_di        = target[di_train_idx]
    X_di_test   = test.loc[di_test_idx].drop(columns=[c for c in DI_DROP_COLS if c in test.columns])

    print("\n" + "=" * 50)
    print("  IVF 모델 학습")
    print("=" * 50)
    ivf_oof, ivf_test_preds = train_on_subset(X_ivf_train, y_ivf, X_ivf_test, "IVF")

    print("\n" + "=" * 50)
    print("  DI 모델 학습")
    print("=" * 50)
    di_oof, di_test_preds = train_on_subset(X_di_train, y_di, X_di_test, "DI")

    # 전체 OOF 합치기
    oof_preds = np.zeros(len(train))
    oof_preds[ivf_train_idx] = ivf_oof
    oof_preds[di_train_idx]  = di_oof
    overall_auc = roc_auc_score(target, oof_preds)

    print("\n" + "=" * 50)
    print(f"  전체 OOF AUC : {overall_auc:.4f}")
    print("=" * 50)

    # submission 합치기
    test_preds = np.zeros(len(test))
    test_preds[ivf_test_idx] = ivf_test_preds
    test_preds[di_test_idx]  = di_test_preds

    score_str = f"{overall_auc:.4f}".replace(".", "_")
    filepath  = os.path.join(OUTPUT_DIR, f"submission_ensemble_auc{score_str}.csv")
    sub["probability"] = test_preds
    sub.to_csv(filepath, index=False)
    print(f"저장 완료 → {filepath}")


if __name__ == "__main__":
    main()