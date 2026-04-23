import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import os
import sys

# src 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, SEED, N_SPLITS
from preprocess import load_data, map_count_cols, fill_missing

# ── CatBoost용 범주형 컬럼 ───────────────────────────────────
CAT_FEATURES = [
    "시술 시기 코드",
    "시술 당시 나이",
    "시술 유형",
    "특정 시술 유형",
    "배란 유도 유형",
    "배아 생성 주요 이유",
    "난자 출처",
    "정자 출처",
    "난자 기증자 나이",
    "정자 기증자 나이",
]

# ── CatBoost 파라미터 ────────────────────────────────────────
CAT_PARAMS = {
    "iterations":        2000,
    "learning_rate":     0.05,
    "depth":             6,
    "loss_function":     "Logloss",
    "eval_metric":       "AUC",
    "random_seed":       SEED,
    "early_stopping_rounds": 50,
    "verbose":           100,
    "task_type":         "CPU",
}


def preprocess_for_catboost():
    """
    CatBoost용 전처리
    - 범주형 컬럼을 문자열로 유지 (CatBoost가 알아서 처리)
    - 라벨 인코딩 X
    """
    from config import COUNT_COLS, COUNT_MAP, BIN_COLS, NUM_COLS

    train, test, sub = load_data()

    # 타겟 / ID 분리
    target = train["임신 성공 여부"].values
    train  = train.drop(columns=["임신 성공 여부", "ID"])
    test   = test.drop(columns=["ID"])

    # 횟수형 매핑
    train = map_count_cols(train)
    test  = map_count_cols(test)

    # 결측치 처리
    train, test = fill_missing(train, test)

    # 범주형 컬럼 → 문자열 유지 (CatBoost는 인코딩 불필요)
    for col in CAT_FEATURES:
        if col in train.columns:
            train[col] = train[col].astype(str)
        if col in test.columns:
            test[col]  = test[col].astype(str)

    # 피처 엔지니어링
    for df in [train, test]:
        infertility_cols = [c for c in df.columns if "불임 원인" in c]
        df["불임원인_총개수"]   = df[infertility_cols].sum(axis=1)
        df["배아_활용률"]       = df["이식된 배아 수"] / (df["총 생성 배아 수"] + 1)
        df["미세주입_이식비율"] = df["미세주입 배아 이식 수"] / (df["이식된 배아 수"] + 1)
        df["신선난자_활용률"]   = df["혼합된 난자 수"] / (df["수집된 신선 난자 수"] + 1)
        df["임신_시술_비율"]    = df["총 임신 횟수"] / (df["총 시술 횟수"] + 1)
        df["실패_횟수"]         = df["총 시술 횟수"] - df["총 임신 횟수"]
        df["첫_시술_여부"]      = (df["총 시술 횟수"] == 0).astype(int)

    print(f"전처리 완료 → Train: {train.shape} | Test: {test.shape}")
    return train, test, target, sub


def train_cv(X, y, X_test, cat_features):
    """5-Fold Stratified CV 학습"""
    skf         = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds   = np.zeros(len(X))
    test_preds  = np.zeros(len(X_test))
    fold_scores = []

    # CatBoost용 범주형 인덱스
    cat_idx = [X.columns.get_loc(c) for c in cat_features if c in X.columns]

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr,  X_val = X.iloc[tr_idx],  X.iloc[val_idx]
        y_tr,  y_val = y[tr_idx],       y[val_idx]

        model = CatBoostClassifier(**CAT_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            cat_features=cat_idx,
        )

        val_pred           = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        test_preds        += model.predict_proba(X_test)[:, 1] / N_SPLITS

        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_auc)
        print(f"  Fold {fold+1} AUC: {fold_auc:.4f}  (best iter: {model.best_iteration_})")

    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\n{'='*45}")
    print(f"  OOF AUC : {oof_auc:.4f}")
    print(f"  Fold 평균: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"{'='*45}\n")

    return oof_preds, test_preds, oof_auc


def save_submission(sub, test_preds, oof_auc):
    """submission 저장"""
    score_str = f"{oof_auc:.4f}".replace(".", "_")
    filename  = f"submission_catboost_auc{score_str}.csv"
    filepath  = os.path.join(OUTPUT_DIR, filename)

    sub["probability"] = test_preds
    sub.to_csv(filepath, index=False)
    print(f"저장 완료 → {filepath}")
    return filepath


def main():
    print("=" * 45)
    print("  CatBoost 전처리 시작")
    print("=" * 45)
    train, test, target, sub = preprocess_for_catboost()

    # CatBoost용 범주형 컬럼 (실제 존재하는 것만)
    cat_features = [c for c in CAT_FEATURES if c in train.columns]

    print("\n" + "=" * 45)
    print("  학습 시작 (CatBoost 5-Fold CV)")
    print("=" * 45 + "\n")
    oof_preds, test_preds, oof_auc = train_cv(train, target, test, cat_features)

    save_submission(sub, test_preds, oof_auc)


if __name__ == "__main__":
    main()