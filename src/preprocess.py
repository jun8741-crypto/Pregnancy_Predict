import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_FILE, TEST_FILE, SUB_FILE, OUTPUT_DIR,
    CAT_COLS, BIN_COLS, COUNT_COLS, COUNT_MAP, NUM_COLS,
    SEED, N_SPLITS,
)

# 결측 여부를 피처로 만들 컬럼
MISSING_INDICATOR_COLS = [
    "난자 해동 경과일",
    "배아 해동 경과일",
    "PGS 시술 여부",
    "PGD 시술 여부",
    "착상 전 유전 검사 사용 여부",
    "착상 전 유전 진단 사용 여부",
    "임신 시도 또는 마지막 임신 경과 연수",
    "난자 채취 경과일",
    "난자 혼합 경과일",
    "배아 이식 경과일",
]

# 나이 수치 매핑
AGE_MAP = {
    "만18-34세": 26,
    "만35-37세": 36,
    "만38-39세": 38,
    "만40-42세": 41,
    "만43-44세": 43,
    "만45-50세": 47,
    "알 수 없음": -1,
}

DONOR_AGE_MAP = {
    "만20세 이하": 19,
    "만21-25세":   23,
    "만26-30세":   28,
    "만31-35세":   33,
    "만36-40세":   38,
    "만41-45세":   43,
    "알 수 없음":  -1,
}


def load_data():
    train = pd.read_csv(TRAIN_FILE)
    test  = pd.read_csv(TEST_FILE)
    sub   = pd.read_csv(SUB_FILE)
    print(f"Train: {train.shape} | Test: {test.shape}")
    return train, test, sub


def add_missing_indicators(df):
    """결측 여부 자체를 피처로 — 결측치 채우기 전에 실행"""
    for col in MISSING_INDICATOR_COLS:
        if col in df.columns:
            df[f"{col}_결측"] = df[col].isna().astype(int)
    return df


def extract_age_features(df):
    """나이 수치화 — 인코딩 전 원본 컬럼에서 실행"""
    if "시술 당시 나이" in df.columns:
        df["나이_수치"] = df["시술 당시 나이"].map(AGE_MAP).fillna(-1)
        df["고령_여부"]     = (df["나이_수치"] >= 38).astype(int)
        df["초고령_여부"]   = (df["나이_수치"] >= 43).astype(int)
        df["최적나이_여부"] = (df["나이_수치"] < 35).astype(int)
    return df


def extract_treatment_type_features(df):
    """인코딩 전 실행 — 시술 유형 텍스트에서 피처 추출"""
    col = "특정 시술 유형"
    df["배반포_여부"]   = df[col].astype(str).str.contains("BLASTOCYST", na=False).astype(int)
    df["FER_여부"]      = df[col].astype(str).str.contains("FER", na=False).astype(int)
    df["ICSI_여부"]     = df[col].astype(str).str.contains("ICSI", na=False).astype(int)
    df["IVF_여부"]      = df[col].astype(str).str.contains("IVF", na=False).astype(int)
    df["복합시술_여부"] = df[col].astype(str).str.contains(r"[/:]", na=False).astype(int)
    return df


def add_donor_age_features(df):
    """기증자 나이 수치화 — 인코딩 전 원본 컬럼에서 실행"""
    if "난자 기증자 나이" in df.columns:
        df["난자기증자_나이_수치"] = df["난자 기증자 나이"].map(DONOR_AGE_MAP).fillna(-1)
    if "정자 기증자 나이" in df.columns:
        df["정자기증자_나이_수치"] = df["정자 기증자 나이"].map(DONOR_AGE_MAP).fillna(-1)
    # 환자 vs 난자 기증자 나이 차 (기증 난자 사용 케이스에서 유효)
    if "난자기증자_나이_수치" in df.columns and "나이_수치" in df.columns:
        valid = (df["난자기증자_나이_수치"] != -1) & (df["나이_수치"] != -1)
        df["환자_기증자_나이차"] = np.where(
            valid, df["나이_수치"] - df["난자기증자_나이_수치"], -1
        )
    return df


def add_clinic_target_encoding(train, test, target):
    """시술 시기 코드 OOF target encoding — 데이터 누수 없이 클리닉 성공률 반영"""
    col        = "시술 시기 코드"
    new_col    = "클리닉_성공률_TE"
    global_mean = float(np.mean(target))

    train = train.reset_index(drop=True)
    oof_te = np.full(len(train), global_mean)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for tr_idx, val_idx in skf.split(train, target):
        fold_df   = pd.DataFrame({"code": train[col].iloc[tr_idx].values, "target": target[tr_idx]})
        fold_mean = fold_df.groupby("code")["target"].mean()
        oof_te[val_idx] = train[col].iloc[val_idx].map(fold_mean).fillna(global_mean).values

    train[new_col] = oof_te

    clinic_mean    = pd.DataFrame({"code": train[col].values, "target": target}).groupby("code")["target"].mean()
    test[new_col]  = test[col].map(clinic_mean).fillna(global_mean)

    return train, test


def map_count_cols(df):
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].map(COUNT_MAP)
    return df


def fill_missing(train, test):
    for col in CAT_COLS:
        if col in train.columns:
            train[col] = train[col].fillna("unknown")
        if col in test.columns:
            test[col]  = test[col].fillna("unknown")

    for col in BIN_COLS:
        if col in train.columns:
            train[col] = train[col].fillna(train[col].mode()[0])
        if col in test.columns:
            test[col]  = test[col].fillna(test[col].mode()[0])

    for col in COUNT_COLS:
        if col in train.columns:
            train[col] = train[col].fillna(train[col].median())
        if col in test.columns:
            test[col]  = test[col].fillna(test[col].median())

    for col in NUM_COLS:
        if col in train.columns:
            train[col] = train[col].fillna(train[col].median())
        if col in test.columns:
            test[col]  = test[col].fillna(test[col].median())

    return train, test


def encode_cat_cols(train, test):
    all_data = pd.concat([train, test], axis=0, ignore_index=True)
    encoders = {}

    for col in CAT_COLS:
        if col not in all_data.columns:
            continue
        le = LabelEncoder()
        le.fit(all_data[col].astype(str))
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))
        encoders[col] = le

    return train, test, encoders


def feature_engineering(df):
    """피처 엔지니어링"""

    # 불임 원인 총 개수
    infertility_cols = [c for c in df.columns if "불임 원인" in c]
    df["불임원인_총개수"] = df[infertility_cols].sum(axis=1)

    # 배아 활용률
    df["배아_활용률"]       = df["이식된 배아 수"] / (df["총 생성 배아 수"] + 1)

    # 미세주입 이식 비율
    df["미세주입_이식비율"] = df["미세주입 배아 이식 수"] / (df["이식된 배아 수"] + 1)

    # 신선 난자 활용률
    df["신선난자_활용률"]   = df["혼합된 난자 수"] / (df["수집된 신선 난자 수"] + 1)

    # 시술 대비 임신 비율
    df["임신_시술_비율"]    = df["총 임신 횟수"] / (df["총 시술 횟수"] + 1)

    # 반복 실패 횟수
    df["실패_횟수"]         = df["총 시술 횟수"] - df["총 임신 횟수"]

    # 첫 시술 여부
    df["첫_시술_여부"]      = (df["총 시술 횟수"] == 0).astype(int)

    # 출산 이력 여부
    df["출산이력_여부"]     = (df["총 출산 횟수"] > 0).astype(int)

    # 나이 × 시술 횟수 교호작용
    if "나이_수치" in df.columns:
        df["나이x시술횟수"]   = df["나이_수치"] * df["총 시술 횟수"]
        df["나이x임신성공률"] = df["나이_수치"] * df["임신_시술_비율"]

    # ICSI 배아 생성 효율 (미세주입 성공률)
    df["ICSI_배아생성효율"] = df["미세주입에서 생성된 배아 수"] / (df["미세주입된 난자 수"] + 1)

    # 전체 수정 성공률 (난자 → 배아 전환율)
    df["수정성공률"] = df["총 생성 배아 수"] / (df["혼합된 난자 수"] + 1)

    # 동결 배아 해동 성공률
    df["해동성공률"] = df["해동된 배아 수"] / (df["저장된 배아 수"] + 1)

    # 나이 × 배반포 교호작용
    if "배반포_여부" in df.columns and "나이_수치" in df.columns:
        df["고령_배반포"]     = df["고령_여부"] * df["배반포_여부"]
        df["나이x배반포"]     = df["나이_수치"] * df["배반포_여부"]

    return df


def preprocess(save=True):
    """전체 전처리 파이프라인"""

    # 1. 로드
    train, test, sub = load_data()

    # 2. 타겟 / ID 분리
    target = train["임신 성공 여부"].values
    train  = train.drop(columns=["임신 성공 여부", "ID"])
    test   = test.drop(columns=["ID"])

    # 3. 결측 여부 피처 추출 (결측치 채우기 전에!)
    train = add_missing_indicators(train)
    test  = add_missing_indicators(test)

    # 4. 나이 수치화 (인코딩 전에!)
    train = extract_age_features(train)
    test  = extract_age_features(test)

    # 5. 인코딩 전 시술 유형 피처 추출
    train = extract_treatment_type_features(train)
    test  = extract_treatment_type_features(test)

    # 6. 기증자 나이 수치화 (인코딩 전)
    train = add_donor_age_features(train)
    test  = add_donor_age_features(test)

    # 7. 횟수형 매핑
    train = map_count_cols(train)
    test  = map_count_cols(test)

    # 8. 결측치 처리 (각각)
    train, test = fill_missing(train, test)

    # 9. 클리닉 OOF target encoding (label encoding 전)
    train, test = add_clinic_target_encoding(train, test, target)

    # 10. 범주형 인코딩 (합쳐서 fit)
    train, test, encoders = encode_cat_cols(train, test)

    # 11. 피처 엔지니어링
    train = feature_engineering(train)
    test  = feature_engineering(test)

    print(f"전처리 완료 → Train: {train.shape} | Test: {test.shape}")
    print(f"피처 수: {len(train.columns)}개")

    if save:
        train.to_csv(os.path.join(OUTPUT_DIR, "train_processed.csv"), index=False)
        test.to_csv(os.path.join(OUTPUT_DIR,  "test_processed.csv"),  index=False)
        print(f"저장 완료 → {OUTPUT_DIR}")

    return train, test, target, sub


if __name__ == "__main__":
    preprocess(save=True)