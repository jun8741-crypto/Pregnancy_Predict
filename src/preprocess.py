import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import sys

# src 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_FILE, TEST_FILE, SUB_FILE, OUTPUT_DIR,
    CAT_COLS, BIN_COLS, COUNT_COLS, COUNT_MAP, NUM_COLS,
)


def load_data():
    """데이터 로드"""
    train = pd.read_csv(TRAIN_FILE)
    test  = pd.read_csv(TEST_FILE)
    sub   = pd.read_csv(SUB_FILE)
    print(f"Train: {train.shape} | Test: {test.shape}")
    return train, test, sub


def map_count_cols(df):
    """횟수형 컬럼 숫자 매핑"""
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].map(COUNT_MAP)
    return df


def fill_missing(train, test):
    """
    결측치 처리 — 대회 규칙 준수
    - 범주형: 'unknown' 으로 채우기
    - 이진/횟수형: 최빈값 (train은 train, test는 test)
    - 수치형: 중앙값 (train은 train, test는 test)
    """
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
    """
    범주형 인코딩 — 대회 규칙 준수
    train + test 합쳐서 fit → 각각 transform
    """
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
    """피처 엔지니어링 — train/test 각각 호출"""

    # 불임 원인 총 개수
    infertility_cols = [c for c in df.columns if "불임 원인" in c]
    df["불임원인_총개수"] = df[infertility_cols].sum(axis=1)

    # 배아 활용률 (이식 / 생성)
    df["배아_활용률"] = df["이식된 배아 수"] / (df["총 생성 배아 수"] + 1)

    # 미세주입 이식 비율
    df["미세주입_이식비율"] = df["미세주입 배아 이식 수"] / (df["이식된 배아 수"] + 1)

    # 신선 난자 활용률
    df["신선난자_활용률"] = df["혼합된 난자 수"] / (df["수집된 신선 난자 수"] + 1)

    # 시술 대비 임신 비율
    df["임신_시술_비율"] = df["총 임신 횟수"] / (df["총 시술 횟수"] + 1)

    return df


def preprocess(save=True):
    """전체 전처리 파이프라인"""

    # 1. 로드
    train, test, sub = load_data()

    # 2. 타겟 / ID 분리
    target = train["임신 성공 여부"].values
    train  = train.drop(columns=["임신 성공 여부", "ID"])
    test   = test.drop(columns=["ID"])

    # 3. 횟수형 매핑
    train = map_count_cols(train)
    test  = map_count_cols(test)

    # 4. 결측치 처리 (각각)
    train, test = fill_missing(train, test)

    # 5. 범주형 인코딩 (합쳐서 fit)
    train, test, encoders = encode_cat_cols(train, test)

    # 6. 피처 엔지니어링
    train = feature_engineering(train)
    test  = feature_engineering(test)

    print(f"전처리 완료 → Train: {train.shape} | Test: {test.shape}")
    print(f"피처 목록 ({len(train.columns)}개):")
    print(train.columns.tolist())

    # 7. 저장
    if save:
        train.to_csv(os.path.join(OUTPUT_DIR, "train_processed.csv"), index=False)
        test.to_csv(os.path.join(OUTPUT_DIR,  "test_processed.csv"),  index=False)
        print(f"저장 완료 → {OUTPUT_DIR}")

    return train, test, target, sub


if __name__ == "__main__":
    preprocess(save=True)