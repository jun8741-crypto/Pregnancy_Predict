import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# ── 경로 설정 ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 범주형 컬럼 목록 ────────────────────────────────────────
CAT_COLS = [
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

# 0/1 이진 컬럼 (결측치만 채우면 됨, 인코딩 불필요)
BIN_COLS = [
    "배란 자극 여부",
    "단일 배아 이식 여부",
    "착상 전 유전 검사 사용 여부",
    "착상 전 유전 진단 사용 여부",
    "남성 주 불임 원인", "남성 부 불임 원인",
    "여성 주 불임 원인", "여성 부 불임 원인",
    "부부 주 불임 원인", "부부 부 불임 원인",
    "불명확 불임 원인",
    "불임 원인 - 난관 질환", "불임 원인 - 남성 요인",
    "불임 원인 - 배란 장애", "불임 원인 - 여성 요인",
    "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도", "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태",
    "동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부",
    "대리모 여부", "PGD 시술 여부", "PGS 시술 여부",
]

# 횟수형 컬럼 (순서형 → 숫자 매핑)
COUNT_COLS = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
    "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수",
]
COUNT_MAP = {
    "0회": 0, "1회": 1, "2회": 2,
    "3회": 3, "4회": 4, "5회": 5, "6회 이상": 6,
}

# 수치형 컬럼
NUM_COLS = [
    "임신 시도 또는 마지막 임신 경과 연수",
    "총 생성 배아 수", "미세주입된 난자 수",
    "미세주입에서 생성된 배아 수", "이식된 배아 수",
    "미세주입 배아 이식 수", "저장된 배아 수",
    "미세주입 후 저장된 배아 수", "해동된 배아 수",
    "해동 난자 수", "수집된 신선 난자 수",
    "저장된 신선 난자 수", "혼합된 난자 수",
    "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수",
    "난자 채취 경과일", "난자 해동 경과일",
    "난자 혼합 경과일", "배아 이식 경과일", "배아 해동 경과일",
]


def load_data():
    """데이터 로드"""
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sub   = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
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
    결측치 처리
    - 이진/횟수/수치형: train은 train 통계, test는 test 통계 (규칙 준수)
    - 범주형: 'unknown' 으로 채우기
    """
    # 범주형 결측 → 'unknown'
    for col in CAT_COLS:
        if col in train.columns:
            train[col] = train[col].fillna("unknown")
        if col in test.columns:
            test[col]  = test[col].fillna("unknown")

    # 이진형 결측 → 최빈값
    for col in BIN_COLS:
        if col in train.columns:
            train[col] = train[col].fillna(train[col].mode()[0])
        if col in test.columns:
            test[col]  = test[col].fillna(test[col].mode()[0])

    # 횟수형 결측 → 중앙값
    for col in COUNT_COLS:
        if col in train.columns:
            train[col] = train[col].fillna(train[col].median())
        if col in test.columns:
            test[col]  = test[col].fillna(test[col].median())

    # 수치형 결측 → 중앙값
    for col in NUM_COLS:
        if col in train.columns:
            train[col] = train[col].fillna(train[col].median())
        if col in test.columns:
            test[col]  = test[col].fillna(test[col].median())

    return train, test


def encode_cat_cols(train, test):
    """
    범주형 인코딩
    - train + test 합쳐서 fit → 규칙 준수
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
        encoders[col] = le  # 나중에 역변환 필요시 사용

    return train, test, encoders


def feature_engineering(df):
    """
    피처 엔지니어링
    - train/test 각각 호출해서 사용
    """
    # 불임 원인 총 개수
    infertility_cols = [c for c in df.columns if "불임 원인" in c]
    df["불임원인_총개수"] = df[infertility_cols].sum(axis=1)

    # 배아 활용률 (이식 / 생성)
    df["배아_활용률"] = df["이식된 배아 수"] / (df["총 생성 배아 수"] + 1)

    # 미세주입 비율 (미세주입 배아 / 이식 배아)
    df["미세주입_이식비율"] = df["미세주입 배아 이식 수"] / (df["이식된 배아 수"] + 1)

    # 신선 난자 활용률
    df["신선난자_활용률"] = df["혼합된 난자 수"] / (df["수집된 신선 난자 수"] + 1)

    # 시술 대비 임신 성공률 (누적 이력)
    df["임신_시술_비율"] = df["총 임신 횟수"] / (df["총 시술 횟수"] + 1)

    return df


def preprocess(save=True):
    """전체 전처리 파이프라인 실행"""
    # 1. 로드
    train, test, sub = load_data()

    # 2. 타겟 분리
    target = train["임신 성공 여부"].values
    train_ids = train["ID"]
    test_ids  = test["ID"]
    train = train.drop(columns=["임신 성공 여부", "ID"])
    test  = test.drop(columns=["ID"])

    # 3. 횟수형 매핑
    train = map_count_cols(train)
    test  = map_count_cols(test)

    # 4. 결측치 처리 (각각)
    train, test = fill_missing(train, test)

    # 5. 범주형 인코딩 (합쳐서)
    train, test, encoders = encode_cat_cols(train, test)

    # 6. 피처 엔지니어링
    train = feature_engineering(train)
    test  = feature_engineering(test)

    print(f"전처리 완료 → Train: {train.shape} | Test: {test.shape}")
    print(f"피처 목록 ({len(train.columns)}개):")
    print(train.columns.tolist())

    # 7. 저장 (선택)
    if save:
        train.to_csv(os.path.join(OUTPUT_DIR, "train_processed.csv"), index=False)
        test.to_csv(os.path.join(OUTPUT_DIR, "test_processed.csv"), index=False)
        print(f"저장 완료 → {OUTPUT_DIR}")

    return train, test, target, sub


if __name__ == "__main__":
    train, test, target, sub = preprocess(save=True)