import pandas as pd
import numpy as np

# ── 나이 구간 → 수치 매핑 ────────────────────────────────────
AGE_MAP = {
    "만18-34세": 26,
    "만35-37세": 36,
    "만38-39세": 38,
    "만40-42세": 41,
    "만43-44세": 43,
    "만45-50세": 47,
    "알 수 없음": -1,
}


def age_features(df):
    """나이 관련 피처"""
    # 나이 수치화 (인코딩 전 원본 컬럼 필요 → preprocess 전에 호출 or 따로 저장)
    if "시술 당시 나이" in df.columns:
        df["나이_수치"] = df["시술 당시 나이"].map(AGE_MAP)
        df["나이_수치"] = df["나이_수치"].fillna(-1)

        # 고령 여부 (38세 이상 → 성공률 급감)
        df["고령_여부"] = (df["나이_수치"] >= 38).astype(int)

        # 초고령 여부 (43세 이상)
        df["초고령_여부"] = (df["나이_수치"] >= 43).astype(int)

        # 최적 나이 여부 (35세 미만)
        df["최적나이_여부"] = (df["나이_수치"] < 35).astype(int)

    return df


def embryo_features(df):
    """배아 품질 및 활용 관련 피처"""

    # 배아 이식 효율 (이식 / 생성)
    df["배아_이식효율"] = df["이식된 배아 수"] / (df["총 생성 배아 수"] + 1)

    # 배아 저장 효율 (저장 / 생성) → 남은 배아 많으면 품질 좋은 편
    df["배아_저장효율"] = df["저장된 배아 수"] / (df["총 생성 배아 수"] + 1)

    # 미세주입 비율 (ICSI 사용 정도)
    df["미세주입_비율"] = df["미세주입된 난자 수"] / (df["혼합된 난자 수"] + 1)

    # 미세주입 배아 이식 비율
    df["미세주입_이식비율"] = df["미세주입 배아 이식 수"] / (df["이식된 배아 수"] + 1)

    # 신선 난자 활용률
    df["신선난자_활용률"] = df["혼합된 난자 수"] / (df["수집된 신선 난자 수"] + 1)

    # 총 배아 대비 해동 배아 비율
    df["해동배아_비율"] = df["해동된 배아 수"] / (df["총 생성 배아 수"] + 1)

    # 이식 배아 수 구간 (1개 vs 2개 vs 3개 이상)
    df["이식배아_1개"] = (df["이식된 배아 수"] == 1).astype(int)
    df["이식배아_2개"] = (df["이식된 배아 수"] == 2).astype(int)
    df["이식배아_3개이상"] = (df["이식된 배아 수"] >= 3).astype(int)

    return df


def treatment_history_features(df):
    """시술 이력 관련 피처"""

    # 반복 실패 횟수
    df["실패_횟수"] = df["총 시술 횟수"] - df["총 임신 횟수"]

    # 과거 임신 성공률
    df["과거_임신성공률"] = df["총 임신 횟수"] / (df["총 시술 횟수"] + 1)

    # 첫 시술 여부
    df["첫_시술_여부"] = (df["총 시술 횟수"] == 0).astype(int)

    # 클리닉 충성도 (클리닉 내 횟수 / 전체 횟수)
    df["클리닉_충성도"] = df["클리닉 내 총 시술 횟수"] / (df["총 시술 횟수"] + 1)

    # IVF 집중도 (IVF 횟수 / 전체 횟수)
    df["IVF_집중도"] = df["IVF 시술 횟수"] / (df["총 시술 횟수"] + 1)

    # 출산 성공 이력 여부
    df["출산이력_여부"] = (df["총 출산 횟수"] > 0).astype(int)

    # 임신 경험 여부
    df["임신이력_여부"] = (df["총 임신 횟수"] > 0).astype(int)

    return df


def infertility_cause_features(df):
    """불임 원인 조합 관련 피처"""

    # 불임 원인 총 개수
    cause_cols = [c for c in df.columns if "불임 원인 -" in c]
    df["불임원인_총개수"] = df[cause_cols].sum(axis=1)

    # 남성 단독 불임
    df["불임_남성단독"] = (
        (df["남성 주 불임 원인"] == 1) &
        (df["여성 주 불임 원인"] == 0)
    ).astype(int)

    # 여성 단독 불임
    df["불임_여성단독"] = (
        (df["남성 주 불임 원인"] == 0) &
        (df["여성 주 불임 원인"] == 1)
    ).astype(int)

    # 부부 복합 불임
    df["불임_복합"] = (
        (df["남성 주 불임 원인"] == 1) &
        (df["여성 주 불임 원인"] == 1)
    ).astype(int)

    # 원인 불명
    df["불임_원인불명"] = (
        (df["불명확 불임 원인"] == 1) &
        (df["불임원인_총개수"] == 0)
    ).astype(int)

    # 정자 관련 문제 총합
    sperm_cols = [c for c in df.columns if "정자" in c and "불임 원인" in c]
    df["정자문제_총개수"] = df[sperm_cols].sum(axis=1)

    return df


def treatment_type_features(df):
    """시술 유형 관련 피처"""

    # 동결 배아 사용 여부 (최근 연구에서 신선보다 성공률 높은 경우 있음)
    df["동결배아_전용"] = (
        (df["동결 배아 사용 여부"] == 1) &
        (df["신선 배아 사용 여부"] == 0)
    ).astype(int)

    # 신선 배아 전용
    df["신선배아_전용"] = (
        (df["동결 배아 사용 여부"] == 0) &
        (df["신선 배아 사용 여부"] == 1)
    ).astype(int)

    # PGD 또는 PGS 시술 여부
    df["유전검사_시술"] = (
        (df["PGD 시술 여부"] == 1) |
        (df["PGS 시술 여부"] == 1)
    ).astype(int)

    return df


def interaction_features(df):
    """교호작용 피처 (나이 × 시술 관련)"""

    if "나이_수치" in df.columns:
        # 나이 × 이식 배아 수
        df["나이x이식배아"] = df["나이_수치"] * df["이식된 배아 수"]

        # 나이 × 과거 임신 성공률
        df["나이x임신성공률"] = df["나이_수치"] * df["과거_임신성공률"]

        # 나이 × 총 시술 횟수
        df["나이x시술횟수"] = df["나이_수치"] * df["총 시술 횟수"]

    return df


def run_feature_engineering(df):
    """전체 피처 엔지니어링 실행"""
    df = age_features(df)
    df = embryo_features(df)
    df = treatment_history_features(df)
    df = infertility_cause_features(df)
    df = treatment_type_features(df)
    # df = interaction_features(df)  ← 일단 주석 처리

    print(f"피처 엔지니어링 완료 → 총 {len(df.columns)}개 피처")
    return df