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
    """나이 관련 피처 — 인코딩 전 원본 컬럼에서 실행"""
    if "시술 당시 나이" in df.columns:
        df["나이_수치"] = df["시술 당시 나이"].map(AGE_MAP).fillna(-1)
        df["고령_여부"]    = (df["나이_수치"] >= 38).astype(int)
        df["초고령_여부"]  = (df["나이_수치"] >= 43).astype(int)
        df["최적나이_여부"] = (df["나이_수치"] < 35).astype(int)
    return df


def treatment_type_features(df):
    """
    특정 시술 유형 분해 — 핵심 피처!
    BLASTOCYST 포함 여부가 임신 성공률과 강한 상관관계
    """
    col = "특정 시술 유형"
    if col in df.columns:
        # 배반포(5일 배양) 여부 → 성공률 높음
        df["배반포_여부"] = df[col].astype(str).str.contains(
            "BLASTOCYST", na=False).astype(int)

        # 동결 배아 이식(FER) 여부
        df["FER_여부"] = df[col].astype(str).str.contains(
            "FER", na=False).astype(int)

        # ICSI(미세주입) 여부
        df["ICSI_여부"] = df[col].astype(str).str.contains(
            "ICSI", na=False).astype(int)

        # IVF 여부
        df["IVF_여부"] = df[col].astype(str).str.contains(
            "IVF", na=False).astype(int)

        # 복합 시술 여부 (/ 또는 : 포함)
        df["복합시술_여부"] = df[col].astype(str).str.contains(
            r"[/:]", na=False).astype(int)

    # 동결 배아 전용 이식
    if "동결 배아 사용 여부" in df.columns and "신선 배아 사용 여부" in df.columns:
        df["동결배아_전용"] = (
            (df["동결 배아 사용 여부"] == 1) &
            (df["신선 배아 사용 여부"] == 0)
        ).astype(int)

        df["신선배아_전용"] = (
            (df["동결 배아 사용 여부"] == 0) &
            (df["신선 배아 사용 여부"] == 1)
        ).astype(int)

    return df


def embryo_features(df):
    """배아 품질 및 활용 관련 피처"""

    # 배아 이식 효율
    df["배아_이식효율"] = df["이식된 배아 수"] / (df["총 생성 배아 수"] + 1)

    # 배아 저장 효율
    df["배아_저장효율"] = df["저장된 배아 수"] / (df["총 생성 배아 수"] + 1)

    # 미세주입 비율
    df["미세주입_비율"] = df["미세주입된 난자 수"] / (df["혼합된 난자 수"] + 1)

    # 미세주입 배아 이식 비율
    df["미세주입_이식비율"] = df["미세주입 배아 이식 수"] / (df["이식된 배아 수"] + 1)

    # 신선 난자 활용률
    df["신선난자_활용률"] = df["혼합된 난자 수"] / (df["수집된 신선 난자 수"] + 1)

    # 난자 수 구간 피처 (비선형성 반영)
    # 너무 적어도, 너무 많아도 성공률 낮아짐
    df["난자수_적정"] = (
        (df["수집된 신선 난자 수"] >= 5) &
        (df["수집된 신선 난자 수"] <= 15)
    ).astype(int)

    df["난자수_과다"] = (df["수집된 신선 난자 수"] > 15).astype(int)
    df["난자수_부족"] = (df["수집된 신선 난자 수"] < 5).astype(int)

    # 이식 배아 수 구간
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

    # 출산 성공 이력 여부 → 착상/임신 유지 가능하다는 증거
    df["출산이력_여부"] = (df["총 출산 횟수"] > 0).astype(int)

    # 임신 경험 여부
    df["임신이력_여부"] = (df["총 임신 횟수"] > 0).astype(int)

    # 클리닉 충성도
    df["클리닉_충성도"] = df["클리닉 내 총 시술 횟수"] / (df["총 시술 횟수"] + 1)

    # 시술 대비 임신 비율
    df["임신_시술_비율"] = df["총 임신 횟수"] / (df["총 시술 횟수"] + 1)

    return df


def infertility_cause_features(df):
    """불임 원인 조합 관련 피처"""

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

    # 자궁내막증 여부 → 착상 환경 저해
    if "불임 원인 - 자궁내막증" in df.columns:
        df["자궁내막증_여부"] = df["불임 원인 - 자궁내막증"]

    # 배란 장애 여부
    if "불임 원인 - 배란 장애" in df.columns:
        df["배란장애_여부"] = df["불임 원인 - 배란 장애"]

    return df


def run_feature_engineering(df):
    """전체 피처 엔지니어링 실행"""
    df = treatment_type_features(df)   # 특정 시술 유형 분해 (핵심!)
    df = embryo_features(df)
    df = treatment_history_features(df)
    df = infertility_cause_features(df)
    print(f"피처 엔지니어링 완료 → 총 {len(df.columns)}개 피처")
    return df

def feature_engineering(df):
    """기본 피처 엔지니어링"""

    # 불임 원인 총 개수
    infertility_cols = [c for c in df.columns if "불임 원인" in c]
    df["불임원인_총개수"] = df[infertility_cols].sum(axis=1)

    # 배아 활용률
    df["배아_활용률"] = df["이식된 배아 수"] / (df["총 생성 배아 수"] + 1)

    # 미세주입 이식 비율
    df["미세주입_이식비율"] = df["미세주입 배아 이식 수"] / (df["이식된 배아 수"] + 1)

    # 신선 난자 활용률
    df["신선난자_활용률"] = df["혼합된 난자 수"] / (df["수집된 신선 난자 수"] + 1)

    # 시술 대비 임신 비율
    df["임신_시술_비율"] = df["총 임신 횟수"] / (df["총 시술 횟수"] + 1)

    # ── 실험 1: BLASTOCYST 피처 ──────────────────
    df["배반포_여부"] = df["특정 시술 유형"].astype(str).str.contains(
        "BLASTOCYST", na=False).astype(int)
    df["FER_여부"] = df["특정 시술 유형"].astype(str).str.contains(
        "FER", na=False).astype(int)
    df["ICSI_여부"] = df["특정 시술 유형"].astype(str).str.contains(
        "ICSI", na=False).astype(int)

    return df