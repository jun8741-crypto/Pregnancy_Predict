from __future__ import annotations
import argparse
import os
import random
import time
import warnings
from pathlib import Path
from typing import Callable, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
SEED: int = 42
SEED_LIST: List[int] = [42, 2025, 7]
N_SPLITS: int = 5
ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT / 'data'
SUBMISSION_DIR: Path = ROOT / 'submission'
OUTPUTS_DIR: Path = ROOT / 'outputs'
OOF_DIR: Path = OUTPUTS_DIR / 'oof'
PREDS_DIR: Path = OUTPUTS_DIR / 'preds'
MODELS_DIR: Path = ROOT / 'models'
LOGS_DIR: Path = ROOT / 'logs'
TRAIN_CSV: Path = DATA_DIR / 'train.csv'
TEST_CSV: Path = DATA_DIR / 'test.csv'
SAMPLE_SUBMISSION_CSV: Path = SUBMISSION_DIR / 'sample_submission.csv'
ID_COL: str = 'ID'
TARGET: str = '임신 성공 여부'
for _d in (OUTPUTS_DIR, OOF_DIR, PREDS_DIR, MODELS_DIR, LOGS_DIR, SUBMISSION_DIR):
    _d.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int=SEED) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def auc_score(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))
COUNT_COLUMNS: List[str] = ['총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수', '클리닉 내 총 시술 횟수']

def to_int_count(s: pd.Series) -> pd.Series:
    if s.dtype != object:
        return s.astype('float')
    cleaned = s.astype(str).str.replace('회', '', regex=False).str.replace('이상', '', regex=False).str.strip().replace({'nan': np.nan, '': np.nan, 'None': np.nan})
    return pd.to_numeric(cleaned, errors='coerce')

def convert_count_columns(df: pd.DataFrame, cols: List[str]=COUNT_COLUMNS) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = to_int_count(df[c])
    return df
COMPRESS_DI_NUMERIC: List[str] = ['혼합된 난자 수', '단일 배아 이식 여부', '신선 배아 사용 여부', '이식된 배아 수', '저장된 배아 수', '저장된 신선 난자 수', '착상 전 유전 진단 사용 여부', '총 생성 배아 수', '미세주입에서 생성된 배아 수', '미세주입된 난자 수', '미세주입 후 저장된 배아 수', '수집된 신선 난자 수', '동결 배아 사용 여부', '대리모 여부', '미세주입 배아 이식 수', '기증 배아 사용 여부', '기증자 정자와 혼합된 난자 수', '해동된 배아 수', '파트너 정자와 혼합된 난자 수', '해동 난자 수', '난자 해동 경과일', '배아 해동 경과일', '난자 채취 경과일', 'PGS 시술 여부', 'PGD 시술 여부', '착상 전 유전 검사 사용 여부']
COMPRESS_DI_CATEGORICAL: List[str] = ['배아 생성 주요 이유']
DROP_LOW_INFO: List[str] = ['임신 시도 또는 마지막 임신 경과 연수']
DROP_NO_VARIANCE: List[str] = ['불임 원인 - 여성 요인']
DROP_MULTICOLLINEAR: List[str] = ['미세주입에서 생성된 배아 수']

def apply_missing_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in COMPRESS_DI_NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    for c in COMPRESS_DI_CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].fillna('없음')
    if '배아 이식 경과일' in df.columns:
        df['배아 이식 경과일'] = pd.to_numeric(df['배아 이식 경과일'], errors='coerce').fillna(0.0)
    if '난자 혼합 경과일' in df.columns:
        df['난자 혼합 경과일'] = pd.to_numeric(df['난자 혼합 경과일'], errors='coerce').fillna(0.0)
    drop_cols = DROP_LOW_INFO + DROP_NO_VARIANCE
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df

def _detect_object_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if df[c].dtype == object or pd.api.types.is_categorical_dtype(df[c])]

def encode_categorical(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    cat_cols = _detect_object_columns(train_df)
    for c in cat_cols:
        train_series = train_df[c].fillna('__missing__').astype(str)
        test_series = test_df[c].fillna('__missing__').astype(str) if c in test_df.columns else None
        categories = sorted(train_series.unique().tolist())
        cat_to_code = {cat: i for i, cat in enumerate(categories)}
        train_df[c] = train_series.map(cat_to_code).astype('int32')
        if test_series is not None:
            unknown_code = len(categories)
            test_df[c] = test_series.map(cat_to_code).fillna(unknown_code).astype('int32')
    return (train_df, test_df, cat_cols)

def preprocess_basic(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    train_df = convert_count_columns(train_df)
    test_df = convert_count_columns(test_df)
    train_df = apply_missing_strategy(train_df)
    test_df = apply_missing_strategy(test_df)
    y_train = train_df[TARGET].astype(float)
    train_df = train_df.drop(columns=[TARGET])
    if ID_COL in train_df.columns:
        train_df = train_df.drop(columns=[ID_COL])
    if ID_COL in test_df.columns:
        test_df = test_df.drop(columns=[ID_COL])
    return (train_df, test_df, y_train, [])
AGE_ORDER: dict = {'만18-34세': 0, '만35-37세': 1, '만38-39세': 2, '만40-42세': 3, '만43-44세': 4, '만45-50세': 5, '알 수 없음': -1}
AGE_43PLUS_LABELS: List[str] = ['만43-44세', '만45-50세']

def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if '이식된 배아 수' in df.columns:
        df['is_transfer_canceled'] = (pd.to_numeric(df['이식된 배아 수'], errors='coerce').fillna(-1) == 0).astype(int)
    if '시술 유형' in df.columns:
        df['is_di'] = df['시술 유형'].astype(str).str.contains('DI', na=False).astype(int)
    else:
        df['is_di'] = 0
    yes_tokens = {'1', '1.0', '예', 'Y', 'Yes', 'y', 'yes'}
    pgs = df['PGS 시술 여부'].astype(str) if 'PGS 시술 여부' in df.columns else pd.Series([''] * len(df))
    pgd = df['PGD 시술 여부'].astype(str) if 'PGD 시술 여부' in df.columns else pd.Series([''] * len(df))
    df['is_pgt_performed'] = (pgs.isin(yes_tokens) | pgd.isin(yes_tokens)).astype(int)
    if '동결 배아 사용 여부' in df.columns:
        df['is_frozen_cycle'] = (pd.to_numeric(df['동결 배아 사용 여부'], errors='coerce') == 1.0).astype(int)
    else:
        df['is_frozen_cycle'] = 0
    if '난자 혼합 경과일' in df.columns:
        df['is_mix_date_missing'] = (df['난자 혼합 경과일'].isnull() & (df['is_di'] == 0)).astype(int)
    else:
        df['is_mix_date_missing'] = 0
    return df

def add_base_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if '배아 이식 경과일' in df.columns:
        df['is_day5'] = (pd.to_numeric(df['배아 이식 경과일'], errors='coerce') == 5).astype(int)
    else:
        df['is_day5'] = 0
    if '단일 배아 이식 여부' in df.columns:
        df['is_eset'] = (pd.to_numeric(df['단일 배아 이식 여부'], errors='coerce').fillna(0) == 1.0).astype(int)
    else:
        df['is_eset'] = 0
    if '특정 시술 유형' in df.columns:
        df['is_blastocyst'] = df['특정 시술 유형'].astype(str).str.upper().str.contains('BLASTOCYST', na=False).astype(int)
    else:
        df['is_blastocyst'] = 0
    if '저장된 배아 수' in df.columns:
        df['is_no_stored_embryo'] = (pd.to_numeric(df['저장된 배아 수'], errors='coerce').fillna(0) == 0).astype(int)
    else:
        df['is_no_stored_embryo'] = 0
    if '시술 당시 나이' in df.columns:
        df['age_group'] = df['시술 당시 나이'].map(AGE_ORDER).fillna(-1).astype(int)
        df['age_group_43plus'] = df['시술 당시 나이'].isin(AGE_43PLUS_LABELS).astype(int)
    else:
        df['age_group'] = -1
        df['age_group_43plus'] = 0
    return df

def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors='coerce').fillna(0.0)
    den = pd.to_numeric(den, errors='coerce').fillna(0.0)
    return np.where(den > 0, num / den.replace(0, np.nan), 0.0)

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if '미세주입에서 생성된 배아 수' in df.columns and '미세주입된 난자 수' in df.columns:
        df['icsi_fertilization_rate'] = _safe_div(df['미세주입에서 생성된 배아 수'], df['미세주입된 난자 수'])
    else:
        df['icsi_fertilization_rate'] = 0.0
    if '이식된 배아 수' in df.columns and '총 생성 배아 수' in df.columns:
        df['embryo_transfer_pressure'] = _safe_div(df['이식된 배아 수'], df['총 생성 배아 수'])
    else:
        df['embryo_transfer_pressure'] = 0.0
    if '저장된 배아 수' in df.columns and '총 생성 배아 수' in df.columns:
        df['storage_rate'] = _safe_div(df['저장된 배아 수'], df['총 생성 배아 수'])
    else:
        df['storage_rate'] = 0.0
    if 'IVF 시술 횟수' in df.columns and 'IVF 임신 횟수' in df.columns:
        ivf_attempts = to_int_count(df['IVF 시술 횟수']).fillna(0.0)
        ivf_pregs = to_int_count(df['IVF 임신 횟수']).fillna(0.0)
        df['cumulative_ivf_failure'] = (ivf_attempts - ivf_pregs).clip(lower=0).astype(float)
    else:
        df['cumulative_ivf_failure'] = 0.0
    return df

def _add_specific_procedure_tokens(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if '특정 시술 유형' not in df.columns:
        for c in ['has_icsi', 'has_iui', 'has_ah', 'has_mixed_procedure', 'is_unknown_specific']:
            df[c] = 0
        return df
    s = df['특정 시술 유형'].astype(str).str.upper()
    df['has_icsi'] = s.str.contains('ICSI', na=False).astype(int)
    df['has_iui'] = s.str.contains('IUI', na=False).astype(int)
    df['has_ah'] = s.str.contains('\\bAH\\b', regex=True, na=False).astype(int)
    df['has_mixed_procedure'] = (s.str.contains(':', na=False) | s.str.contains('/', na=False)).astype(int)
    df['is_unknown_specific'] = s.str.contains('UNKNOWN', na=False).astype(int)
    return df

def _add_cause_complexity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cause_cols = ['불임 원인 - 남성 요인', '불임 원인 - 난관 질환', '불임 원인 - 배란 장애', '불임 원인 - 자궁내막증', '불임 원인 - 자궁경부 문제', '불명확 불임 원인']
    valid = [c for c in cause_cols if c in df.columns]
    if not valid:
        df['cause_complexity_score'] = 0.0
        df['is_multi_cause'] = 0
        df['multi_cause_young'] = 0
        return df
    score = sum((pd.to_numeric(df[c], errors='coerce').fillna(0.0) for c in valid))
    df['cause_complexity_score'] = score.astype(float)
    df['is_multi_cause'] = (score >= 2).astype(int)
    if 'age_group' in df.columns:
        young = (df['age_group'] == 0).astype(int)
        df['multi_cause_young'] = (df['is_multi_cause'] * young).astype(int)
    else:
        df['multi_cause_young'] = 0
    return df

def _add_count_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _num(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        v = df[col]
        if v.dtype == 'object':
            v = to_int_count(v)
        return pd.to_numeric(v, errors='coerce').fillna(0.0)
    total_proc = _num('총 시술 횟수')
    total_preg = _num('총 임신 횟수')
    total_birth = _num('총 출산 횟수')
    clinic_proc = _num('클리닉 내 총 시술 횟수')
    df['diff_procedures_pregnancies'] = (total_proc - total_preg).clip(lower=0).astype(float)
    df['has_miscarriage'] = (total_preg - total_birth > 0).astype(int)
    df['diff_clinic_external'] = (total_proc - clinic_proc).clip(lower=0).astype(float)
    transfer = _num('이식된 배아 수')
    transfer_icsi = _num('미세주입 배아 이식 수')
    df['diff_transfer_non_icsi'] = (transfer - transfer_icsi).clip(lower=0).astype(float)
    embryo = _num('총 생성 배아 수')
    embryo_icsi = _num('미세주입에서 생성된 배아 수')
    df['diff_embryo_non_icsi'] = (embryo - embryo_icsi).clip(lower=0).astype(float)
    storage = _num('저장된 배아 수')
    storage_icsi = _num('미세주입 후 저장된 배아 수')
    df['diff_storage_non_icsi'] = (storage - storage_icsi).clip(lower=0).astype(float)
    return df

def _add_culture_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _num(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    transfer_day = _num('배아 이식 경과일')
    mix_day = _num('난자 혼합 경과일')
    thaw_day = _num('배아 해동 경과일')
    df['fresh_culture_days'] = (transfer_day - mix_day).astype(float)
    df['frozen_culture_days'] = (transfer_day - thaw_day).astype(float)
    return df

def _add_eset_transfer_boost(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    transfer = pd.to_numeric(df.get('이식된 배아 수', pd.Series(0.0, index=df.index)), errors='coerce').fillna(0.0)
    eset = df.get('is_eset', pd.Series(0, index=df.index)).astype(int)
    boost_mask = (eset == 1) & (transfer == 1)
    adj = transfer.copy()
    adj.loc[boost_mask] = 1.5
    df['transfer_embryo_eset_adj'] = adj.astype(float)
    return df

def _add_ideal_culture_day(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if '배아 이식 경과일' in df.columns:
        v = pd.to_numeric(df['배아 이식 경과일'], errors='coerce')
        df['is_ideal_culture_day'] = v.isin([3, 5]).astype(int)
    else:
        df['is_ideal_culture_day'] = 0
    return df

def _add_period_embryo_combo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    period = df.get('시술 시기 코드', pd.Series('Z', index=df.index)).astype(str).fillna('Z')
    frozen = pd.to_numeric(df.get('동결 배아 사용 여부', 0), errors='coerce').fillna(0).astype(int).astype(str)
    fresh = pd.to_numeric(df.get('신선 배아 사용 여부', 0), errors='coerce').fillna(0).astype(int).astype(str)
    donor = pd.to_numeric(df.get('기증 배아 사용 여부', 0), errors='coerce').fillna(0).astype(int).astype(str)
    df['period_embryo_combo'] = (period + '_' + frozen + fresh + donor).astype(str)
    return df

def _add_pgs_inconsistency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    chk = pd.to_numeric(df.get('착상 전 유전 검사 사용 여부', 0), errors='coerce').fillna(0)
    pgs = pd.to_numeric(df.get('PGS 시술 여부', 0), errors='coerce').fillna(0)
    df['pgs_inconsistency'] = ((chk == 1) & (pgs == 0)).astype(int)
    return df

def _add_prior_solution_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _num(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    def _safe_ratio(num: pd.Series, den: pd.Series, invalid: float=-1.0) -> pd.Series:
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(den > 0, num / den.replace(0, np.nan), invalid)
        return pd.Series(r, index=df.index).astype(float)

    def _count(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        v = df[col]
        if v.dtype == 'object':
            v = to_int_count(v)
        return pd.to_numeric(v, errors='coerce').fillna(0.0)
    ivf_proc = _count('IVF 시술 횟수')
    ivf_preg = _count('IVF 임신 횟수')
    ivf_birth = _count('IVF 출산 횟수')
    di_proc = _count('DI 시술 횟수')
    di_preg = _count('DI 임신 횟수')
    total_preg = _count('총 임신 횟수')
    total_birth = _count('총 출산 횟수')
    clinic_proc = _count('클리닉 내 총 시술 횟수')
    df['ivf_pregnancy_rate'] = _safe_ratio(ivf_preg, ivf_proc)
    df['di_pregnancy_rate'] = _safe_ratio(di_preg, di_proc)
    df['ivf_birth_rate'] = _safe_ratio(ivf_birth, ivf_preg)
    df['total_birth_rate'] = _safe_ratio(total_birth, total_preg)
    df['ivf_procedure_ratio'] = _safe_ratio(ivf_proc, ivf_proc + di_proc)
    df['clinic_procedure_ratio'] = _safe_ratio(clinic_proc, ivf_proc + di_proc)
    fresh_collected = _num('수집된 신선 난자 수')
    fresh_stored = _num('저장된 신선 난자 수')
    icsi_eggs = _num('미세주입된 난자 수')
    mixed_eggs = _num('혼합된 난자 수')
    df['egg_success_rate'] = _safe_ratio(icsi_eggs, fresh_collected)
    df['mixed_vs_icsi_ratio'] = _safe_ratio(icsi_eggs, mixed_eggs)
    net_fresh = (fresh_collected - fresh_stored).clip(lower=0)
    df['fresh_selection_ratio'] = _safe_ratio(mixed_eggs, net_fresh)
    transfer = _num('이식된 배아 수')
    thawed = _num('해동된 배아 수')
    total_embryo = _num('총 생성 배아 수')
    stored = _num('저장된 배아 수')
    df['transfer_vs_all_usage'] = _safe_ratio(transfer, thawed + total_embryo + stored)
    transfer_day = _num('배아 이식 경과일')
    df['transfer_per_day_ratio'] = _safe_ratio(transfer, transfer_day)
    df['total_used_embryo'] = (thawed + total_embryo).astype(float)
    partner_mix = _num('파트너 정자와 혼합된 난자 수')
    donor_mix = _num('기증자 정자와 혼합된 난자 수')
    is_ivf = df.get('시술 유형', pd.Series('', index=df.index)).astype(str).str.contains('IVF', na=False)
    df['is_ivf_no_sperm_mix'] = (is_ivf & (partner_mix == 0) & (donor_mix == 0)).astype(int)
    male_cols = ['불임 원인 - 남성 요인', '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태']
    female_cols = ['불임 원인 - 난관 질환', '불임 원인 - 배란 장애', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증']
    male_sum = sum((_num(c) for c in male_cols))
    female_sum = sum((_num(c) for c in female_cols))
    df['male_cause_severity'] = male_sum.astype(float)
    df['female_cause_severity'] = female_sum.astype(float)
    df['female_severity_ratio'] = _safe_ratio(female_sum, male_sum + female_sum)
    return df

def _add_domain_expert_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _num(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    def _safe_ratio(num: pd.Series, den: pd.Series, invalid: float=-1.0) -> pd.Series:
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(den > 0, num / den.replace(0, np.nan), invalid)
        return pd.Series(r, index=df.index).astype(float)
    total_embryo = _num('총 생성 배아 수')
    mixed_eggs = _num('혼합된 난자 수')
    fresh_collected = _num('수집된 신선 난자 수')
    transfer = _num('이식된 배아 수')
    stored = _num('저장된 배아 수')
    thawed = _num('해동된 배아 수')
    df['total_fertilization_rate'] = _safe_ratio(total_embryo, mixed_eggs)
    df['ovarian_utilization_rate'] = _safe_ratio(fresh_collected, mixed_eggs)
    df['embryo_utilization_rate'] = _safe_ratio(transfer + stored, total_embryo)
    is_day5 = df.get('is_day5', pd.Series(0, index=df.index)).astype(float)
    df['blastocyst_proxy_rate'] = _safe_ratio(is_day5 * transfer, total_embryo)
    df['thaw_dependency_rate'] = _safe_ratio(thawed, total_embryo)
    age_group = df.get('age_group', pd.Series(-1, index=df.index)).astype(float)
    cum_fail = df.get('cumulative_ivf_failure', pd.Series(0.0, index=df.index)).astype(float)
    df['age_x_cumulative_failure'] = (age_group * cum_fail).astype(float)
    ivf_n = df.get('IVF 시술 횟수', pd.Series(0, index=df.index))
    if ivf_n.dtype == 'object':
        ivf_n = to_int_count(ivf_n)
    ivf_n = pd.to_numeric(ivf_n, errors='coerce').fillna(0.0)
    df['transfer_per_ivf_attempt'] = _safe_ratio(transfer, ivf_n + 1, invalid=0.0)
    is_frozen = df.get('is_frozen_cycle', pd.Series(0, index=df.index)).astype(float)
    df['age_x_frozen_cycle'] = (age_group * is_frozen).astype(float)
    return df

def _add_embryo_usage_tier(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _num(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    fresh = _num('수집된 신선 난자 수')
    thawed = _num('해동된 배아 수')
    fresh_tier = pd.Series(0, index=df.index)
    fresh_tier = fresh_tier.mask((fresh > 0) & (fresh <= 7), 1)
    fresh_tier = fresh_tier.mask(fresh > 7, 2)
    thawed_tier = pd.Series(0, index=df.index)
    thawed_tier = thawed_tier.mask((thawed > 0) & (thawed <= 2), 1)
    thawed_tier = thawed_tier.mask(thawed > 2, 2)
    df['fresh_eggs_tier'] = fresh_tier.astype(int)
    df['thawed_embryos_tier'] = thawed_tier.astype(int)
    df['embryo_usage_combo'] = (fresh_tier * 3 + thawed_tier).astype(int)
    return df

def _add_mid_expert_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _num(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    def _safe_ratio(num, den, invalid=-1.0):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(den > 0, num / den.replace(0, np.nan), invalid)
        return pd.Series(r, index=df.index).astype(float)
    transfer = _num('이식된 배아 수')
    stored = _num('저장된 배아 수')
    df['storage_to_transfer_ratio'] = (stored / (transfer + 1.0)).astype(float)
    total_proc = df.get('총 시술 횟수', pd.Series(0, index=df.index))
    total_preg = df.get('총 임신 횟수', pd.Series(0, index=df.index))
    if total_proc.dtype == 'object':
        total_proc = to_int_count(total_proc)
    if total_preg.dtype == 'object':
        total_preg = to_int_count(total_preg)
    total_proc = pd.to_numeric(total_proc, errors='coerce').fillna(0.0)
    total_preg = pd.to_numeric(total_preg, errors='coerce').fillna(0.0)
    df['historical_pregnancy_rate'] = _safe_ratio(total_preg, total_proc)
    fresh = _num('수집된 신선 난자 수')
    df['is_poor_responder'] = ((fresh > 0) & (fresh < 4)).astype(int)
    return df

def _add_donor_rejuvenation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    age_mid = {'만18-34세': 30.0, '만35-37세': 36.0, '만38-39세': 38.5, '만40-42세': 41.0, '만43-44세': 43.5, '만45-50세': 47.0, '알 수 없음': np.nan}
    donor_mid = {'만20세 이하': 19.0, '만21-25세': 23.0, '만26-30세': 28.0, '만31-35세': 33.0, '만36-40세': 38.0, '만41-45세': 43.0, '알 수 없음': np.nan}
    donor_quality_map = {'만20세 이하': 0.85, '만21-25세': 0.93, '만26-30세': 1.0, '만31-35세': 0.88, '만36-40세': 0.7, '만41-45세': 0.5, '알 수 없음': np.nan}
    age_str = df.get('시술 당시 나이', pd.Series('알 수 없음', index=df.index)).astype(str)
    donor_age_str = df.get('난자 기증자 나이', pd.Series('알 수 없음', index=df.index)).astype(str)
    egg_source = df.get('난자 출처', pd.Series('', index=df.index)).astype(str)
    patient_age = age_str.map(age_mid)
    donor_age = donor_age_str.map(donor_mid)
    is_donor_egg = (egg_source == '기증 제공').astype(int)
    effective_age = np.where(is_donor_egg == 1, np.where(donor_age.notna(), donor_age, patient_age), patient_age)
    df['effective_maternal_age'] = pd.to_numeric(effective_age, errors='coerce').astype(float)
    quality = donor_age_str.map(donor_quality_map)
    df['donor_quality_score'] = np.where(is_donor_egg == 1, quality.fillna(0).astype(float), 0.0)
    df['rejuv_gap'] = np.where(is_donor_egg == 1, (patient_age.fillna(0) - donor_age.fillna(0)).astype(float), 0.0)
    is_elderly_patient = age_str.isin(['만43-44세', '만45-50세'])
    is_young_donor = donor_age_str == '만20세 이하'
    df['donor_age_mismatch'] = ((is_donor_egg == 1) & is_elderly_patient & is_young_donor).astype(int)
    df['is_donor_optimal'] = ((is_donor_egg == 1) & donor_age_str.isin(['만21-25세', '만26-30세', '만31-35세'])).astype(int)
    return df

def _add_di_ivf_nan_mask(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    is_di = df.get('is_di', pd.Series(0, index=df.index)).astype(int) == 1
    ivf_only_cols = ['미세주입된 난자 수', '미세주입 배아 이식 수', '파트너 정자와 혼합된 난자 수', '혼합된 난자 수', '수집된 신선 난자 수', '저장된 신선 난자 수', 'icsi_fertilization_rate', 'embryo_transfer_pressure', 'storage_rate']
    for col in ivf_only_cols:
        if col not in df.columns:
            continue
        new_col = f'{col}_ivf_only'
        df[new_col] = np.where(is_di, np.nan, df[col].astype(float))
    return df
EXTRA_FEATURE_REGISTRY: dict = {'specific_procedure_tokens': _add_specific_procedure_tokens, 'cause_complexity': _add_cause_complexity, 'count_diff_features': _add_count_diff_features, 'culture_days': _add_culture_days, 'eset_transfer_boost': _add_eset_transfer_boost, 'ideal_culture_day': _add_ideal_culture_day, 'period_embryo_combo': _add_period_embryo_combo, 'pgs_inconsistency': _add_pgs_inconsistency, 'prior_solution_ratios': _add_prior_solution_ratios, 'domain_expert_ratios': _add_domain_expert_ratios, 'embryo_usage_tier': _add_embryo_usage_tier, 'mid_expert_features': _add_mid_expert_features, 'donor_rejuvenation': _add_donor_rejuvenation, 'di_ivf_nan_mask': _add_di_ivf_nan_mask}
EXTRA_TE_INTERACTION_PAIRS: dict = {'interaction_male_factor_age': ('interaction_male_factor_age', '불임 원인 - 남성 요인', 'age_group')}
INTERACTION_PAIRS: List[Tuple[str, str, str]] = [('interaction_age_day5', 'age_group', 'is_day5'), ('interaction_age_eset', 'age_group', 'is_eset'), ('interaction_eset_day5', 'is_eset', 'is_day5'), ('interaction_cancel_noembryo', 'is_transfer_canceled', '_is_no_embryo_created'), ('interaction_age_di', 'age_group', 'is_di')]

def _make_interaction_key(df: pd.DataFrame, left: str, right: str) -> pd.Series:
    return df[left].astype(str) + '|' + df[right].astype(str)

def _kfold_target_encode(train_keys: pd.Series, test_keys: pd.Series, y: pd.Series, n_splits: int=N_SPLITS, seed: int=SEED, smoothing: float=20.0, round_decimals: int=4) -> Tuple[np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    global_mean = float(y.mean())
    train_te = np.full(len(train_keys), global_mean, dtype=np.float64)
    test_te_folds: List[np.ndarray] = []
    train_keys_arr = train_keys.values
    y_arr = y.values
    for tr_idx, va_idx in skf.split(train_keys_arr, y_arr):
        fold_df = pd.DataFrame({'k': train_keys_arr[tr_idx], 'y': y_arr[tr_idx]})
        grouped = fold_df.groupby('k')['y']
        cat_mean = grouped.mean()
        cat_count = grouped.count()
        smoothed = (cat_count * cat_mean + smoothing * global_mean) / (cat_count + smoothing)
        va_keys = pd.Series(train_keys_arr[va_idx])
        train_te[va_idx] = va_keys.map(smoothed).fillna(global_mean).values
        test_te_folds.append(test_keys.map(smoothed).fillna(global_mean).values.astype(np.float64))
    test_te = np.mean(test_te_folds, axis=0)
    train_te = np.round(train_te, round_decimals)
    test_te = np.round(test_te, round_decimals)
    return (train_te, test_te)

def add_interactions(train_df: pd.DataFrame, test_df: pd.DataFrame, y: pd.Series, n_splits: int=N_SPLITS, extra_pairs: List[str]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    if '총 생성 배아 수' in train_df.columns:
        train_df['_is_no_embryo_created'] = (pd.to_numeric(train_df['총 생성 배아 수'], errors='coerce').fillna(-1) == 0).astype(int)
        test_df['_is_no_embryo_created'] = (pd.to_numeric(test_df['총 생성 배아 수'], errors='coerce').fillna(-1) == 0).astype(int)
    else:
        train_df['_is_no_embryo_created'] = 0
        test_df['_is_no_embryo_created'] = 0
    all_pairs: List[Tuple[str, str, str]] = list(INTERACTION_PAIRS)
    if extra_pairs:
        for key in extra_pairs:
            if key in EXTRA_TE_INTERACTION_PAIRS:
                all_pairs.append(EXTRA_TE_INTERACTION_PAIRS[key])
    for feat_name, left, right in all_pairs:
        if left not in train_df.columns or right not in train_df.columns:
            train_df[feat_name] = 0.0
            test_df[feat_name] = 0.0
            continue
        train_keys = _make_interaction_key(train_df, left, right)
        test_keys = _make_interaction_key(test_df, left, right)
        train_te, test_te = _kfold_target_encode(train_keys, test_keys, y, n_splits=n_splits)
        train_df[feat_name] = train_te
        test_df[feat_name] = test_te
    train_df = train_df.drop(columns=['_is_no_embryo_created'], errors='ignore')
    test_df = test_df.drop(columns=['_is_no_embryo_created'], errors='ignore')
    return (train_df, test_df)

def build_features(raw_train: pd.DataFrame, raw_test: pd.DataFrame, extra_features: List[str]=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    train_df = raw_train.copy()
    test_df = raw_test.copy()
    train_df = add_missing_flags(train_df)
    test_df = add_missing_flags(test_df)
    train_df = add_base_derived(train_df)
    test_df = add_base_derived(test_df)
    train_df = add_ratio_features(train_df)
    test_df = add_ratio_features(test_df)
    train_df, test_df, y_train, _ = preprocess_basic(train_df, test_df)
    drop_cols = [c for c in DROP_MULTICOLLINEAR if c in train_df.columns]
    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols, errors='ignore')
    te_extra_keys: List[str] = []
    if extra_features:
        simple_keys = [k for k in extra_features if k in EXTRA_FEATURE_REGISTRY]
        te_extra_keys = [k for k in extra_features if k in EXTRA_TE_INTERACTION_PAIRS]
        for key in simple_keys:
            fn = EXTRA_FEATURE_REGISTRY[key]
            train_df = fn(train_df)
            test_df = fn(test_df)
    train_df, test_df = add_interactions(train_df, test_df, y_train, extra_pairs=te_extra_keys or None)
    train_df, test_df, cat_cols = encode_categorical(train_df, test_df)
    test_df = test_df[train_df.columns]
    return (train_df, test_df, y_train, cat_cols)
EXTRA_FEATURES_H2_H9: List[str] = ['interaction_male_factor_age', 'specific_procedure_tokens', 'cause_complexity', 'count_diff_features', 'culture_days', 'eset_transfer_boost', 'ideal_culture_day', 'period_embryo_combo', 'pgs_inconsistency', 'prior_solution_ratios', 'domain_expert_ratios', 'embryo_usage_tier', 'mid_expert_features', 'donor_rejuvenation', 'di_ivf_nan_mask']
LGBM_BEST_PARAMS: dict = {'learning_rate': 0.011948517432255166, 'num_leaves': 26, 'max_depth': 10, 'min_child_samples': 188, 'feature_fraction': 0.5456279354970026, 'bagging_fraction': 0.705351790065326, 'bagging_freq': 2, 'reg_alpha': 0.2410094544452788, 'reg_lambda': 0.0004405172434986868, 'n_estimators': 2000}
LGBM_EARLY_STOP = 100

def train_lgbm_5fold(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, cat_cols: List[str], fold_seed: int, model_seed: int) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    import lightgbm as lgb
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=fold_seed)
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs: List[float] = []
    cat_feat = [c for c in cat_cols if c in X.columns]
    base_params = dict(LGBM_BEST_PARAMS)
    base_params.update(objective='binary', metric='auc', random_state=model_seed, n_jobs=-1, verbose=-1)
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        t0 = time.time()
        X_tr, X_va = (X.iloc[tr_idx], X.iloc[va_idx])
        y_tr, y_va = (y.iloc[tr_idx], y.iloc[va_idx])
        model = lgb.LGBMClassifier(**base_params)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc', categorical_feature=cat_feat if cat_feat else 'auto', callbacks=[lgb.early_stopping(LGBM_EARLY_STOP, verbose=False), lgb.log_evaluation(period=0)])
        va_pred = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
        oof[va_idx] = va_pred
        fa = auc_score(y_va.values, va_pred)
        fold_aucs.append(fa)
        test_pred += model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]
        print(f'  [LGBM seed={model_seed} Fold {fold_idx}/5] AUC={fa:.6f} ({time.time() - t0:.1f}s)')
    test_pred /= N_SPLITS
    return (oof, test_pred, fold_aucs)

def run_lgbm_5fold_seedavg(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, cat_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    seed_oofs = []
    seed_preds = []
    for s in SEED_LIST:
        print(f'\n[LGBM 5fold seed={s}] 학습 시작')
        set_seed(s)
        oof, pred, _ = train_lgbm_5fold(X, y, X_test, cat_cols, fold_seed=s, model_seed=s)
        a = auc_score(y.values, oof)
        print(f'  → seed {s} OOF = {a:.6f}')
        seed_oofs.append(oof)
        seed_preds.append(pred)
    oof_avg = np.mean(seed_oofs, axis=0)
    pred_avg = np.mean(seed_preds, axis=0)
    print(f'[LGBM 5fold seedavg] OOF = {auc_score(y.values, oof_avg):.6f}')
    return (oof_avg, pred_avg)
CAT_OLD_PARAMS: dict = dict(loss_function='Logloss', eval_metric='AUC', learning_rate=0.05, depth=6, l2_leaf_reg=3.0, bootstrap_type='Bernoulli', subsample=0.9, rsm=0.9, iterations=2000, random_seed=SEED, verbose=0)
CAT_OLD_EARLY_STOP = 100

def train_catboost_5fold(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, cat_cols: List[str], seed: int=SEED) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    from catboost import CatBoostClassifier, Pool
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs: List[float] = []
    cat_in_cols = [c for c in cat_cols if c in X.columns]
    X_cast = X.copy()
    X_test_cast = X_test.copy()
    for c in cat_in_cols:
        X_cast[c] = X_cast[c].astype('int64')
        if c in X_test_cast.columns:
            X_test_cast[c] = X_test_cast[c].astype('int64')
    test_pool = Pool(data=X_test_cast, cat_features=cat_in_cols)
    params = dict(CAT_OLD_PARAMS)
    params['random_seed'] = seed
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_cast, y), start=1):
        t0 = time.time()
        X_tr, X_va = (X_cast.iloc[tr_idx], X_cast.iloc[va_idx])
        y_tr, y_va = (y.iloc[tr_idx], y.iloc[va_idx])
        train_pool = Pool(data=X_tr, label=y_tr.values, cat_features=cat_in_cols)
        valid_pool = Pool(data=X_va, label=y_va.values, cat_features=cat_in_cols)
        model = CatBoostClassifier(**params, early_stopping_rounds=CAT_OLD_EARLY_STOP)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        va_pred = model.predict_proba(valid_pool)[:, 1]
        oof[va_idx] = va_pred
        fa = auc_score(y_va.values, va_pred)
        fold_aucs.append(fa)
        test_pred += model.predict_proba(test_pool)[:, 1]
        print(f'  [CAT Fold {fold_idx}/5] AUC={fa:.6f} ({time.time() - t0:.1f}s)')
    test_pred /= N_SPLITS
    return (oof, test_pred, fold_aucs)
MLP_PARAMS_BASE: dict = dict(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=0.0001, batch_size=512, learning_rate='adaptive', learning_rate_init=0.001, max_iter=100, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, verbose=False)

def train_mlp_5fold(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, seed: int=SEED) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    X_arr = X.fillna(0).to_numpy(dtype=np.float64)
    X_te_arr = X_test.fillna(0).to_numpy(dtype=np.float64)
    y_arr = y.to_numpy()
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs: List[float] = []
    params = dict(MLP_PARAMS_BASE)
    params['random_state'] = seed
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_arr), start=1):
        t0 = time.time()
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_arr[tr_idx])
        X_va_s = scaler.transform(X_arr[va_idx])
        X_te_s = scaler.transform(X_te_arr)
        model = MLPClassifier(**params)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_tr_s, y_arr[tr_idx])
        va_pred = model.predict_proba(X_va_s)[:, 1]
        oof[va_idx] = va_pred
        test_pred += model.predict_proba(X_te_s)[:, 1]
        a = auc_score(y_arr[va_idx], va_pred)
        fold_aucs.append(a)
        print(f'  [MLP seed={seed} Fold {fold_idx}/5] AUC={a:.6f} (n_iter={model.n_iter_}, {time.time() - t0:.1f}s)')
    test_pred /= N_SPLITS
    return (oof, test_pred, fold_aucs)

def train_mlp_h2h9_seedavg(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, seeds: List[int]=SEED_LIST) -> Tuple[np.ndarray, np.ndarray]:
    print(f'\n  [MLP h2h9 seedavg] seeds={seeds}, 5fold × {len(seeds)} = {5 * len(seeds)} fits')
    oof_list: List[np.ndarray] = []
    pred_list: List[np.ndarray] = []
    for seed in seeds:
        set_seed(seed)
        oof_s, pred_s, fold_aucs = train_mlp_5fold(X, y, X_test, seed=seed)
        seed_oof_auc = auc_score(y.to_numpy(), oof_s)
        print(f'  [MLP seed={seed}] OOF AUC={seed_oof_auc:.6f}, fold mean={np.mean(fold_aucs):.6f}')
        oof_list.append(oof_s)
        pred_list.append(pred_s)
    oof_avg = np.mean(np.stack(oof_list, axis=0), axis=0)
    pred_avg = np.mean(np.stack(pred_list, axis=0), axis=0)
    print(f'  [MLP h2h9 seedavg] 평균 OOF AUC={auc_score(y.to_numpy(), oof_avg):.6f}')
    return (oof_avg, pred_avg)
FINAL_WEIGHTS: dict = {'lgbm_h2h9': 0.55, 'cat_h2h9': 0.265, 'mlp_h2h9': 0.185}
COMP_FILES: dict = {'lgbm_h2h9': ('lgbm_h2h9_seedavg_exp_059.npy', 'lgbm_h2h9_seedavg_exp_059.npy'), 'cat_h2h9': ('catboost_h2h9_exp_064.npy', 'catboost_h2h9_exp_064.npy'), 'mlp_h2h9': ('mlp_h2h9_seedavg_exp_074.npy', 'mlp_h2h9_seedavg_exp_074.npy')}

def cached(comp: str) -> bool:
    oof_name, pred_name = COMP_FILES[comp]
    return (OOF_DIR / oof_name).exists() and (PREDS_DIR / pred_name).exists()

def load_component(comp: str) -> Tuple[np.ndarray, np.ndarray]:
    oof_name, pred_name = COMP_FILES[comp]
    return (np.load(OOF_DIR / oof_name), np.load(PREDS_DIR / pred_name))

def save_component(comp: str, oof: np.ndarray, pred: np.ndarray) -> None:
    oof_name, pred_name = COMP_FILES[comp]
    np.save(OOF_DIR / oof_name, oof)
    np.save(PREDS_DIR / pred_name, pred)

def make_submission(test_blend: np.ndarray, out_name: str='final_submission_260428_v62h.csv') -> Path:
    sample_path = SAMPLE_SUBMISSION_CSV
    if not sample_path.exists():
        alt = DATA_DIR / 'sample_submission.csv'
        if alt.exists():
            sample_path = alt
    sample = pd.read_csv(sample_path)
    sample.iloc[:, 1] = test_blend
    out = SUBMISSION_DIR / out_name
    sample.to_csv(out, index=False)
    return out

def main():
    parser = argparse.ArgumentParser(description='난임 임신 성공 예측 — v62h 최종 챔피언 (사이클 5)')
    parser.add_argument('--skip_train', action='store_true', help='모든 캐시 hit 가정. 학습 skip → 블렌드만.')
    parser.add_argument('--out_name', type=str, default='final_submission_260428_v62h.csv')
    args = parser.parse_args()
    t_global = time.time()
    print('=' * 78)
    print('난임 환자 임신 성공 여부 예측 — v62h 최종 챔피언 (사이클 5, 2026-04-29)')
    print('AI헬스케어 3기 해커톤 / DACON x 오즈코딩스쿨 / jun8741@gmail.com')
    print('3위 코드 0% — 100% 자체 작성 코드')
    print('=' * 78)
    print(f'\n[Step 0] 데이터 로드: {TRAIN_CSV}, {TEST_CSV}')
    raw_train = pd.read_csv(TRAIN_CSV)
    raw_test = pd.read_csv(TEST_CSV)
    print(f'  train: {raw_train.shape},  test: {raw_test.shape}')
    needs_features = not all((cached(c) for c in ['lgbm_h2h9', 'cat_h2h9', 'mlp_h2h9']))
    if needs_features and (not args.skip_train):
        print('\n[Step 1] H2+H9 EXTRAS features 빌드 (LGBM/CAT/MLP 공통)')
        set_seed(SEED)
        X, X_te, y, cat_cols = build_features(raw_train, raw_test, extra_features=EXTRA_FEATURES_H2_H9)
        print(f'  X: {X.shape},  X_te: {X_te.shape},  cat_cols: {len(cat_cols)}')
    if cached('lgbm_h2h9'):
        print('\n[Step 2] LGBM h2h9 5fold seedavg — 캐시 hit, skip')
        lgbm_oof, lgbm_pred = load_component('lgbm_h2h9')
    elif args.skip_train:
        raise FileNotFoundError('LGBM h2h9 캐시 없음.')
    else:
        print('\n[Step 2] LGBM h2h9 5fold seedavg 학습 (~15분)')
        lgbm_oof, lgbm_pred = run_lgbm_5fold_seedavg(X, y, X_te, cat_cols)
        save_component('lgbm_h2h9', lgbm_oof, lgbm_pred)
    if cached('cat_h2h9'):
        print('\n[Step 3] CAT h2h9 5fold — 캐시 hit, skip')
        cat_oof, cat_pred = load_component('cat_h2h9')
    elif args.skip_train:
        raise FileNotFoundError('CAT h2h9 캐시 없음.')
    else:
        print('\n[Step 3] CAT h2h9 5fold 학습 (~20분)')
        set_seed(SEED)
        cat_oof, cat_pred, _ = train_catboost_5fold(X, y, X_te, cat_cols, seed=SEED)
        save_component('cat_h2h9', cat_oof, cat_pred)
    if cached('mlp_h2h9'):
        print('\n[Step 4] MLP h2h9 5fold seedavg — 캐시 hit, skip')
        mlp_oof, mlp_pred = load_component('mlp_h2h9')
    elif args.skip_train:
        raise FileNotFoundError('MLP h2h9 캐시 없음.')
    else:
        print('\n[Step 4] MLP h2h9 5fold seedavg 학습 (~10분, 5fold × 3 seed)')
        mlp_oof, mlp_pred = train_mlp_h2h9_seedavg(X, y, X_te, seeds=SEED_LIST)
        save_component('mlp_h2h9', mlp_oof, mlp_pred)
    print('\n[Step 5] 3-component v62h 가중 블렌드 (사이클 5)')
    y_true = pd.read_csv(TRAIN_CSV)[TARGET].values
    oofs = {'lgbm_h2h9': lgbm_oof, 'cat_h2h9': cat_oof, 'mlp_h2h9': mlp_oof}
    preds = {'lgbm_h2h9': lgbm_pred, 'cat_h2h9': cat_pred, 'mlp_h2h9': mlp_pred}
    print('  단독 OOF:')
    for name, oof in oofs.items():
        print(f'    {name:>12}: {auc_score(y_true, oof):.6f}')
    weights = np.array([FINAL_WEIGHTS[k] for k in oofs.keys()])
    print(f'\n  가중치 (사이클 6 Hill climb 재탐색): {dict(zip(oofs.keys(), weights))}')
    oof_blend = np.zeros(len(y_true))
    test_blend = np.zeros(len(preds['cat_h2h9']))
    for w, name in zip(weights, oofs.keys()):
        oof_blend += w * oofs[name]
        test_blend += w * preds[name]
    print(f'\n  블렌드 OOF AUC: {auc_score(y_true, oof_blend):.6f}')
    out_path = make_submission(test_blend, args.out_name)
    print(f'\n[Step 6] submission 저장: {out_path}')
    print(f'\n총 소요: {(time.time() - t_global) / 60:.1f}분')
    print(f'검증된 LB (v62h_hillclimb 사이클 6): 0.7423993  (v62h_simple +0.0000031)')
    print('3위 코드 0% — 실격 위험 0')
    print('=' * 78)
if __name__ == '__main__':
    main()
