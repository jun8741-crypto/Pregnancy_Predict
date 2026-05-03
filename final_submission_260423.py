from __future__ import annotations
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
SEED: int = 42
ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT / 'data'
SUBMISSION_DIR: Path = ROOT / 'submission'
OUTPUTS_DIR: Path = ROOT / 'outputs'
OOF_DIR: Path = OUTPUTS_DIR / 'oof'
PREDS_DIR: Path = OUTPUTS_DIR / 'preds'
MODELS_DIR: Path = ROOT / 'models'
TRAIN_CSV: Path = DATA_DIR / 'train.csv'
TEST_CSV: Path = DATA_DIR / 'test.csv'
SAMPLE_SUBMISSION_CSV: Path = SUBMISSION_DIR / 'sample_submission.csv'
FINAL_SUBMISSION_CSV: Path = SUBMISSION_DIR / 'submission_final.csv'
ID_COL: str = 'ID'
TARGET: str = '임신 성공 여부'
N_SPLITS: int = 5
LGBM_SEEDS: List[int] = [42, 2025, 7]
LGBM_OPT_PARAMS: Dict = dict(objective='binary', metric='auc', learning_rate=0.011948517432255166, num_leaves=26, max_depth=10, min_child_samples=188, feature_fraction=0.5456279354970026, bagging_fraction=0.705351790065326, bagging_freq=2, reg_alpha=0.2410094544452788, reg_lambda=0.0004405172434986868, n_estimators=2000, n_jobs=-1, verbose=-1)
CAT_PARAMS: Dict = dict(loss_function='Logloss', eval_metric='AUC', learning_rate=0.05, depth=6, l2_leaf_reg=3.0, bootstrap_type='Bernoulli', subsample=0.9, rsm=0.9, iterations=2000, random_seed=SEED, verbose=0)
MLP_PARAMS: Dict = dict(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=0.0001, batch_size=512, learning_rate='adaptive', learning_rate_init=0.001, max_iter=100, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=SEED, verbose=False)
EARLY_STOPPING_ROUNDS: int = 100

def set_seed(seed: int=SEED) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    return float(roc_auc_score(y_true, y_score))

def get_skf(n_splits: int=N_SPLITS, seed: int=SEED) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

def _ensure_dirs() -> None:
    for d in (SUBMISSION_DIR, OUTPUTS_DIR, OOF_DIR, PREDS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

def _check_data_files() -> None:
    missing = [str(p) for p in (TRAIN_CSV, TEST_CSV, SAMPLE_SUBMISSION_CSV) if not p.exists()]
    if missing:
        raise FileNotFoundError('다음 데이터 파일을 찾을 수 없습니다:\n  - ' + '\n  - '.join(missing) + f'\n\n프로젝트 루트({ROOT}) 기준으로 data/ 와 submission/sample_submission.csv 를 배치해 주세요.')

def load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV)

def load_test() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV)

def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_SUBMISSION_CSV)
COUNT_COLUMNS: List[str] = ['총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수', '클리닉 내 총 시술 횟수']
COMPRESS_DI_NUMERIC: List[str] = ['혼합된 난자 수', '단일 배아 이식 여부', '신선 배아 사용 여부', '이식된 배아 수', '저장된 배아 수', '저장된 신선 난자 수', '착상 전 유전 진단 사용 여부', '총 생성 배아 수', '미세주입에서 생성된 배아 수', '미세주입된 난자 수', '미세주입 후 저장된 배아 수', '수집된 신선 난자 수', '동결 배아 사용 여부', '대리모 여부', '미세주입 배아 이식 수', '기증 배아 사용 여부', '기증자 정자와 혼합된 난자 수', '해동된 배아 수', '파트너 정자와 혼합된 난자 수', '해동 난자 수', '난자 해동 경과일', '배아 해동 경과일', '난자 채취 경과일', 'PGS 시술 여부', 'PGD 시술 여부', '착상 전 유전 검사 사용 여부']
COMPRESS_DI_CATEGORICAL: List[str] = ['배아 생성 주요 이유']
DROP_LOW_INFO: List[str] = ['임신 시도 또는 마지막 임신 경과 연수']
DROP_NO_VARIANCE: List[str] = ['불임 원인 - 여성 요인']
DROP_MULTICOLLINEAR: List[str] = ['미세주입에서 생성된 배아 수']
DROP_AFTER_PGT_FLAG: List[str] = []

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

def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
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
LOG1P_COLUMNS: List[str] = []
SINGLE_TE_COLUMNS: List[str] = []
INTERACTION_PAIRS: List[Tuple[str, str, str]] = [('interaction_age_day5', 'age_group', 'is_day5'), ('interaction_age_eset', 'age_group', 'is_eset'), ('interaction_eset_day5', 'is_eset', 'is_day5'), ('interaction_cancel_noembryo', 'is_transfer_canceled', '_is_no_embryo_created'), ('interaction_age_di', 'age_group', 'is_di')]

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
        ivf_attempts = pd.to_numeric(df['IVF 시술 횟수'], errors='coerce').fillna(0)
        ivf_pregs = pd.to_numeric(df['IVF 임신 횟수'], errors='coerce').fillna(0)
        df['cumulative_ivf_failure'] = (ivf_attempts - ivf_pregs).clip(lower=0).astype(float)
    else:
        df['cumulative_ivf_failure'] = 0.0
    return df

def add_log1p_features(df: pd.DataFrame, cols: List[str]=LOG1P_COLUMNS) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors='coerce').fillna(0.0).clip(lower=0.0)
        df[f'{c}_log1p'] = np.log1p(v).astype(np.float64)
    return df

def _make_interaction_key(df: pd.DataFrame, left: str, right: str) -> pd.Series:
    return df[left].astype(str) + '|' + df[right].astype(str)

def _kfold_target_encode(train_keys: pd.Series, test_keys: pd.Series, y: pd.Series, n_splits: int=N_SPLITS, seed: int=SEED, smoothing: float=0.0) -> Tuple[np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    global_mean = float(y.mean())
    train_te = np.full(len(train_keys), global_mean, dtype=np.float64)
    train_keys_arr = train_keys.values
    y_arr = y.values
    for tr_idx, va_idx in skf.split(train_keys_arr, y_arr):
        fold_df = pd.DataFrame({'k': train_keys_arr[tr_idx], 'y': y_arr[tr_idx]})
        means = fold_df.groupby('k')['y'].mean()
        va_keys = pd.Series(train_keys_arr[va_idx])
        train_te[va_idx] = va_keys.map(means).fillna(global_mean).values
    full_means = pd.DataFrame({'k': train_keys_arr, 'y': y_arr}).groupby('k')['y'].mean()
    test_te = test_keys.map(full_means).fillna(global_mean).values.astype(np.float64)
    return (train_te, test_te)

def add_target_encoded_features(train_df: pd.DataFrame, test_df: pd.DataFrame, y: pd.Series, cols: List[str]=SINGLE_TE_COLUMNS, n_splits: int=N_SPLITS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    for c in cols:
        if c not in train_df.columns:
            continue
        train_keys = train_df[c].astype(str).fillna('__missing__')
        test_keys = test_df[c].astype(str).fillna('__missing__') if c in test_df.columns else None
        train_te, test_te = _kfold_target_encode(train_keys, test_keys, y, n_splits=n_splits)
        train_df[c] = train_te.astype(np.float64)
        if test_keys is not None:
            test_df[c] = test_te.astype(np.float64)
    return (train_df, test_df)

def add_interactions(train_df: pd.DataFrame, test_df: pd.DataFrame, y: pd.Series, n_splits: int=N_SPLITS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    if '총 생성 배아 수' in train_df.columns:
        train_df['_is_no_embryo_created'] = (pd.to_numeric(train_df['총 생성 배아 수'], errors='coerce').fillna(-1) == 0).astype(int)
        test_df['_is_no_embryo_created'] = (pd.to_numeric(test_df['총 생성 배아 수'], errors='coerce').fillna(-1) == 0).astype(int)
    else:
        train_df['_is_no_embryo_created'] = 0
        test_df['_is_no_embryo_created'] = 0
    for feat_name, left, right in INTERACTION_PAIRS:
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

def build_features(raw_train: pd.DataFrame, raw_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    train_df = raw_train.copy()
    test_df = raw_test.copy()
    train_df = add_missing_flags(train_df)
    test_df = add_missing_flags(test_df)
    train_df = add_base_derived(train_df)
    test_df = add_base_derived(test_df)
    train_df = add_ratio_features(train_df)
    test_df = add_ratio_features(test_df)
    train_df, test_df, y_train, _ = preprocess(train_df, test_df)
    drop_cols = [c for c in DROP_MULTICOLLINEAR if c in train_df.columns]
    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols, errors='ignore')
    train_df = add_log1p_features(train_df)
    test_df = add_log1p_features(test_df)
    train_df, test_df = add_target_encoded_features(train_df, test_df, y_train)
    train_df, test_df = add_interactions(train_df, test_df, y_train)
    train_df, test_df, cat_cols = encode_categorical(train_df, test_df)
    test_df = test_df[train_df.columns]
    return (train_df, test_df, y_train, cat_cols)

def train_lgbm_kfold_seed(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, cat_cols: List[str], base_params: Dict, seed: int, n_splits: int=N_SPLITS) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    import lightgbm as lgb
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs: List[float] = []
    cat_feat = [c for c in cat_cols if c in X.columns]
    params = dict(base_params)
    params['random_state'] = seed
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = (X.iloc[tr_idx], X.iloc[va_idx])
        y_tr, y_va = (y.iloc[tr_idx], y.iloc[va_idx])
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc', categorical_feature=cat_feat if cat_feat else 'auto', callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(period=0)])
        va_pred = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
        oof[va_idx] = va_pred
        fold_auc = auc(y_va.values, va_pred)
        fold_aucs.append(fold_auc)
        print(f'    [seed={seed} fold {fold_idx}/{n_splits}] AUC = {fold_auc:.6f}  (best_iter={model.best_iteration_})')
        test_pred += model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]
    test_pred /= n_splits
    return (oof, test_pred, fold_aucs)

def train_catboost_kfold(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, cat_cols: List[str], n_splits: int=N_SPLITS, seed: int=SEED) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    from catboost import CatBoostClassifier, Pool
    skf = get_skf(n_splits=n_splits, seed=seed)
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
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_cast, y), start=1):
        X_tr, X_va = (X_cast.iloc[tr_idx], X_cast.iloc[va_idx])
        y_tr, y_va = (y.iloc[tr_idx], y.iloc[va_idx])
        train_pool = Pool(data=X_tr, label=y_tr.values, cat_features=cat_in_cols)
        valid_pool = Pool(data=X_va, label=y_va.values, cat_features=cat_in_cols)
        model = CatBoostClassifier(**CAT_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        va_pred = model.predict_proba(valid_pool)[:, 1]
        oof[va_idx] = va_pred
        fold_auc = auc(y_va.values, va_pred)
        fold_aucs.append(fold_auc)
        print(f'  [CAT Fold {fold_idx}/{n_splits}] AUC = {fold_auc:.6f}  (best_iter={model.get_best_iteration()})')
        test_pred += model.predict_proba(test_pool)[:, 1]
    test_pred /= n_splits
    return (oof, test_pred, fold_aucs)

def _fit_predict_mlp(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, X_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    model = MLPClassifier(**MLP_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X_tr, y_tr)
    return (model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1], int(model.n_iter_))

def train_mlp_kfold(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, n_splits: int=N_SPLITS, seed: int=SEED) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    skf = get_skf(n_splits=n_splits, seed=seed)
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs: List[float] = []
    X_arr = X.to_numpy(dtype=np.float64)
    X_test_arr = X_test.to_numpy(dtype=np.float64)
    y_arr = y.to_numpy()
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_arr), start=1):
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_arr[tr_idx])
        X_va_s = scaler.transform(X_arr[va_idx])
        X_te_s = scaler.transform(X_test_arr)
        va_pred, te_pred, n_iter = _fit_predict_mlp(X_tr_s, y_arr[tr_idx], X_va_s, X_te_s)
        oof[va_idx] = va_pred
        test_pred += te_pred
        fold_auc = auc(y_arr[va_idx], va_pred)
        fold_aucs.append(fold_auc)
        print(f'  [MLP Fold {fold_idx}/{n_splits}] AUC = {fold_auc:.6f}  (n_iter={n_iter})')
    test_pred /= n_splits
    return (oof, test_pred, fold_aucs)

def _project_to_simplex(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= 1e-12:
        return np.full_like(w, 1.0 / len(w))
    return w / s

def find_optimal_blend_weights(oofs: List[np.ndarray], y: np.ndarray) -> Tuple[np.ndarray, float]:
    K = len(oofs)
    stack = np.column_stack(oofs)

    def _neg_auc(raw_w: np.ndarray) -> float:
        w = _project_to_simplex(raw_w)
        blended = stack @ w
        return -auc(y, blended)
    x0 = np.full(K, 1.0 / K)
    try:
        res = minimize(_neg_auc, x0, method='Nelder-Mead', options={'xatol': 1e-05, 'fatol': 1e-07, 'maxiter': 500})
        weights = _project_to_simplex(res.x)
        blend_auc = -float(res.fun)
        if not np.isfinite(blend_auc) or blend_auc < auc(y, stack @ x0):
            print('  경고: 최적화 결과가 동등 평균보다 낮아 fallback (1/K) 사용')
            weights = x0
            blend_auc = auc(y, stack @ x0)
    except Exception as exc:
        print(f'  경고: scipy 최적화 실패 ({exc}) — 동등 평균 fallback')
        weights = x0
        blend_auc = auc(y, stack @ x0)
    return (weights, blend_auc)

def compute_oof_correlations(oofs: List[np.ndarray], names: List[str]) -> List[Tuple[str, str, float]]:
    pairs: List[Tuple[str, str, float]] = []
    K = len(oofs)
    for i in range(K):
        for j in range(i + 1, K):
            r, _ = pearsonr(oofs[i], oofs[j])
            pairs.append((names[i], names[j], float(r)))
    return pairs

def save_individual_outputs(oof: np.ndarray, test_pred: np.ndarray, tag: str) -> Tuple[Path, Path]:
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    oof_path = OOF_DIR / f'{tag}.npy'
    pred_path = PREDS_DIR / f'{tag}.npy'
    np.save(oof_path, oof)
    np.save(pred_path, test_pred)
    return (oof_path, pred_path)

def save_final_submission(test_pred: np.ndarray, test_ids: pd.Series) -> Path:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    sample = load_sample_submission()
    target_col = [c for c in sample.columns if c != 'ID'][0]
    sub = pd.DataFrame({'ID': test_ids.values, target_col: test_pred})
    sub.to_csv(FINAL_SUBMISSION_CSV, index=False)
    return FINAL_SUBMISSION_CSV

def main() -> None:
    set_seed(SEED)
    _ensure_dirs()
    _check_data_files()
    t_total0 = time.time()
    print('[1/6] 데이터 로드')
    train_df = load_train()
    test_df = load_test()
    test_ids = test_df[ID_COL].copy()
    print(f'      train: {train_df.shape}, test: {test_df.shape}')
    print('\n[2/6] 전처리 + 피처 엔지니어링 (build_features)')
    X_train, X_test, y_train, cat_cols = build_features(train_df, test_df)
    print(f'      X_train: {X_train.shape}, X_test: {X_test.shape}, y mean: {y_train.mean():.4f}, cat_cols: {len(cat_cols)}개')
    n_features = X_train.shape[1]
    y_arr = y_train.to_numpy()
    timing: Dict[str, float] = {}
    print(f'\n[3/6] LightGBM_opt 시드 앙상블 (seeds={LGBM_SEEDS} × {N_SPLITS}-Fold)')
    seed_oofs: Dict[int, np.ndarray] = {}
    seed_preds: Dict[int, np.ndarray] = {}
    seed_aucs: Dict[int, float] = {}
    t0 = time.time()
    for s in LGBM_SEEDS:
        print(f'  >> seed={s}')
        ts = time.time()
        oof_s, pred_s, _ = train_lgbm_kfold_seed(X_train, y_train, X_test, cat_cols, LGBM_OPT_PARAMS, seed=s)
        seed_oofs[s] = oof_s
        seed_preds[s] = pred_s
        a_s = auc(y_arr, oof_s)
        seed_aucs[s] = a_s
        save_individual_outputs(oof_s, pred_s, f'lgbm_opt_seed{s}_final')
        print(f'     seed={s} 단일 OOF AUC = {a_s:.6f} ({time.time() - ts:.1f}초)')
    oof_lgbm_avg = np.mean([seed_oofs[s] for s in LGBM_SEEDS], axis=0)
    pred_lgbm_avg = np.mean([seed_preds[s] for s in LGBM_SEEDS], axis=0)
    lgbm_avg_auc = auc(y_arr, oof_lgbm_avg)
    save_individual_outputs(oof_lgbm_avg, pred_lgbm_avg, 'lgbm_opt_seedavg_final')
    timing['lgbm'] = time.time() - t0
    print(f"      평균 LGBM_opt OOF AUC = {lgbm_avg_auc:.6f}  (시드 앙상블 {len(LGBM_SEEDS)}개, {timing['lgbm']:.1f}초)")
    print('\n[4/6] CatBoost 5-Fold OOF')
    t0 = time.time()
    oof_cat, pred_cat, cat_aucs = train_catboost_kfold(X_train, y_train, X_test, cat_cols=cat_cols)
    timing['cat'] = time.time() - t0
    save_individual_outputs(oof_cat, pred_cat, 'catboost_final')
    cat_oof_auc = auc(y_arr, oof_cat)
    print(f"      CAT OOF AUC = {cat_oof_auc:.6f}  (fold mean {np.mean(cat_aucs):.6f} ± {np.std(cat_aucs):.6f}, {timing['cat']:.1f}초)")
    print('\n[5/6] MLP(128,64) 5-Fold OOF (StandardScaler fold-내 fit)')
    t0 = time.time()
    oof_mlp, pred_mlp, mlp_aucs = train_mlp_kfold(X_train, y_train, X_test)
    timing['mlp'] = time.time() - t0
    save_individual_outputs(oof_mlp, pred_mlp, 'mlp_final')
    mlp_oof_auc = auc(y_arr, oof_mlp)
    print(f"      MLP OOF AUC = {mlp_oof_auc:.6f}  (fold mean {np.mean(mlp_aucs):.6f} ± {np.std(mlp_aucs):.6f}, {timing['mlp']:.1f}초)")
    print('\n[6/6] 3M 가중 블렌딩 (LGBM_avg + CAT + MLP)')
    names_3 = ['LGBM_opt_avg', 'CAT', 'MLP']
    oofs_3 = [oof_lgbm_avg, oof_cat, oof_mlp]
    preds_3 = [pred_lgbm_avg, pred_cat, pred_mlp]
    single_aucs: Dict[str, float] = {'LGBM_opt_avg': lgbm_avg_auc, 'CAT': cat_oof_auc, 'MLP': mlp_oof_auc}
    print('  단일 OOF AUC: ' + ' / '.join((f'{k}={v:.6f}' for k, v in single_aucs.items())))
    corrs = compute_oof_correlations(oofs_3, names_3)
    print('  OOF Pearson 상관:')
    for a, b, r in corrs:
        flag = ''
        if r >= 0.98:
            flag = ' (≥0.98 — 다양성 부족)'
        elif r < 0.95:
            flag = ' (<0.95 — 다양성 OK)'
        print(f'    {a}-{b}: r = {r:.4f}{flag}')
    weights, blend_auc = find_optimal_blend_weights(oofs_3, y_arr)
    weight_str = ', '.join((f'{n}={w:.3f}' for n, w in zip(names_3, weights)))
    print(f'\n  최적 가중치: [{weight_str}]')
    print(f'  블렌딩 OOF AUC = {blend_auc:.6f}')
    best_single_name = max(single_aucs, key=single_aucs.get)
    best_single_auc = single_aucs[best_single_name]
    delta = blend_auc - best_single_auc
    print(f'  단일 최고 ({best_single_name}) {best_single_auc:.6f} → 블렌딩 {blend_auc:.6f} (개선폭 {delta:+.6f})')
    blend_oof = np.column_stack(oofs_3) @ weights
    blend_pred = np.column_stack(preds_3) @ weights
    save_individual_outputs(blend_oof, blend_pred, 'blend_final')
    sub_path = save_final_submission(blend_pred, test_ids)
    print(f'\n  submission 저장: {sub_path}')
    print('\n' + '=' * 70)
    print('최종 요약 (exp_011 재현)')
    print('=' * 70)
    print(f'  피처 수: {n_features}개')
    print(f'  LGBM seed별 OOF AUC: ' + ' / '.join((f'seed{s}={a:.6f}' for s, a in seed_aucs.items())))
    print(f'  LGBM_opt 평균 OOF AUC: {lgbm_avg_auc:.6f}')
    print(f'  CatBoost OOF AUC    : {cat_oof_auc:.6f}')
    print(f'  MLP OOF AUC         : {mlp_oof_auc:.6f}')
    print(f'  3M 블렌딩 OOF AUC   : {blend_auc:.6f}')
    print(f'  최종 가중치         : [{weight_str}]')
    print(f"\n  소요 — LGBM(seed×{len(LGBM_SEEDS)}) {timing['lgbm']:.1f}s / CAT {timing['cat']:.1f}s / MLP {timing['mlp']:.1f}s")
    print(f'  총 소요 시간: {time.time() - t_total0:.1f}초')
    print('=' * 70)
if __name__ == '__main__':
    main()
