# 난임 환자 임신 성공 여부 예측 — AI 솔루션

> **AI헬스케어 3기 해커톤 (DACON × 오즈코딩스쿨, 2026-04)**
>
> 256K 임상 데이터 기반 임신 성공 여부 예측 (ROC-AUC)
>
> **Public LB 0.7423993**

---

## TL;DR

> *"단일 행 임상 피처의 천장을 **표현 변환 카드**로 돌파했다."*

- **임상-ML hybrid 접근**: 산부인과 도메인 지식 + 외부 학술 논문 4편 검증을 통해 단순 ML이 도달하지 못하는 신호를 추가
- **체계적 ablation**: 26+ 카드 시도로 단변수 phenotype 천장의 메커니즘을 정확히 규명
- **차원 전환의 발견**: phenotype clustering이 아닌 *표현 변환*(NaN-mask, age replacement)이 LightGBM 자체 학습 한계를 넘는 본질이라는 메타 인사이트 도출
- **Adversarial Validation 기반 leakage 제거**: 마감 D-1에 interaction TE 누수를 발견하고 수정 → CV-LB 비례 22.5% → 47.4%로 2배 개선
- **결과**: 베이스라인 v10 LB 0.74211에서 6사이클의 단계적 개선을 통해 최종 LB 0.7423993까지 누적 +0.000289 게인 달성

---

## 1. 문제 정의 — 데이터가 던지는 도전

| 항목 | 값 |
|------|------|
| 학습 데이터 | 256,351 행 × 68 컬럼 (한국어 임상 컬럼) |
| 테스트 데이터 | 90,067 행 × 68 컬럼 |
| 양성률 | 25.83% (약한 클래스 불균형) |
| IVF / DI 비율 | 97.55% / 2.45% (구조적 segment) |
| 평가지표 | ROC-AUC (Public LB 100% Test) |
| 학술 상한 | ~0.745 (관련 학술·산업 보고 종합) |

### 데이터의 본질적 어려움

- **DI vs IVF 구조적 분리**: 결측의 80%+가 DI cycle의 시술-비적용 구조적 결측. 동일한 NaN이라도 *모집단이 다름*.
- **Fertility cliff**: 만35세 이후 양성률 급감 (만45-50세 양성률 6.8%, lift 0.26). ESHRE/ASRM 임상 cliff와 정확히 일치.
- **Donor age paradox**: 기증 난자 사용 시 환자 본인 나이가 아닌 *donor 나이*가 임신율을 결정. 단순 환자 나이 변수는 반대 신호를 줄 수 있음.
- **모델 천장의 진짜 원인**: SHAP·오답 분석 결과, Day5 이식 / eSET / Frozen cycle 같은 *임상 실무상 질 높은 시술* 세그먼트에서 AUC 0.62~0.65로 모델이 사실상 무능. 이 데이터의 이론적 상한이 0.68 부근임을 확인.

→ **단순 ML 알고리즘 경쟁이 아닌, 임상 도메인 지식 통합과 데이터 표현의 재설계가 본질**.

---

## 2. 프로젝트 진행 — 10 Phase Narrative

```
Phase 1-3  ─ 베이스라인 구축
Phase 4    ─ SHAP/오답 진단으로 천장 원인 발견
Phase 5-7  ─ 26+ 카드 누적 시도 → 단변수 phenotype 천장 확인
Phase 8    ⭐ 차원 전환: 표현 변환 카드 발굴
Phase 9    ⭐ 컴포넌트 단독 교체 메커니즘
Phase 10   ⭐ Adversarial Validation으로 leakage 발견·제거
```

### Phase 1-3 — 베이스라인 구축

`LightGBM (Optuna best params) seed×3 + CatBoost + MLP`의 3-component 가중 블렌드로 단일 진입점 파이프라인을 구축. fold 내 StandardScaler fit으로 leakage를 방지하고, simplex 사영 기반 Nelder-Mead로 가중치를 최적화.

→ **v10 LB 0.74211** (이후 모든 사이클의 baseline)

### Phase 4 — SHAP / 오답 분석으로 천장 원인 진단

전체 AUC가 0.74 부근에서 정체되는 *진짜 원인*을 찾기 위해, 세그먼트별 OOF AUC와 FN/FP 프로필을 분석.

| 세그먼트 | n | 양성률 | AUC | Δ vs 전체 |
|---------|------|---------|------|----------|
| **Day5 이식** | 81,459 | 40.4% | **0.6196** | **-0.1212** 🔴 |
| eSET (단일배아 이식) | 58,383 | 36.7% | 0.6340 | -0.1068 |
| Frozen cycle | 40,126 | 22.9% | 0.6547 | -0.0861 |

**임상 해석**:
> Day5 / eSET / frozen은 IVF 실무상 "질 높은 시술"의 식별자. 이 세그먼트 내부에서 성공·실패를 가르는 결정 변수 — *자궁 수용성, 면역 호환성, 정자 DNA fragmentation* — 가 본 데이터에 부재. 본 데이터의 이론적 상한이 ~0.68 부근임을 확인.

→ **결론**: 단순한 단변수 추가로는 이 천장을 넘을 수 없다. *다른 차원의 카드*가 필요하다.

### Phase 5-7 — 26+ 카드 누적 시도와 천장 확인

체계적으로 시도한 카드 분류:

```
[A] 단변수 임상 phenotype : 18+ 카드 (Day5 storage interaction, has_prior_ivf_success,
                            POSEIDON 분류, TFF Stage, ICSI state, RIF/RPL 등)
[B] 인코딩 차원          : feature selection, frequency encoding
[C] 결측 처리            : MICE imputation 단독
[D] 모델 다양성          : XGBoost, HistGB, TabPFN, AutoGluon
[E] Hyperopt             : CatBoost / XGBoost Optuna 재탐색
[F] 블렌드               : Hill climb 9M~14M
[G] 표현학습             : DAE embedding, LogitBoost
```

**핵심 결론**:
- 단변수 phenotype을 추가해도 LightGBM이 이미 split으로 학습한 패턴이라 **추가 게인 0**.
- 모델 다양화(XGBoost/HistGB)도 GBT 동일 메커니즘이라 무력.
- Optuna 재탐색은 기존 best_params이 우연히 강한 안착점이라 오히려 음수.

→ **메타 인사이트**: "지금까지의 모든 카드는 같은 차원에서의 변형이었다. *다른 차원의 카드 = 표현 변환*이 필요하다."

### Phase 8 — ⭐ 차원 전환: 표현 변환 카드 발굴

산부인과 도메인 전문가의 deep brief와 외부 학술 논문 검증을 통해 두 핵심 카드를 발굴:

#### H2 — Donor Rejuvenation Index

donor egg 사용 시 환자 본인 나이가 아닌 **donor 나이**가 임신율을 결정한다는 임상 사실을 변수 표현으로 직접 인코딩.

```
effective_maternal_age = donor 사용 시 donor_age로 replacement
donor_quality_score    = Pearce 2019 기반 사전 정의 (Leakage-free)
rejuv_gap              = patient_age - donor_age
donor_age_mismatch     = 만43+ 환자 + 만20 이하 donor (위험 조합)
is_donor_optimal       = donor 만21-35 (literature optimal cohort)
```

**Pearce et al. 2019 (PubMed 31183626)** 외부 검증:
> *"Cycles using donors under 25 years were less likely to result in clinical pregnancy and live birth compared with cycles using donors 25 to under 30 years."*

본 데이터의 lift 패턴이 외부 논문 결론과 정확히 일치:

| 기증자 나이 | n | 양성률 | lift |
|------------|------|---------|------|
| 만20세 이하 | 294 | 26.2% | 1.014 |
| 만21-25세 | 2,334 | 33.0% | 1.277 |
| **만26-30세** | **4,976** | **34.8%** | **1.348** ⭐ |
| 만31-35세 | 6,366 | 30.5% | 1.181 |

→ Pearce 2019의 "26-30 prime cohort" 결론과 본 데이터 lift 정점이 정확 일치. H2 카드의 **임상적 정당성 + 데이터 효과** 동시 확보.

#### H9 — DI / IVF NaN Masking (segment-aware split)

DI cycle에서 IVF 전용 피처는 *시술 비적용*이므로 noise. 0으로 채우는 기존 처리는 LightGBM이 0을 일반 값으로 처리해 segment 정보를 잃게 한다. 이를 NaN으로 마스킹하면 LightGBM의 NaN-direction split이 자연스럽게 *segment-aware*로 학습.

```
변환 전: DI 행 IVF 컬럼 = 0  → segment 정보 손실
변환 후: DI 행 IVF 컬럼 = NaN → NaN-direction split이 자동 segment-aware 학습
```

대상 9개 IVF 전용 피처에 `_ivf_only` suffix를 부여:
*미세주입된 난자 수, 미세주입 배아 이식 수, 파트너 정자와 혼합된 난자 수, 혼합된 난자 수, 수집된 신선 난자 수, 저장된 신선 난자 수, icsi_fertilization_rate, embryo_transfer_pressure, storage_rate*.

#### H2 + H9 시너지

| 카드 | 단독 OOF Δ | 누적 OOF Δ |
|------|-----------|-----------|
| H2 단독 | +0.000044 | +0.000044 |
| H9 단독 | +0.000004 | +0.000048 (단순합) |
| **H2 + H9 누적** | — | **+0.000143 (3배 시너지)** |

→ **v59 LB 0.74227, v60 LB 0.74230** (26+ 카드 천장 첫 돌파, +0.00019 vs v10).

### Phase 9 — ⭐ 컴포넌트 단독 교체 메커니즘

전체 모델을 한 번에 강화하면 OOF는 오르지만 LB가 떨어지는 경우(과맞춤)를 여러 차례 관찰. 이를 회피하기 위해 *안정 강한 컴포넌트는 유지하고, 약한 컴포넌트만 단독 교체*하는 안전 패턴을 발견.

| 사이클 | 변경 | 결과 |
|--------|------|------|
| v60 → **v62b** | CAT 컴포넌트만 H2+H9 EXTRAS로 교체 | LB +0.00007 |
| v62b → **v62h_simple** | MLP 컴포넌트만 H2+H9 EXTRAS로 교체 | LB +0.0000246 |
| v62h_simple → **v62h_hillclimb** | Hill climb 가중치 재탐색 (grid step=0.005) | LB +0.0000031 |

**대조군**: 모든 컴포넌트를 동시에 강화한 v62c는 LB -0.0000291 (CV-LB 비례 음수), fold + 카드 동시 변경한 v62d 역시 -0.0000264. → *"단일 변수 변경 + fold 동일"이 안전 LB 게인 메커니즘*임을 검증.

### Phase 10 — ⭐ Adversarial Validation으로 leakage 발견·제거 (D-1)

마감 하루 전, train과 test의 분포 차이를 검증하기 위해 train+test를 합쳐 binary classifier(`label = 0 if train else 1`)를 학습. **5-fold AUC가 모두 1.000000** — 모델이 train/test를 100% 구분 가능.

원인: 기존 `features.py`의 interaction Target Encoding 5개가 train(fold-OOF)과 test(전체 단일 통계)에서 분포 자체가 달라, 모델이 그 "디지털 지문"으로 train/test를 식별.

수정:
1. test도 fold별 통계 평균으로 — *구조 대칭화*
2. Bayesian smoothing (prior=global mean, strength=20)
3. round_decimals=4로 디지털 지문 제거

CV-LB 비례 검증:

| 자산 | OOF Δ vs v62b | LB Δ vs v62b | 비례 |
|------|---------------|--------------|------|
| leakage 있음 | +0.000420 | +0.0000946 | **22.5%** |
| **leakage 제거 후** | +0.000317 | +0.0000458 | **47.4%** ⭐ 2배 개선 |

→ leakage 제거 자체로 LB 게인보다도, *모델의 일반화 신뢰도가 2배 향상*된 것이 더 큰 자산.

---

## 3. 핵심 포인트

### ① 임상-ML hybrid 접근

> ML 알고리즘 경쟁이 아니라 **도메인 지식 통합**이 본질이라는 인식 자체가 첫 차별점.

산부인과 reproductive-medicine-expert와의 4회 협업으로 단순 ML이 닿지 못하는 신호 — donor age paradox, DI/IVF 구조적 결측, fertility cliff segmentation — 를 변수 표현으로 직접 인코딩.

### ② 외부 학술 논문 4편 검증

| 논문 | 검증 내용 |
|------|----------|
| Pearce 2019 (PubMed 31183626) | Donor 만26-30 prime → 본 데이터 lift 1.348 정확 일치 |
| Humaidan 2021 POSEIDON (PMC8083858) | poor-responder 분류 표준 → phenotype 한계 검증 → 표현 변환으로 전환 |
| Polyzos 2015 Bologna POR (PMC4650906) | 만45-50 양성률 6.8% → Bologna POR 환자 비율 일치 |
| Nature Comm 2025 | Center-specific ML 우월성 → 시술 시기 코드 7그룹 활용 정당성 |

→ 본 솔루션의 모든 핵심 카드는 학술적 정당성과 데이터 패턴이 동시에 검증됨.

### ③ 표현 변환 메커니즘 발굴 — 가장 큰 메타 인사이트

> *"phenotype clustering ≠ 표현 변환"*

| 차원 | 정의 | 효과 |
|------|------|------|
| **Phenotype clustering** | LightGBM split으로 만들 수 있는 명시적 카테고리 | 모델이 이미 학습 → 추가 게인 0 |
| **표현 변환** | 변수의 의미·구조 자체를 바꾸기 (replacement, NaN-mask) | 모델이 자체 학습 불가 → 본질적 신호 추가 |

H2 (donor 나이 replacement)와 H9 (DI cycle NaN masking) 모두 LightGBM/CatBoost가 *기존 데이터로는 만들어낼 수 없는 변환*. 이것이 26+ 카드 천장을 넘은 본질.

### ④ Adversarial Validation 기반 leakage 검증

대회 마감 D-1에 발견한 가장 결정적 인사이트. CV가 LB와 비례하지 않는 근본 원인을 찾고, 일반화 능력을 2배 끌어올림.

---

## 4. 결과

### 6사이클 LB 진행

| 사이클 | 버전 | LB | Δ vs 직전 | Δ vs v10 | 핵심 변화 |
|--------|------|------|----------|---------|----------|
| 1 | v10 (베이스라인) | 0.74211 | — | 0 | LGBM seed×3 + CAT + MLP |
| 2 | v59 | 0.74227 | +0.00016 | +0.00016 | H2+H9 single seed 추가 |
| 3 | v60 | 0.74230 | +0.00003 | +0.00019 | H2+H9 seed×3 평균 |
| 4 | v62b | 0.74237 | +0.00007 | +0.00026 | CAT 컴포넌트 단독 교체 |
| 5 | v62h_simple | 0.7423962 | +0.0000246 | +0.000286 | MLP 컴포넌트 단독 교체 |
| **6** | **v62h_hillclimb** ⭐ | **0.7423993** | **+0.0000031** | **+0.000289** | **Hill climb 가중치 재탐색** |

### 최종 솔루션 — 사이클 6 챔피언

| 컴포넌트 | 가중치 | 5-fold OOF | 비고 |
|----------|--------|-----------|------|
| LGBM_h2h9_seedavg | 0.550 | 0.7407 | seed=[42, 2025, 7] 평균 |
| CatBoost_h2h9 | 0.265 | 0.7404 | 사이클 4에서 도입 |
| MLP_h2h9_seedavg | 0.185 | 0.7349 | 사이클 5에서 도입 |

**최종 블렌드 OOF AUC: 0.741044  /  Public LB: 0.7423993**

---

## 5. 메타 인사이트 — 이 프로젝트에서 배운 것

### Phenotype clustering ≠ 표현 변환

LightGBM split으로 자연스럽게 만들어지는 카테고리는 명시적으로 추가해도 모델이 이미 학습한 정보. 진짜 게인은 *모델이 자체 학습할 수 없는 표현 변환*에서 나온다.

### 단독 OOF 게인 ≠ 블렌드 게인

CAT seed×3로 단독 OOF +0.000154 게인을 얻었지만 블렌드에서는 음수. 컴포넌트의 OOF 향상이 블렌드 다양성을 깨뜨릴 수 있다는 점을 직접 검증.

### 컴포넌트 단독 교체 = 안전한 LB 게인 패턴

전체를 동시에 강화하면 OOF 과맞춤. 안정 강한 컴포넌트는 유지하고 약한 컴포넌트만 단독으로 교체하면, 검증된 메커니즘이 무너지지 않으면서 LB가 올라간다. 5번 연속 재현(v62b → v62h_simple → v62h_hillclimb까지)으로 메커니즘 확정.

### CV-LB 비례 검증의 중요성

CV가 오르면 LB가 오를 것이라는 직관은 leakage가 있을 때 깨진다. Adversarial Validation으로 train/test 분포 차이를 정량화하고, CV-LB 비례 계수를 추적하면 모델의 일반화 신뢰도를 객관 지표로 다룰 수 있다.

### Optuna 재탐색이 항상 좋은 것은 아니다

기존 best_params이 우연히 강한 안착점일 때, 새 데이터·새 카드와 함께 재탐색하면 오히려 음수가 나올 수 있다. 이미 검증된 안착점은 보존이 답.

---

## 6. 회고

### 잘한 점

1. **천장 원인을 추측이 아니라 진단으로 규명** — Phase 4 SHAP/오답 분석이 이후 모든 의사결정의 출발점.
2. **체계적 ablation 26+ 카드** — 막연한 시도가 아니라, 카드 분류와 누적 효과 추적으로 천장 메커니즘을 정확히 규명.
3. **임상-ML hybrid 사고 전환** — "ML 알고리즘 경쟁이 아니다"라는 인식이 표현 변환 카드 발굴의 토대.
4. **외부 학술 검증** — 4편 논문 인용으로 *임상적 정당성*과 *데이터 효과*를 동시 확보. 단순 데이터 fitting을 넘는 신뢰성.
5. **마감 D-1 Adversarial 발견** — 시간 압박 속에서도 일반화 신뢰도 검증의 근본 원인을 찾아낸 자세.

### 한계 및 개선 여지

- **자궁 수용성 변수 부재** — endometrial receptivity 같은 변수가 없어 Day5 세그먼트 AUC 0.62 천장이 본 데이터의 한계. 외부 데이터(가능 시)로 보완 가능.
- **center-specific 효과 미활용** — Nature Comm 2025의 center-specific ML 인사이트를 시술 시기 코드 7그룹 deep dive까지는 끌고 가지 못함.
- **상위 솔루션 정독 미흡** — 3위 솔루션만 정독, 1·2위 솔루션의 더 본질적 차별점 학습은 향후 과제.

### 시사점

> **AI 헬스케어 솔루션은 ML 알고리즘 경쟁이 아니라, *임상 도메인 지식 통합 + 외부 학술 검증 + 데이터 표현의 재설계*의 결합이다.**
>
> 본 데이터에서 LightGBM/CatBoost 단변수 천장은 ~0.741. 표현 변환 + 외부 논문 + 임상 phenotype의 결합이 +0.001~+0.003 게인을 만드는 본질이었다.

---

## 부록 A — 코드 구조

| 파일 | 사이클 | LB | 비고 |
|------|--------|------|------|
| `final_submission_260423.py` | 1 | 0.74211 | 베이스라인 (LGBM seed×3 + CAT + MLP 3M 블렌드) |
| `final_submission_260425.py` | 2-3 | 0.74230 | H2 + H9 표현 변환 카드 추가 |
| `final_submission_260428_v62b.py` | 4 | 0.74237 | CAT 컴포넌트 단독 교체 |
| `final_submission_260428_v62h.py` | 5-6 | **0.7423993** | MLP 단독 교체 + 가중치 재탐색 (챔피언) |

각 파일은 self-contained 단일 .py로, `data/` 디렉토리에 `train.csv`, `test.csv`, `submission/sample_submission.csv` 위치 시 단독 실행 가능.

## 부록 B — 환경

```
pandas>=2.2,<3
numpy>=1.26,<3
scipy>=1.13
scikit-learn>=1.4
lightgbm>=4.5
catboost>=1.2
```

```bash
pip install -r requirements.txt
python final_submission_260428_v62h.py    # 챔피언 사이클 6 재현
```

`SEED=42`, `SEED_LIST=[42, 2025, 7]`, `StratifiedKFold(5, shuffle=True, random_state=seed)` 일괄 고정. fold 내 fit / leakage-free.

## 부록 C — 인용 논문

| # | 인용 | Link |
|---|------|------|
| 1 | Pearce et al. 2019 — Donor age <25 not better (H2 외부 검증) | [PubMed 31183626](https://pubmed.ncbi.nlm.nih.gov/31183626/) |
| 2 | Humaidan et al. 2021 — POSEIDON criteria | [PMC8083858](https://pmc.ncbi.nlm.nih.gov/articles/PMC8083858/) |
| 3 | Polyzos et al. 2015 — Bologna POR critical appraisal | [PMC4650906](https://pmc.ncbi.nlm.nih.gov/articles/PMC4650906/) |
| 4 | Nature Communications 2025 — Center-specific ML > National | [nature.com/articles/s41467-025-58744-z](https://www.nature.com/articles/s41467-025-58744-z) |

---

## 데이터 사용 안내

대회 데이터(`data/`, `submission/`)와 발표 자료(`*.pdf`, `*.pptx`)는 외부 공유 금지로 `.gitignore`에 의해 제외. 본 저장소는 코드와 README만 공개.
