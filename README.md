# Pregnancy Predict

> 난임 환자 대상 임신 성공 여부 예측 AI 모델
> 오즈코딩스쿨 헬스케어AI 3기 × 데이콘 해커톤

## Overview

| 항목 | 내용 |
|------|------|
| Task | Binary classification (probability) |
| Metric | ROC-AUC |
| Data | 정형 데이터 (256K train / 90K test, 68 features) |
| Approach | GBDT 우선 트랙 (CatBoost / LightGBM / XGBoost), DL 병행 검토 |

## Repository Policy

본 저장소는 **최종 정리된 코드**만 추적합니다.
대회 데이터, 작업용 노트북, 모델 바이너리, 실험 산출물 등은 로컬에서만 관리되며 git에 포함되지 않습니다 (`.gitignore` 참조).

## Project Structure (planned)

```
src/                  # 최종 파이프라인 코드 (.py)
  ├── config.py       # 시드, 경로, 하이퍼파라미터
  ├── cv.py           # CV 분할 / OOF
  ├── metrics.py      # 평가 함수
  ├── preprocess.py   # 전처리·인코딩
  ├── features.py     # 피처 엔지니어링
  ├── train.py        # 학습 진입점
  └── predict.py      # 추론 진입점
```

## Reproducibility

- Python 3.10 (conda env: `torch_env`)
- Random seed: 42 (모든 실행에서 고정)
- 라이브러리 버전은 코드 제출 시 별도 명시

## Author

[jun8741-crypto](https://github.com/jun8741-crypto)
