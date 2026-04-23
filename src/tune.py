import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import optuna
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, SEED, N_SPLITS
from preprocess import preprocess

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 전처리는 한 번만 실행
print("전처리 중...")
train, test, target, sub = preprocess(save=False)
print("전처리 완료!")


def objective(trial):
    params = {
        "objective":         "binary",
        "metric":            "auc",
        "verbose":           -1,
        "random_state":      SEED,
        "n_jobs":            -1,
        # 튜닝 대상 파라미터
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
        "max_depth":         trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":      trial.suggest_int("bagging_freq", 1, 10),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }

    skf       = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(train))

    for tr_idx, val_idx in skf.split(train, target):
        X_tr,  X_val = train.iloc[tr_idx], train.iloc[val_idx]
        y_tr,  y_val = target[tr_idx],     target[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(False),
            ],
        )
        oof_preds[val_idx] = model.predict(X_val)

    return roc_auc_score(target, oof_preds)


def main():
    print("=" * 45)
    print("  Optuna 하이퍼파라미터 튜닝 시작 (50 trials)")
    print("=" * 45)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(f"\n최적 AUC : {study.best_value:.4f}")
    print(f"최적 파라미터:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 최적 파라미터 저장
    import json
    best_params_path = os.path.join(OUTPUT_DIR, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\n저장 완료 → {best_params_path}")


if __name__ == "__main__":
    main()