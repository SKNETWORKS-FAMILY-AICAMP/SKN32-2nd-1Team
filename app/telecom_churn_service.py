"""통신사 고객 이탈 예측 서비스 모듈입니다.

⚠️⚠️⚠️ 매우 중요 — 절대 임의로 수정하지 마세요 ⚠️⚠️⚠️

이 파일의 FeatureEngineer 클래스는 모델 학습 코드(churn_modeling.ipynb)에
정의된 것과 글자 하나까지 동일해야 합니다. 학습 시 모델이 본 입력 변환 방식과
여기서 다르게 계산하면, 모델이 학습한 적 없는 패턴의 입력이 들어가서
예측 확률이 미묘하게(또는 크게) 틀어집니다.

검증된 입력 컬럼 범위 (extracted_data.csv, 147,915행 직접 확인 결과):
    age                : 1~8 코드값
    income             : 1~8 코드값 (1=소득없음 ~ 8=500만원 이상)
    school             : 1~6 코드값 (0=무학, 9999=모름/무응답 없음 — 학습 데이터에 미존재)
    provider           : 1~5 코드값 (9999=모름/무응답 없음 — 학습 데이터에 미존재)
    household_size     : 1~3 코드값
    is_mobile_bundled  : 1=예, 2=아니오 (0 없음)
    monthly_total_cost : 천원 단위 (예: 70 = 7만원)
    monthly_installment: 천원 단위 (예: 30 = 3만원)
    area               : 사용하지 않음 (모델 입력에서 제외됨)

이 범위를 바꾸기 전에는 반드시 extracted_data.csv를 다시 검증하세요.
"""

import os
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin


# ============================================================
# FeatureEngineer
# 학습 코드(churn_modeling.ipynb, CELL 2)와 동일한 로직입니다.
# joblib.load가 이 클래스를 찾을 수 있도록 모델을 로드하기 전에
# 반드시 이 클래스가 먼저 정의(import)되어 있어야 합니다.
# ============================================================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # 패널 피처 계산 전 정렬 (id별 시간 순서가 중요합니다)
        df = df.sort_values(["id", "year"]).reset_index(drop=True)

        # =========================
        # 기본 비율/차이 피처
        # =========================
        df["income_zero_flag"] = np.where(
            df["income"].notna(),
            (df["income"] == 1).astype("Int64"),
            pd.NA,
        )

        df["installment_ratio"] = (
            df["monthly_installment"] / (df["monthly_total_cost"] + self.eps)
        )

        # income 코드값(1~8)을 실제 원화 소득으로 환산합니다.
        income_proxy = df["income"] * 500000 - 750000

        df["cost_income_ratio"] = np.where(
            df["income_zero_flag"] == 0,
            df["monthly_total_cost"] * 1000 / income_proxy,
            np.nan,
        )

        df["installment_income_ratio"] = np.where(
            df["income_zero_flag"] == 0,
            df["monthly_installment"] * 1000 / income_proxy,
            np.nan,
        )

        df["income_per_person"] = income_proxy.clip(lower=0) / df["household_size"]
        df["cost_per_person"] = df["monthly_total_cost"] * 1000 / df["household_size"]
        df["installment_per_person"] = (
            df["monthly_installment"] * 1000 / df["household_size"]
        )
        df["service_cost"] = df["monthly_total_cost"] - df["monthly_installment"]

        df["disposable_income_proxy"] = np.where(
            df["income_zero_flag"] == 0,
            income_proxy - df["monthly_total_cost"] * 1000,
            np.nan,
        )

        df["income_remaining_ratio"] = df["disposable_income_proxy"] / income_proxy
        df["service_cost_ratio"] = df["service_cost"] / (df["monthly_total_cost"] + self.eps)

        df["married_large_family"] = (
            (df["marriage"] == 2) & (df["household_size"] >= 3)
        ).astype(int)

        df["income_log"] = np.log1p(df["income"])
        df["monthly_total_cost_log"] = np.log1p(df["monthly_total_cost"])
        df["monthly_installment_log"] = np.log1p(df["monthly_installment"])

        # =========================
        # 패널 데이터 기반 피처 (id + year 순 정렬 후 사용)
        # =========================
        df["is_first_observation"] = (
            df.groupby("id")["year"].shift(1).isna().astype(int)
        )

        df["prev_income"] = df.groupby("id")["income"].shift(1)
        df["prev_monthly_total_cost"] = df.groupby("id")["monthly_total_cost"].shift(1)
        df["prev_monthly_installment"] = df.groupby("id")["monthly_installment"].shift(1)
        df["prev_provider"] = df.groupby("id")["provider"].shift(1)
        df["prev_provider"] = df["prev_provider"].fillna(df["provider"])

        df["income_change"] = df["income"] - df["prev_income"]
        df["cost_change"] = df["monthly_total_cost"] - df["prev_monthly_total_cost"]
        df["installment_change"] = (
            df["monthly_installment"] - df["prev_monthly_installment"]
        )

        df["income_change_rate"] = df["income_change"] / (df["prev_income"] + self.eps)
        df["cost_change_rate"] = (
            df["cost_change"] / (df["prev_monthly_total_cost"] + self.eps)
        )
        df["installment_change_rate"] = (
            df["installment_change"] / (df["prev_monthly_installment"] + self.eps)
        )

        df["provider_changed"] = np.where(
            df["prev_provider"].isna(),
            0,
            (df["provider"] != df["prev_provider"]).astype(int),
        )

        df["cost_jump_flag"] = np.where(
            df["is_first_observation"] == 1,
            0,
            (df["cost_change_rate"] >= 0.2).astype(int),
        )

        df["installment_jump_flag"] = np.where(
            df["is_first_observation"] == 1,
            0,
            (df["installment_change_rate"] >= 0.2).astype(int),
        )

        # =========================
        # 이동통계 피처
        # =========================
        df["cost_roll_mean_3"] = df.groupby("id")["monthly_total_cost"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        df["cost_roll_std_3"] = df.groupby("id")["monthly_total_cost"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=2).std()
        )
        df["income_roll_mean_3"] = df.groupby("id")["income"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        df["income_roll_std_3"] = df.groupby("id")["income"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=2).mean()
        )

        df["cost_above_history_mean"] = np.where(
            df["cost_roll_mean_3"].isna(),
            0,
            (df["monthly_total_cost"] > df["cost_roll_mean_3"]).astype(int),
        )

        df["income_zero_flag"] = df["income_zero_flag"].fillna(1).astype(int)

        return df


# ============================================================
# 모델 입력 컬럼 정의
# area는 더 이상 사용하지 않습니다 (모델 입력에서 제외됨).
# id는 모델 ColumnTransformer 계산에 실제로 쓰이지 않지만(remainder로 통과),
# FeatureEngineer.transform 안에서 groupby('id')에 필요하므로 입력에 포함합니다.
# ============================================================
PIPELINE_RAW_INPUT_COLS = [
    "id",
    "year",
    "age",
    "gender",
    "income",
    "school",
    "job",
    "marriage",
    "monthly_total_cost",
    "monthly_installment",
    "cost_payer",
    "provider",
    "household_size",
    "is_mobile_bundled",
]

# 신규(단일) 예측 시 패널 이력이 없는 고객을 구분하기 위한 고정 id입니다.
# 실제 모델 계산에는 영향이 없습니다(remainder 컬럼이라 그냥 통과됩니다).
SINGLE_PREDICT_FIXED_ID = 99999999


# ============================================================
# 모델 로드
# ============================================================
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "xgb_full_pipeline.joblib",
)

_model = None


def _get_model():
    global _model
    if _model is None:
        # app.py(또는 streamlit이 실행하는 스크립트)가 __main__ 모듈이 되므로,
        # joblib.load가 호출되기 바로 직전에 그 __main__ 모듈에 FeatureEngineer를
        # 등록합니다. 모듈 import 시점에만 등록하면 실행 환경에 따라
        # __main__이 아직 이 클래스를 못 보는 경우가 있어, 매번 로드 직전에
        # 다시 확실히 등록합니다.
        sys.modules["__main__"].FeatureEngineer = FeatureEngineer
        _model = joblib.load(_MODEL_PATH)
    return _model


def predict_churn_pipeline(input_values: dict) -> dict:
    """단일 고객 정보를 입력받아 이탈 확률을 예측합니다.

    input_values는 PIPELINE_RAW_INPUT_COLS 중 "id"를 제외한 나머지 키를
    포함해야 합니다. id가 없으면 SINGLE_PREDICT_FIXED_ID로 채웁니다.
    """
    model = _get_model()

    row = dict(input_values)
    row.setdefault("id", SINGLE_PREDICT_FIXED_ID)

    df = pd.DataFrame([row], columns=PIPELINE_RAW_INPUT_COLS)

    proba = model.predict_proba(df)[:, 1][0]
    pred = int(proba >= 0.5)

    return {
        "churn_probability": float(proba),
        "retention_probability": float(1 - proba),
        "prediction": pred,
    }


def predict_churn_pipeline_batch(raw_df: pd.DataFrame) -> pd.DataFrame:
    """다수 고객 데이터(DataFrame)를 입력받아 일괄로 이탈을 예측합니다.

    raw_df는 PIPELINE_RAW_INPUT_COLS 컬럼(또는 id를 제외한 나머지)을
    포함해야 합니다. id 컬럼이 없으면 고정값으로 채웁니다.
    """
    model = _get_model()

    df = raw_df.copy()
    if "id" not in df.columns:
        df["id"] = SINGLE_PREDICT_FIXED_ID

    df = df[PIPELINE_RAW_INPUT_COLS]

    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)

    result_df = raw_df.copy()
    result_df["churn_probability"] = proba
    result_df["retention_probability"] = 1 - proba
    result_df["prediction"] = pred

    return result_df