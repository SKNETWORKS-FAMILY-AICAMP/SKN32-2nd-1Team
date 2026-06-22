from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
import numpy as np

TELECOM_MODEL_PATH = Path("models/xgb_model.joblib")

# 모델 학습 시 사용된 피처 순서 (순서가 틀리면 예측이 깨집니다)
FEATURE_ORDER = [
    "a02014", "a03008", "age", "gender", "income",
    "school", "area", "hhldsiz", "job1", "mar",
    "c01001", "c01003", "c02001",
]

_TELECOM_CHURN_MODEL = None


def load_telecom_churn_model():
    global _TELECOM_CHURN_MODEL

    if _TELECOM_CHURN_MODEL is not None:
        return _TELECOM_CHURN_MODEL

    if not TELECOM_MODEL_PATH.exists():
        raise FileNotFoundError("models/xgb_model.joblib 파일이 없습니다.")

    _TELECOM_CHURN_MODEL = joblib.load(TELECOM_MODEL_PATH)
    return _TELECOM_CHURN_MODEL


def make_input_dataframe(values: Dict[str, object]) -> pd.DataFrame:
    missing = [f for f in FEATURE_ORDER if f not in values]
    if missing:
        raise ValueError(f"다음 입력값이 누락되었습니다: {missing}")

    return pd.DataFrame([values])[FEATURE_ORDER]


def predict_churn(values: Dict[str, object]) -> Dict[str, object]:
    model = load_telecom_churn_model()
    input_df = make_input_dataframe(values)

    probabilities = model.predict_proba(input_df)[0]
    churn_probability = float(probabilities[1])
    prediction = int(churn_probability >= 0.5)
    label = "이탈 위험" if prediction == 1 else "잔류 가능성 높음"


    return {
        "prediction": prediction,
        "label": label,
        "churn_probability": churn_probability,
        "retention_probability": float(probabilities[0]),
    }


# -------------------------------------------------------
# xgb_pipeline.joblib 관련
# -------------------------------------------------------

PIPELINE_MODEL_PATH = Path("models/xgb_pipeline.joblib")

PIPELINE_NUMERIC_FEATURES = [
    "year", "age", "income", "monthly_total_cost", "monthly_installment",
    "household_size", "installment_ratio", "cost_income_ratio",
    "income_per_person", "cost_per_person",
]
PIPELINE_CATEGORICAL_FEATURES = [
    "gender", "school", "area", "job", "marriage",
    "cost_payer", "provider", "married_large_family", "is_mobile_bundled",
]
PIPELINE_FEATURE_COLS = PIPELINE_NUMERIC_FEATURES + PIPELINE_CATEGORICAL_FEATURES

_PIPELINE_MODEL = None


def load_pipeline_model():
    global _PIPELINE_MODEL

    if _PIPELINE_MODEL is not None:
        return _PIPELINE_MODEL

    if not PIPELINE_MODEL_PATH.exists():
        raise FileNotFoundError("models/xgb_pipeline.joblib 파일이 없습니다.")

    _PIPELINE_MODEL = joblib.load(PIPELINE_MODEL_PATH)
    return _PIPELINE_MODEL


def make_pipeline_input_dataframe(values: Dict[str, object]) -> pd.DataFrame:
    df = pd.DataFrame([values])

    df["installment_ratio"] = df["monthly_installment"] / (df["monthly_total_cost"] + 1)
    df["cost_income_ratio"] = df["monthly_total_cost"] / (df["income"] + 1)
    df["income_per_person"] = df["income"] / df["household_size"]
    df["cost_per_person"] = df["monthly_total_cost"] / df["household_size"]
    df["married_large_family"] = (
        (df["marriage"] == 2) & (df["household_size"] >= 3)
    ).astype(int)

    return df[PIPELINE_FEATURE_COLS]


def predict_churn_pipeline(values: Dict[str, object]) -> Dict[str, object]:
    model = load_pipeline_model()
    input_df = make_pipeline_input_dataframe(values)

    probabilities = model.predict_proba(input_df)[0]
    churn_probability = float(probabilities[1])
    prediction = int(churn_probability >= 0.5)
    label = "이탈 위험" if prediction == 1 else "잔류 가능성 높음"

    return {
        "prediction": prediction,
        "label": label,
        "churn_probability": churn_probability,
        "retention_probability": float(probabilities[0]),
    }


# -------------------------------------------------------
# 다수 고객 배치 예측 (CSV / XLSX 업로드용)
# -------------------------------------------------------

# 업로드 파일에 반드시 있어야 하는 원본 입력 컬럼(파생변수 제외).
# 화면 단일 예측에서 input_values로 넘기던 키와 동일합니다.
PIPELINE_RAW_INPUT_COLS = [
    "year", "age", "gender", "income", "school", "area",
    "household_size", "job", "marriage", "provider",
    "monthly_total_cost", "monthly_installment", "cost_payer",
    "is_mobile_bundled",
]


def make_pipeline_input_dataframe_batch(df: pd.DataFrame) -> pd.DataFrame:
    """여러 행이 든 DataFrame에 파생변수를 벡터 연산으로 한 번에 계산합니다.

    단일 예측용 make_pipeline_input_dataframe와 계산식이 동일하며,
    행 단위 반복 대신 컬럼 전체 연산으로 처리합니다.
    """
    df = df.copy()

    df["installment_ratio"] = df["monthly_installment"] / (df["monthly_total_cost"] + 1)
    df["cost_income_ratio"] = df["monthly_total_cost"] / (df["income"] + 1)
    df["income_per_person"] = df["income"] / df["household_size"]
    df["cost_per_person"] = df["monthly_total_cost"] / df["household_size"]
    df["married_large_family"] = (
        (df["marriage"] == 2) & (df["household_size"] >= 3)
    ).astype(int)

    return df[PIPELINE_FEATURE_COLS]


def predict_churn_pipeline_batch(raw_df: pd.DataFrame) -> pd.DataFrame:
    """원본 입력 컬럼이 든 DataFrame을 받아 행마다 이탈 예측을 수행합니다.

    반환값은 입력 원본 컬럼 + 다음 3개 컬럼이 추가된 DataFrame입니다.
        - churn_probability: 이탈 확률 (0~1)
        - retention_probability: 잔류 확률 (0~1)
        - prediction: 이탈 여부 (1=이탈 위험, 0=잔류)
    """
    missing = [c for c in PIPELINE_RAW_INPUT_COLS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"업로드 파일에 다음 컬럼이 없습니다: {missing}")

    if len(raw_df) == 0:
        raise ValueError("업로드 파일에 데이터 행이 없습니다.")

    model = load_pipeline_model()
    input_df = make_pipeline_input_dataframe_batch(raw_df)

    probabilities = model.predict_proba(input_df)
    churn_probs = probabilities[:, 1]

    result = raw_df.copy().reset_index(drop=True)
    result["churn_probability"] = churn_probs
    result["retention_probability"] = probabilities[:, 0]
    result["prediction"] = (churn_probs >= 0.5).astype(int)

    return result