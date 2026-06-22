"""이탈(통신사 변경) 시 다음 통신사를 예측하는 서비스 모듈입니다.

⚠️⚠️⚠️ 매우 중요 — 절대 임의로 수정하지 마세요 ⚠️⚠️⚠️

이 모델(next_lgb_churn_model.joblib)은 통신사 이탈 여부(0/1)를 맞추는
모델(xgb_full_pipeline.joblib)과는 완전히 다른 별개의 모델입니다.

이 모델은 "이미 통신사를 변경하기로 한(또는 변경한) 고객이 SKT/KT/LG U+
중 어디로 이동할지"를 맞추는 다중분류(3-class) 모델입니다. (학습 코드:
next_churn_v3.py 참고)

학습 시 사용한 입력 피처는 아래 17개이며, 이 순서와 이름이 정확히
일치해야 합니다 (model.feature_name() 으로 직접 확인한 순서):
    age, gender, income, school, area, job, marriage,
    monthly_total_cost, monthly_installment, cost_payer, provider,
    household_size, is_mobile_bundled,
    provider_changed, total_changes, cost_change_rate, tenure

이 중 area는 화면에서 사용자에게 입력받지 않기로 결정되어
AREA_FIXED_VALUE로 고정값을 채워 넘깁니다. 변수 중요도 분석 결과
area는 영향력이 낮은 변수로 확인되었습니다 (data.md 학습 결과 참고).

provider_changed, total_changes, cost_change_rate, tenure는 원래
"여러 해에 걸친 이력"이 있어야 정확히 계산되는 값입니다. 단일 입력
화면에서는 아래 단순화된 가정으로 대체합니다:
    - provider_changed = 1 (이번에 통신사를 변경한다고 가정하고 예측하는 화면이므로)
    - total_changes    = 1 (사용자가 직접 입력, 과거 누적 변경 횟수)
    - cost_change_rate = 사용자가 입력한 "이전 통신비"와 현재 통신비로 계산
    - tenure           = 사용자가 직접 입력 (가입 후 관측된 연수)
"""

import os
import sys

import numpy as np
import pandas as pd
import joblib


# 화면에서 입력받지 않는 area의 고정값입니다.
# 변수 중요도 분석 결과 영향력이 낮아 임의의 고정값(최빈값 가정)을 사용합니다.
AREA_FIXED_VALUE = 1

PROVIDER_MAP = {1: "SKT", 2: "KT", 3: "LG U+"}

# 모델 입력 피처 순서 (model.feature_name()과 정확히 일치해야 함)
NEXT_PROVIDER_FEATURE_ORDER = [
    "age", "gender", "income", "school", "area", "job", "marriage",
    "monthly_total_cost", "monthly_installment", "cost_payer", "provider",
    "household_size", "is_mobile_bundled",
    "provider_changed", "total_changes", "cost_change_rate", "tenure",
]

# 화면에서 사용자가 직접 입력하는 컬럼 (area, provider_changed 제외)
USER_INPUT_COLS = [
    "age", "gender", "income", "school", "job", "marriage",
    "monthly_total_cost", "monthly_installment", "cost_payer", "provider",
    "household_size", "is_mobile_bundled",
    "total_changes", "prev_monthly_total_cost", "tenure",
]

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
)
_MODEL_PATH = os.path.join(_MODELS_DIR, "next_lgb_churn_model.joblib")
_ENCODER_PATH = os.path.join(_MODELS_DIR, "next_label_encoder.joblib")

_model = None
_label_encoder = None


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(_MODEL_PATH)
    return _model


def _get_label_encoder():
    global _label_encoder
    if _label_encoder is None:
        _label_encoder = joblib.load(_ENCODER_PATH)
    return _label_encoder


def predict_next_provider(input_values: dict) -> dict:
    """이미 통신사를 변경하기로 한 고객이 SKT/KT/LG U+ 중 어디로 이동할지 예측합니다.

    input_values 키:
        age, gender, income, school, job, marriage,
        monthly_total_cost, monthly_installment, cost_payer, provider,
        household_size, is_mobile_bundled,
        total_changes, prev_monthly_total_cost, tenure

    반환값: {"SKT": 확률, "KT": 확률, "LG U+": 확률} (현재 통신사는 0% 처리)
    """
    model = _get_model()
    le = _get_label_encoder()

    row = dict(input_values)

    # area는 화면에서 받지 않으므로 고정값으로 채웁니다.
    row["area"] = AREA_FIXED_VALUE

    # 이 화면은 "지금 통신사를 변경한다면"을 가정하는 예측이므로 1로 고정합니다.
    row["provider_changed"] = 1

    # 전년 대비 통신비 변화율 = (현재 - 이전) / (이전 + 1)
    prev_cost = row.pop("prev_monthly_total_cost", row["monthly_total_cost"])
    row["cost_change_rate"] = (
        (row["monthly_total_cost"] - prev_cost) / (prev_cost + 1)
    )

    df = pd.DataFrame([row], columns=NEXT_PROVIDER_FEATURE_ORDER)

    proba = model.predict(df)[0]
    target_names = [PROVIDER_MAP[int(c)] for c in le.classes_]

    result = {name: float(p) for name, p in zip(target_names, proba)}

    # 현재 통신사로는 "이동"이 아니므로 0%로 표기합니다.
    current_provider_name = PROVIDER_MAP.get(int(row["provider"]))
    if current_provider_name in result:
        result[current_provider_name] = 0.0

    return result