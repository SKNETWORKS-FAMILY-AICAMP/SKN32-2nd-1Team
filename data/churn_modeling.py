import warnings
warnings.filterwarnings('ignore')

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.utils.class_weight import compute_sample_weight


# =====================================================
# 1. 데이터 로드
# =====================================================
df = pd.read_csv('extracted_data.csv')
df = df[df['churn'].notna()].copy()

print('전체 데이터 수:', len(df))


# =====================================================
# 2. 파생 변수 생성 Transformer
# =====================================================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, eps=1e-6):
        self.eps = eps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        eps = self.eps

        required_cols = [
            'id', 'year', 'age', 'gender', 'income', 'school', 'household_size',
            'job', 'marriage', 'provider', 'monthly_total_cost',
            'monthly_installment', 'cost_payer', 'is_mobile_bundled'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"필수 컬럼 누락: {missing_cols}")

        df = df.sort_values(['id', 'year']).reset_index(drop=True)

        # ----------------------------------
        # 기본 비율/차이 피처
        # ----------------------------------
        df['income_zero_flag'] = np.where(
            df['income'].notna(),
            (df['income'] == 1).astype('Int64'),
            pd.NA
        )

        income_proxy = df['income'] * 500000 - 750000
        household_safe = df['household_size'].clip(lower=1)

        df['installment_ratio'] = (
            df['monthly_installment'] / (df['monthly_total_cost'] + eps)
        )

        df['cost_income_ratio'] = np.where(
            df['income_zero_flag'] == 0,
            df['monthly_total_cost'] * 1000 / income_proxy,
            np.nan
        )

        df['installment_income_ratio'] = np.where(
            df['income_zero_flag'] == 0,
            df['monthly_installment'] * 1000 / income_proxy,
            np.nan
        )

        df['income_per_person'] = (
            income_proxy.clip(lower=0) / household_safe
        )

        df['cost_per_person'] = (
            df['monthly_total_cost'] * 1000 / household_safe
        )

        df['installment_per_person'] = (
            df['monthly_installment'] * 1000 / household_safe
        )

        df['service_cost'] = (
            df['monthly_total_cost'] - df['monthly_installment']
        )

        df['disposable_income_proxy'] = np.where(
            df['income_zero_flag'] == 0,
            income_proxy - df['monthly_total_cost'] * 1000,
            np.nan
        )

        df['income_remaining_ratio'] = np.where(
            df['income_zero_flag'] == 0,
            df['disposable_income_proxy'] / income_proxy,
            np.nan
        )

        df['service_cost_ratio'] = (
            df['service_cost'] / (df['monthly_total_cost'] + eps)
        )

        df['married_large_family'] = (
            ((df['marriage'] == 2) & (df['household_size'] >= 3)).astype(int)
        )

        df['income_log'] = np.log1p(df['income'])
        df['monthly_total_cost_log'] = np.log1p(df['monthly_total_cost'])
        df['monthly_installment_log'] = np.log1p(df['monthly_installment'])

        # ----------------------------------
        # 패널 데이터 기반 피처
        # ----------------------------------
        df['is_first_observation'] = (
            df.groupby('id')['year'].shift(1).isna().astype(int)
        )

        df['prev_income'] = df.groupby('id')['income'].shift(1)
        df['prev_monthly_total_cost'] = df.groupby('id')['monthly_total_cost'].shift(1)
        df['prev_monthly_installment'] = df.groupby('id')['monthly_installment'].shift(1)
        df['prev_provider_raw'] = df.groupby('id')['provider'].shift(1)

        df['income_change'] = df['income'] - df['prev_income']
        df['cost_change'] = df['monthly_total_cost'] - df['prev_monthly_total_cost']
        df['installment_change'] = df['monthly_installment'] - df['prev_monthly_installment']

        df['income_change_rate'] = df['income_change'] / (df['prev_income'] + eps)
        df['cost_change_rate'] = df['cost_change'] / (df['prev_monthly_total_cost'] + eps)
        df['installment_change_rate'] = (
            df['installment_change'] / (df['prev_monthly_installment'] + eps)
        )

        df['provider_changed'] = np.where(
            df['is_first_observation'] == 1,
            0,
            (df['provider'] != df['prev_provider_raw']).astype(int)
        )

        df['prev_provider'] = df['prev_provider_raw'].fillna(df['provider'])

        df['cost_jump_flag'] = np.where(
            df['is_first_observation'] == 1,
            0,
            (df['cost_change_rate'] >= 0.2).astype(int)
        )

        df['installment_jump_flag'] = np.where(
            df['is_first_observation'] == 1,
            0,
            (df['installment_change_rate'] >= 0.2).astype(int)
        )

        # ----------------------------------
        # 이동통계 피처
        # ----------------------------------
        df['cost_roll_mean_3'] = df.groupby('id')['monthly_total_cost'].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )

        df['cost_roll_std_3'] = df.groupby('id')['monthly_total_cost'].transform(
            lambda s: s.shift(1).rolling(3, min_periods=2).std()
        )

        df['income_roll_mean_3'] = df.groupby('id')['income'].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )

        df['income_roll_std_3'] = df.groupby('id')['income'].transform(
            lambda s: s.shift(1).rolling(3, min_periods=2).std()
        )

        df['cost_above_history_mean'] = np.where(
            df['cost_roll_mean_3'].isna(),
            0,
            (df['monthly_total_cost'] > df['cost_roll_mean_3']).astype(int)
        )

        df['income_zero_flag'] = df['income_zero_flag'].fillna(1).astype(int)

        return df


# =====================================================
# 3. 사용할 피처 목록
# =====================================================
numeric_features = [
    'year',
    'age',
    'income',
    'monthly_total_cost',
    'monthly_installment',
    'household_size',
    'installment_ratio',
    'cost_income_ratio',
    'installment_income_ratio',
    'income_per_person',
    'cost_per_person',
    'installment_per_person',
    'service_cost',
    'disposable_income_proxy',
    'income_remaining_ratio',
    'service_cost_ratio',
    'income_log',
    'monthly_total_cost_log',
    'monthly_installment_log',
    'prev_income',
    'prev_monthly_total_cost',
    'prev_monthly_installment',
    'income_change',
    'cost_change',
    'installment_change',
    'income_change_rate',
    'cost_change_rate',
    'installment_change_rate',
    'cost_roll_mean_3',
    'cost_roll_std_3',
    'income_roll_mean_3',
    'income_roll_std_3',
]

categorical_features = [
    'gender',
    'school',
    'job',
    'marriage',
    'cost_payer',
    'provider',
    'married_large_family',
    'is_mobile_bundled',
    'income_zero_flag',
    'is_first_observation',
    'prev_provider',
    'provider_changed',
    'cost_jump_flag',
    'installment_jump_flag',
    'cost_above_history_mean',
]

FEATURE_COLS = numeric_features + categorical_features


# =====================================================
# 4. 전처리기 정의
# =====================================================
numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# =====================================================
# 5. 평가 함수
# =====================================================
def evaluate_pipeline(model_pipeline, X, y, dataset_name):
    y_pred = model_pipeline.predict(X)

    if hasattr(model_pipeline, 'predict_proba'):
        y_prob = model_pipeline.predict_proba(X)[:, 1]
    else:
        y_prob = y_pred

    print(f'\n[{dataset_name}]')
    print('Accuracy :', round(accuracy_score(y, y_pred), 4))
    print('Precision:', round(precision_score(y, y_pred, zero_division=0), 4))
    print('Recall   :', round(recall_score(y, y_pred, zero_division=0), 4))
    print('F1-score :', round(f1_score(y, y_pred, zero_division=0), 4))
    print('ROC-AUC  :', round(roc_auc_score(y, y_prob), 4))


def get_feature_importance_from_pipeline(model_pipeline, top_n=20):
    model = model_pipeline.named_steps['model']
    fitted_preprocessor = model_pipeline.named_steps['preprocessing']
    feature_names = fitted_preprocessor.get_feature_names_out()

    if not hasattr(model, 'feature_importances_'):
        print('이 모델은 feature_importances_ 속성이 없습니다.')
        return None

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    print(importance_df.head(top_n))
    return importance_df


# =====================================================
# 6. 하이퍼파라미터
# =====================================================
LR_C = 1.0
LR_MAX_ITER = 1000

RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_LEAF = 5

GB_N_ESTIMATORS = 300
GB_LEARNING_RATE = 0.05
GB_SUBSAMPLE = 0.8

LGB_N_ESTIMATORS = 1000
LGB_LEARNING_RATE = 0.02
LGB_NUM_LEAVES = 63
LGB_MIN_CHILD_SAMPLES = 100
LGB_SUBSAMPLE = 0.8
LGB_COLSAMPLE = 0.8
LGB_REG_ALPHA = 1.0
LGB_REG_LAMBDA = 1.0

XGB_N_ESTIMATORS = 1000
XGB_LEARNING_RATE = 0.03
XGB_MAX_DEPTH = 4
XGB_MIN_CHILD_WEIGHT = 10
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8
XGB_GAMMA = 1
XGB_REG_ALPHA = 1
XGB_REG_LAMBDA = 2


# =====================================================
# 7. 데이터 분할
# =====================================================
train_df = df[df['year'] <= 20].copy()
valid_df = df[(df['year'] >= 21) & (df['year'] <= 22)].copy()
test_df = df[df['year'] >= 23].copy()

print('\n데이터 분할 결과')
print('train:', len(train_df))
print('valid:', len(valid_df))
print('test :', len(test_df))

X_train = train_df.drop(columns=['churn']).copy()
y_train = train_df['churn'].copy()

X_valid = valid_df.drop(columns=['churn']).copy()
y_valid = valid_df['churn'].copy()

X_test = test_df.drop(columns=['churn']).copy()
y_test = test_df['churn'].copy()


# =====================================================
# 8. 모델 정의
# =====================================================
neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
XGB_SCALE_POS_WEIGHT = neg_count / pos_count

lr_estimator = LogisticRegression(
    C=LR_C,
    max_iter=LR_MAX_ITER,
    class_weight='balanced',
    random_state=42
)

rf_estimator = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    min_samples_leaf=RF_MIN_SAMPLES_LEAF,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

gb_estimator = GradientBoostingClassifier(
    n_estimators=GB_N_ESTIMATORS,
    learning_rate=GB_LEARNING_RATE,
    subsample=GB_SUBSAMPLE,
    random_state=42
)

lgb_estimator = LGBMClassifier(
    objective='binary',
    n_estimators=LGB_N_ESTIMATORS,
    learning_rate=LGB_LEARNING_RATE,
    max_depth=-1,
    num_leaves=LGB_NUM_LEAVES,
    min_child_samples=LGB_MIN_CHILD_SAMPLES,
    subsample=LGB_SUBSAMPLE,
    colsample_bytree=LGB_COLSAMPLE,
    reg_alpha=LGB_REG_ALPHA,
    reg_lambda=LGB_REG_LAMBDA,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

xgb_estimator = XGBClassifier(
    n_estimators=XGB_N_ESTIMATORS,
    learning_rate=XGB_LEARNING_RATE,
    max_depth=XGB_MAX_DEPTH,
    min_child_weight=XGB_MIN_CHILD_WEIGHT,
    subsample=XGB_SUBSAMPLE,
    colsample_bytree=XGB_COLSAMPLE,
    objective='binary:logistic',
    gamma=XGB_GAMMA,
    reg_alpha=XGB_REG_ALPHA,
    reg_lambda=XGB_REG_LAMBDA,
    eval_metric='auc',
    scale_pos_weight=XGB_SCALE_POS_WEIGHT,
    random_state=42
)


# =====================================================
# 9. 전체 파이프라인 생성 함수
# =====================================================
def make_full_pipeline(model):
    return Pipeline(
        steps=[
            ('feature_engineering', FeatureEngineer()),
            ('preprocessing', clone(preprocessor)),
            ('model', clone(model))
        ]
    )


# =====================================================
# 10. 파이프라인 생성
# =====================================================
lr_pipe = make_full_pipeline(lr_estimator)
rf_pipe = make_full_pipeline(rf_estimator)
gb_pipe = make_full_pipeline(gb_estimator)
lgb_pipe = make_full_pipeline(lgb_estimator)
xgb_pipe = make_full_pipeline(xgb_estimator)


# =====================================================
# 11. 모델 학습
# =====================================================
print('\n[Logistic Regression] 학습 시작')
lr_pipe.fit(X_train, y_train)

print('[Random Forest] 학습 시작')
rf_pipe.fit(X_train, y_train)

print('[Gradient Boosting] 학습 시작')
gb_pipe.fit(X_train, y_train)

print('[LightGBM] 학습 시작')
lgb_pipe.fit(X_train, y_train)

print('[XGBoost] 학습 시작')
xgb_pipe.fit(X_train, y_train)


# =====================================================
# 12. 평가
# =====================================================
evaluate_pipeline(lr_pipe, X_train, y_train, 'LR Train')
evaluate_pipeline(lr_pipe, X_valid, y_valid, 'LR Validation')
evaluate_pipeline(lr_pipe, X_test, y_test, 'LR Test')

evaluate_pipeline(rf_pipe, X_train, y_train, 'RF Train')
evaluate_pipeline(rf_pipe, X_valid, y_valid, 'RF Validation')
evaluate_pipeline(rf_pipe, X_test, y_test, 'RF Test')

evaluate_pipeline(gb_pipe, X_train, y_train, 'GB Train')
evaluate_pipeline(gb_pipe, X_valid, y_valid, 'GB Validation')
evaluate_pipeline(gb_pipe, X_test, y_test, 'GB Test')

evaluate_pipeline(lgb_pipe, X_train, y_train, 'LGB Train')
evaluate_pipeline(lgb_pipe, X_valid, y_valid, 'LGB Validation')
evaluate_pipeline(lgb_pipe, X_test, y_test, 'LGB Test')

evaluate_pipeline(xgb_pipe, X_train, y_train, 'XGB Train')
evaluate_pipeline(xgb_pipe, X_valid, y_valid, 'XGB Validation')
evaluate_pipeline(xgb_pipe, X_test, y_test, 'XGB Test')


# =====================================================
# 13. 중요 변수 확인
# =====================================================
print('\n[Random Forest 중요 변수]')
rf_importance_df = get_feature_importance_from_pipeline(rf_pipe, top_n=20)

print('\n[Gradient Boosting 중요 변수]')
gb_importance_df = get_feature_importance_from_pipeline(gb_pipe, top_n=20)

print('\n[LightGBM 중요 변수]')
lgb_importance_df = get_feature_importance_from_pipeline(lgb_pipe, top_n=20)

print('\n[XGBoost 중요 변수]')
xgb_importance_df = get_feature_importance_from_pipeline(xgb_pipe, top_n=20)


# =====================================================
# 14. Voting Ensemble
#    주의: sklearn VotingClassifier 안에 전체 파이프라인을 직접 넣으면
#    중첩 구조가 복잡해질 수 있어, 여기서는 동일 feature/preprocess를 공유하는
#    개별 파이프라인 대신 기존 방식과 유사하게 "모델 레벨 voting"은 생략.
#    필요하면 soft voting용 별도 구현 권장.
# =====================================================


# =====================================================
# 15. 저장
# =====================================================
SAVE_DIR = '../models'
os.makedirs(SAVE_DIR, exist_ok=True)

joblib.dump(lr_pipe, os.path.join(SAVE_DIR, 'lr_full_pipeline.joblib'))
joblib.dump(rf_pipe, os.path.join(SAVE_DIR, 'rf_full_pipeline.joblib'))
joblib.dump(gb_pipe, os.path.join(SAVE_DIR, 'gb_full_pipeline.joblib'))
joblib.dump(lgb_pipe, os.path.join(SAVE_DIR, 'lgb_full_pipeline.joblib'))
joblib.dump(xgb_pipe, os.path.join(SAVE_DIR, 'xgb_full_pipeline.joblib'))

metadata = {
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'feature_cols': FEATURE_COLS,
    'train_rows': int(len(train_df)),
    'valid_rows': int(len(valid_df)),
    'test_rows': int(len(test_df)),
    'saved_files': [
        'lr_full_pipeline.joblib',
        'rf_full_pipeline.joblib',
        'gb_full_pipeline.joblib',
        'lgb_full_pipeline.joblib',
        'xgb_full_pipeline.joblib'
    ]
}

with open(os.path.join(SAVE_DIR, 'metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print('\n저장 완료')
for fname in sorted(os.listdir(SAVE_DIR)):
    print('-', fname)