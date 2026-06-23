# Signal-T

> 고객이 이탈하기 전에 보내는 신호를 데이터로 포착하는 통신사 고객 이탈 예측 서비스

Signal-T는 얼굴 인식 기반 로그인과 머신러닝 모델을 결합한 Streamlit 웹 애플리케이션입니다. 사용자는 아이디/비밀번호 인증과 얼굴 2차 인증을 통과한 뒤, 고객 정보를 입력해 이탈 확률과 위험도를 확인할 수 있습니다.

## 목차

- [팀 소개](#팀-소개)
- [WBS](#wbs)
- [프로젝트 개요](#프로젝트-개요)
- [프로젝트 구조](#프로젝트-구조)
- [기술 스택](#기술-스택)
- [EDA](#eda)
- [주요 기능](#주요-기능)
- [데이터 전처리 결과서](#데이터-전처리-결과서)
  - [주요 전처리 기준](#주요-전처리-기준)
  - [전처리 결과](#전처리-결과)
- [학습된 인공지능 모델](#학습된-인공지능-모델)
  - [모델 아티팩트 목록](#모델-아티팩트-목록)
  - [이탈 예측 모델 구조](#이탈-예측-모델-구조)
  - [다음 통신사 예측 모델 구조](#다음-통신사-예측-모델-구조)
- [모델 학습 결과서](#모델-학습-결과서)
  - [이탈 예측 모델](#이탈-예측-모델)
  - [다음 통신사 예측 모델](#다음-통신사-예측-모델)
## 팀 소개

| 구분 | 내용 |
| :--- | :--- |
| 팀명 | SKT |
| 팀원 | 정세환, 정재희, 이근준, 박회종, 홍기백 |
| 역할 | 박회종: 데이터 전처리, 모델 학습<br>정세환: EDA, 모델 학습<br>이근준: 백엔드, 모델 학습<br>정재희: 백엔드, UI 구축 <br>홍기백: UI, 진행 과정 문서화 |

## WBS

| WBS | 작업 항목 | 세부 작업 | 담당자 | 산출물 |
| :--- | :--- | :--- | :--- | :--- |
| 1.0 | 프로젝트 기획 | 주제 선정, 문제 정의, 서비스 목표 설정 | 전원 | 프로젝트 개요 문서 |
| 1.1 | 진행 과정 문서화 | 프로젝트 진행 내용, 역할, 구조, 실행 방법 정리 | 홍기백 | README.md |
| 2.0 | 데이터 분석 | 통신 고객 데이터 구조 파악 및 분석 방향 수립 | 박회종 | 데이터 분석 계획 |
| 2.1 | EDA | 연령, 결합 여부, 연도별, 통신사별 이탈, 타겟 변수 분포 분석 | 정세환 | EDA 결과, 시각화 자료 |
| 2.2 | 데이터 전처리 | 결측치 처리, 범주형 변수 정리, 파생변수 생성 | 박회종 | 전처리 노트북, 학습 데이터셋 |
| 3.0 | 모델 학습 | 이탈 예측 모델 학습 및 성능 비교 | 박회종, 정세환, 이근준 | 학습 노트북, 모델 성능 결과 |
| 3.1 | 베이스라인 모델 학습 | Logistic Regression, Random Forest 등 기본 모델 실험 | 박회종, 정세환 | 베이스라인 모델 |
| 3.2 | 고성능 모델 학습 | LightGBM, XGBoost, Gradient Boosting 모델 실험 | 박회종, 정세환, 이근준 | LGBM/XGB/GB 모델 파일 |
| 3.3 | 최종 모델 선정 | ROC-AUC 기준 모델 비교, 앙상블 모델 구성 | 박회종, 정세환, 이근준 | 최종 joblib 모델 |
| 3.4 | 다음 통신사 예측 모델 | SKT/KT/LG U+ 이동 가능성 예측 모델 학습 | 정세환 | next 계열 모델 파일 |
| 4.0 | 로그인 기능 구축 | 사용자 계정 기반 로그인 기능 구현 | 이근준 | 로그인 화면 및 인증 로직 |
| 4.1 | 얼굴 인증 기능 | InsightFace 기반 얼굴 등록/검증 기능 구현 | 이근준 | face_auth.py, registered_faces/ |
| 4.2 | DB 연동 | 사용자 정보, 비밀번호 해시, 얼굴 임베딩 저장 | 정재희, 이근준 | db.py, MySQL 테이블 |
| 5.0 | UI 구축 | Streamlit 기반 서비스 화면 구성 | 정재희 | Streamlit UI |
| 5.1 | 회원가입/로그인 UI | 회원가입, 로그인, 얼굴 촬영 화면 구현 | 이근준, 정재희 | login.py, register.py |
| 5.2 | 이탈 예측 UI | 고객 정보 입력, 이탈 확률, 위험도 결과 화면 구현 | 정재희 | tab_telecom_churn.py |
| 5.3 | 다음 통신사 예측 UI | 통신사 이동 예측 및 경쟁 분석 화면 구현 | 정재희 | telecom_predict.py |
| 6.0 | 서비스 통합 | 모델 추론 로직과 UI 연결, 앱 실행 흐름 통합 | 정재희, 이근준 | app.py, 서비스 모듈 |
| 6.1 | 예측 서비스 연결 | 입력값 변환, 모델 로딩, 예측 결과 반환 | 정재희, 정세환 | telecom_churn_service.py, next_provider_service.py |
| 7.0 | 테스트 및 검증 | 기능별 실행 확인, 오류 수정, 최종 시연 점검 | 전원 | 테스트 결과, 최종 실행본 |
| 7.1 | 모델 검증 | 모델 성능 지표 확인 및 결과 해석 | 박회종, 정세환, 이근준 | 성능 비교 결과 |
| 7.2 | 기능 검증 | 로그인, 얼굴 인증, 예측 화면 동작 확인 | 이근준, 정재희 | 기능 테스트 결과 |
| 7.3 | 발표자료 제작 | 발표 흐름 구성, 시연 화면 정리, 발표자료 작성 | 정재희, 이근준, 정세환 | 최종 발표자료 |
| 7.4 | 최종 문서 정리 | 프로젝트 구조, 사용 방법, 개선 계획 보완 | 홍기백 | 최종 README.md |

## 프로젝트 개요

### 프로젝트명

**Signal-T**

고객이 이탈하기 전에 보내는 **시그널**을 데이터로 포착해 내겠다는 의미를 담았습니다. 여기서 `T`는 `Telecom`을 의미합니다.

### 프로젝트 배경
#### 통신 시장은 매년 반복되는 가입자 이탈로 인해 신규 고객 유치를 위한 막대한 마케팅 비용이 발생하고 있습니다. 이러한 시장 환경 속에서 단순한 신규 고객 확보보다는 기존 고객의 리텐션(Retention)을 강화하는 전략이 기업의 수익성 개선을 위한 필수 과제로 부상하고 있습니다.


### 프로젝트 목표

통신사의 고객 데이터와 머신러닝 모델을 활용해 이탈 가능성이 높은 고객을 선제적으로 탐지합니다. 이를 통해 불필요한 마케팅 비용을 줄이고, 고위험 고객에게 집중된 리텐션 전략을 제공하는 것을 목표로 합니다.

## 프로젝트 구조

본 프로젝트는 서비스 실행부, 핵심 비즈니스 로직, UI 컴포넌트, 그리고 데이터 분석 및 모델 아티팩트로 체계적으로 구분되어 있습니다.

```text
SKN32-2nd-1Team/
├── app.py                       # [실행] Streamlit 메인 엔트리 포인트
├── churn_view.py                # 이탈 예측 UI 실험/보조 뷰
├── config.toml                  # Streamlit 테마 설정
├── requirements.txt             # Python 의존성 목록
├── run.bat                      # Windows 실행 스크립트
├── .gitignore                   # Git 제외 파일 설정
├── .streamlit                   # Streamlit 설정 파일
├── README.md                    # 프로젝트 문서
│
├── app/                         # [비즈니스 로직]
│   ├── __init__.py
│   ├── churn_service.py         # 예측 모델 로딩 및 추론 서비스
│   ├── db.py                    # MySQL 데이터베이스 연결 관리
│   ├── face_auth.py             # InsightFace 기반 얼굴 인식/인증
│   ├── next_provider_service.py # 다음 통신사 예측 모델 로딩 및 추론
│   ├── telecom_churn_service.py # 통신사 이탈 예측 로직
│   └── ui.py                    # 세션 상태 관리 및 UI 유틸리티
│
├── view/                        # [UI 컴포넌트]
│   ├── login.py                 # 로그인 화면
│   ├── register.py              # 회원 등록 및 얼굴 데이터 저장
│   ├── tab_telecom_churn.py     # 통신사 이탈 예측 대시보드 탭
│   └── telecom_predict.py       # 다음 통신사 예측 화면
│
├── data/                        # [데이터 분석]
│   ├── data.md                  # 데이터셋 기술 문서
│   ├── churn_modeling.ipynb     # 이탈 예측 모델링 노트북
│   ├── churn_modeling.py        # 이탈 예측 모델링 스크립트
│   ├── dataset.ipynb            # 데이터 탐색 및 전처리
│   ├── dataset_setting_model.ipynb
│   ├── model.ipynb              # 모델 성능 실험 기록
│   ├── next_churn_v3.ipynb      # 다음 통신사 예측 모델링 노트북
│   ├── next_churn_v3.py         # 다음 통신사 예측 모델링 스크립트
│   ├── next_xgb_churn_v3.ipynb  # 다음 통신사 XGBoost 실험 노트북
│   └── preprocess.ipynb         # 데이터 전처리 파이프라인
│
├── models/                      # [모델 아티팩트]
│   ├── churn_model.joblib       # 베이스 모델
│   ├── gb_churn_model.joblib    # Gradient Boosting 단일 모델
│   ├── gb_full_pipeline.joblib  # Gradient Boosting 파이프라인
│   ├── lgb_churn_model.joblib   # LightGBM 단일 모델
│   ├── lgb_full_pipeline.joblib # LightGBM 파이프라인 통합 모델
│   ├── lr_churn_model.joblib    # Logistic Regression 모델
│   ├── lr_full_pipeline.joblib  # Logistic Regression 파이프라인
│   ├── next_label_encoder.joblib # 다음 통신사 라벨 인코더
│   ├── next_lgb_churn_model.joblib # 다음 통신사 LGBM 다중 분류 모델
│   ├── next_xgb_churn_v3.joblib # 다음 통신사 XGBoost 모델
│   ├── rf_churn_model.joblib    # Random Forest 모델
│   ├── rf_full_pipeline.joblib  # Random Forest 파이프라인
│   ├── voting_churn_model.joblib # 앙상블 보팅 모델
│   ├── xgb_churn_model.joblib   # XGBoost 단일 모델
│   ├── xgb_model.joblib         # XGBoost 보조 모델
│   ├── xgb_pipeline.joblib      # XGBoost 파이프라인
│   └── xgb_full_pipeline.joblib # XGBoost 전체 파이프라인
│
└── registered_faces/            # 얼굴 인식용 이미지 저장소
```

## 기술 스택

| 분류 | 기술 및 도구 | 주요 패키지 | 활용 목적 |
| :--- | :--- | :--- | :--- |
| 언어 | Python 3.11+ | `python-dateutil`, `pytz` | 데이터 전처리, 모델링, 웹 서비스 구현 |
| 프론트엔드 | Streamlit | `streamlit` | 고객 이탈 위험도 스코어링 대시보드 구현 |
| 얼굴 인증 | InsightFace, OpenCV, ONNX Runtime, Pillow | `insightface`, `opencv-python`, `onnxruntime`, `Pillow` | 얼굴 검출, 임베딩 추출, 이미지 처리, 2차 인증 체계 구축 |
| 데이터 처리/EDA | Pandas, NumPy, Matplotlib, Seaborn, Plotly | `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly` | 데이터 정제, 파생 변수 생성, 시각화 |
| 머신러닝 | Scikit-learn, Joblib | `scikit-learn`, `joblib` | 모델 학습, 전처리 파이프라인 구성, 모델 저장 및 로딩 |
| 최종 알고리즘 | LightGBM, XGBoost | `lightgbm`, `xgboost` | 트리 기반 예측 모델 및 앙상블 실험 |
| 데이터베이스 | MySQL, mysql-connector-python | `mysql-connector-python` | 회원 정보와 얼굴 임베딩 데이터 저장 |
| 협업/문서화 | GitHub, Notion | - | 버전 관리, 일정 관리, 문서화 |



## EDA
### 타겟 변수 분포
<img width="900" height="409" alt="image" src="https://github.com/user-attachments/assets/b869f701-efc5-4931-aa69-9cc416ff6554" />

### 연도별 이탈 현황
<img width="900" height="409" alt="image" src="https://github.com/user-attachments/assets/acbb2d23-c363-4113-82f3-caf3c4639e49" />

### 나이대 X 통신사별 이탈율
<img width="900" height="409" alt="image" src="https://github.com/user-attachments/assets/e80971d3-88e9-42d7-949e-79d84a75e72c" />

### 모바일 결합 여부별 이탈율
<img width="900" height="409" alt="image" src="https://github.com/user-attachments/assets/7f8c5776-f9b8-409a-9d81-7e030d070ee7" />

## 주요 기능
### 얼굴 2차 인증 로그인
- **1차 인증:** 아이디와 비밀번호 기반 로그인
- **2차 인증:** 카메라 촬영 얼굴과 등록 얼굴 비교
- **회원 등록:** 아이디, 비밀번호, 이름, 얼굴 이미지 등록
- **인증 기술:** InsightFace와 OpenCV 기반 얼굴 검출 및 임베딩 비교

### 머신러닝 기반 이탈 예측
- **예측 모델:** XGBoost 기반 이진 분류(이탈 여부) 및 다중 분류(차기 통신사) 모델
- **예측 결과:** 개인 이탈 확률(0~100%) 및 SKT / KT / LGU+ 이동 확률
- **입력 항목:** 나이, 소득, 가구원수, 현재 통신사, 월 통신비, 할부금, 결합할인 여부, 가입 연수, 누적 통신사 변경 횟수 등 17개 피처

### 결과 시각화
- Streamlit 기반 대시보드 제공
- 이탈 확률 표시
- 이탈 여부에 따른 ai 분석 메시지 제공

## 데이터 전처리 결과서

한국미디어패널조사 원시자료를 활용하여 고객 이탈 예측 모델 학습에 사용할 개인-연도 단위의 패널 데이터를 구성했습니다. 고객 이탈은 동일 개인의 현재 조사 연도와 다음 조사 연도의 가입 이동 통신사를 비교하여, 통신사가 변경된 경우 `churn=1`, 동일한 경우 `churn=0`으로 정의했습니다. 마지막 조사 연도인 2024년은 다음 연도 관측치가 없어 이탈 라벨을 만들 수 없으므로 학습 및 평가 대상에서 제외했습니다.

### 데이터 출처 및 범위

| 항목 | 내용 |
| :--- | :--- |
| 데이터 출처 | 정보통신정책연구원(KISDI) 미디어통계포털 한국미디어패널조사 원시자료 |
| 조사 기간 | 2010년 ~ 2024년 |
| 조사 대상 | 전국 17개 시도 만 6세 이상 가구원 |
| 분석 단위 | 개인-연도 단위 패널 데이터 |
| 출처 URL | https://stat.kisdi.re.kr/kor/contents/ContentsList.html |

### 주요 전처리 기준

| 구분 | 처리 내용 |
| :--- | :--- |
| 변수명 통일 | 연도별 원천 변수명을 분석용 변수명으로 변경한 뒤 개인-연도 단위로 결합 |
| 통신사 변수 통합 | 2010 ~ 2014년은 `mobile_provider`, 2015 ~ 2024년은 `smartphone_provider`를 우선 사용하고 결측 시 `cellular_provider`로 보완 |
| 이탈 라벨 생성 | 다음 조사 시점의 `provider`가 현재 시점과 다르면 `1`, 같으면 `0` |
| 단위 보정 | 2010년 월평균 휴대폰 이용 총 금액과 기기 할부금 단위를 다른 연도 기준으로 통일 |
| 결측 처리 | 모름/무응답 범주는 정보 없음으로 간주하여 결측과 동일하게 처리 |
| 모델 입력 처리 | 수치형 변수는 중앙값 대체 후 StandardScaler, 범주형 변수는 최빈값 대체 후 One-hot 인코딩 적용 |

### 주요 원천 변수 매핑

| 원천 변수 | 변경 변수 | 설명 |
| :--- | :--- | :--- |
| `pid` | `id` | 개인 통합 ID |
| `p__age` | `age` | 나이 카테고리 |
| `p__gender` | `gender` | 성별 |
| `p__income` | `income` | 소득 카테고리 |
| `p__school` | `school` | 학력 |
| `p__hhldsiz` | `household_size` | 가구원 수 |
| `p__job1` | `job` | 직업 유무 |
| `p__mar` | `marriage` | 결혼 상태 |
| `p__a02014` | `cellular_provider` | 일반휴대폰 가입 이동 통신사 |
| `p__a03008` | `smartphone_provider` | 스마트폰 가입 이동 통신사 |
| `p__a01027` | `mobile_provider` | 가입 이동 통신사 |
| `p__c01001` | `monthly_total_cost` | 월평균 휴대폰 이용 총 금액 |
| `p__c01003` | `monthly_installment` | 월평균 기기 할부금 |
| `p__c02001` | `cost_payer` | 휴대폰 요금 부담자 |
| `p__c02003` | `is_mobile_bundled` | 휴대폰 결합상품 가입 여부 |

### 최종 가공 변수

| 변수명 | 설명 |
| :--- | :--- |
| `id` | 개인 통합 ID |
| `year` | 조사 연도 |
| `age` | 나이 카테고리 |
| `gender` | 성별 |
| `income` | 소득 카테고리 |
| `school` | 학력 |
| `household_size` | 가구원 수 |
| `job` | 직업 유무 |
| `marriage` | 결혼 상태 |
| `provider` | 가입 이동 통신사 |
| `monthly_total_cost` | 월평균 휴대폰 이용 총 금액 |
| `monthly_installment` | 월평균 기기 할부금 |
| `cost_payer` | 휴대폰 요금 부담자 |
| `is_mobile_bundled` | 휴대폰 결합상품 가입 여부 |
| `churn` | 다음 조사 시 가입 이동 통신사 변경 여부 |

### 전처리 결과

| 항목 | 전처리 전 | 전처리 후 |
| :--- | :--- | :--- |
| 전체 행 수 | 156,356 | 118,336 |

본 데이터는 동일 개인이 여러 해에 걸쳐 반복 관측되는 패널 데이터이므로 연도 간 비교 시 시점 차이를 함께 고려해야 합니다. 또한 이동 통신사 통합 과정에서 `smartphone_provider`와 `cellular_provider`가 불일치한 사례가 일부 존재했으나, 전체 30건으로 규모가 작아 분석 결과에 미치는 영향은 제한적이라고 판단했습니다.

## 학습된 인공지능 모델

학습 결과로 생성된 모델 아티팩트는 `models/` 디렉터리에 저장되어 있습니다. 본 프로젝트는 고객의 **이탈 여부/확률을 예측하는 모델**과, 통신사를 변경한다고 가정했을 때 **다음 이동 통신사 후보를 예측하는 모델**을 분리하여 사용합니다.

### 모델 아티팩트 목록

| 구분 | 파일명 | 알고리즘 | 용도 |
| :--- | :--- | :--- | :--- |
| 이탈 예측 파이프라인 | `xgb_full_pipeline.joblib` | XGBoost + sklearn Pipeline | 고객 이탈 확률 예측 |
| 이탈 예측 파이프라인 | `lgb_full_pipeline.joblib` | LightGBM + sklearn Pipeline | LightGBM 기반 이탈 확률 예측 |
| 이탈 예측 파이프라인 | `gb_full_pipeline.joblib` | Gradient Boosting + sklearn Pipeline | 비교 실험용 이탈 예측 |
| 이탈 예측 파이프라인 | `rf_full_pipeline.joblib` | Random Forest + sklearn Pipeline | 비교 실험용 이탈 예측 |
| 이탈 예측 파이프라인 | `lr_full_pipeline.joblib` | Logistic Regression + sklearn Pipeline | 베이스라인 비교 |
| 단일 모델 | `churn_model.joblib` | LightGBM | 전처리 완료 데이터 기준 단일 이탈 예측 모델 |
| 단일 모델 | `xgb_model.joblib` | XGBoost | XGBoost 보조 모델 |
| 단일 모델 | `lgb_churn_model.joblib` | LightGBM | LightGBM 단일 모델 |
| 단일 모델 | `xgb_churn_model.joblib` | XGBoost | XGBoost 단일 모델 |
| 단일 모델 | `gb_churn_model.joblib` | Gradient Boosting | Gradient Boosting 단일 모델 |
| 단일 모델 | `rf_churn_model.joblib` | Random Forest | Random Forest 단일 모델 |
| 단일 모델 | `lr_churn_model.joblib` | Logistic Regression | Logistic Regression 단일 모델 |
| 앙상블 모델 | `voting_churn_model.joblib` | Voting Ensemble | 여러 모델 결과를 결합한 최종 앙상블 |
| 다음 통신사 예측 | `next_xgb_churn_v3.joblib` | XGBoost Multi-class | SKT/KT/LG U+ 이동 확률 예측 |
| 다음 통신사 예측 | `next_lgb_churn_model.joblib` | LightGBM Multi-class | 다음 통신사 예측 보조 모델 |
| 인코더 | `next_label_encoder.joblib` | LabelEncoder | 다음 통신사 클래스 라벨 복원 |

### 이탈 예측 모델 구조

`xgb_full_pipeline.joblib`은 sklearn Pipeline 기반의 3단계 구조입니다. 원본 입력 13개 컬럼을 받아 파생변수 생성, 전처리, XGBoost 이진분류 모델을 순서대로 통과시킨 뒤 이탈 확률을 반환합니다.

<img width="850" height="712" alt="image" src="https://github.com/user-attachments/assets/0603f8f8-fb8c-4d00-bfbc-f7bd05c17097" />


| 단계 | 구성 | 처리 내용 | 출력 |
| :--- | :--- | :--- | :--- |
| 원본 입력 | 13개 컬럼 | `age`, `gender`, `income`, `school`, `job`, `marriage`, `provider`, `monthly_total_cost` 등 입력 | 원본 feature |
| 1. Feature Engineering | 커스텀 `FeatureEngineer` | 비율, 로그, 전년 대비 변화율, 3개월 이동평균/표준편차 등 파생변수 생성 | 49개 feature |
| 2. Preprocessing | `ColumnTransformer` | 수치형 32개 컬럼은 중앙값 대체 후 StandardScaler 적용, 범주형 15개 컬럼은 최빈값 대체 후 One-hot 인코딩 적용 | 78차원 벡터 |
| 3. Model | `XGBClassifier` | `binary:logistic` 목적함수 기반 이진분류 수행 | 이탈 확률 |

주요 하이퍼파라미터는 `n_estimators=1000`, `max_depth=4`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=10`, `reg_alpha=1`, `reg_lambda=2`, `eval_metric=auc`입니다. 클래스는 `[0, 1]`이며 `0`은 유지, `1`은 이탈을 의미합니다.

```text
원본 입력 13개 컬럼
  -> FeatureEngineer: 13개 컬럼을 49개 feature로 확장
  -> ColumnTransformer: 수치형/범주형 전처리 후 78차원 벡터 생성
  -> XGBClassifier: 이탈 여부 이진분류
  -> 이탈 확률(0.0 ~ 1.0)
```

### 다음 통신사 예측 모델 구조

`next_xgb_churn_v3.joblib`은 사용자가 통신사를 변경한다고 가정했을 때 SKT, KT, LG U+ 중 어느 통신사로 이동할 가능성이 높은지 예측하는 다중분류 모델입니다. 이 모델은 이탈 여부를 판단하는 모델이 아니라, 이동 후보 통신사별 확률을 계산하는 보조 분석 모델입니다.

<img width="850" height="712" alt="image" src="https://github.com/user-attachments/assets/ae704320-2a8a-4466-b623-ee2587b4a7c2" />


| 입력 구분 | 개수 | 내용 |
| :--- | :--- | :--- |
| 사용자 입력 | 9개 | `age`, `income`, `provider`, `tenure`, `monthly_total_cost` 등 |
| 계산값 | 1개 | `cost_change_rate` |
| 고정값 | 7개 | `area`, `gender`, `job`, `marriage` 등 최빈값 기반 고정 입력 |
| 최종 입력 | 17개 | 사용자 입력, 계산값, 고정값을 결합한 feature 벡터 |

모델은 `XGBClassifier`의 다중분류 설정을 사용하며, `objective=multi:softprob`, `max_depth=5`, `learning_rate=0.05`, ~~조기종료~~ 최종학습 396라운드 기준으로 학습되었습니다. 클래스는 0 → KT, 1 → LGU+, 2 → SKT (LabelEncoder 인코딩 순서) 

```text
사용자 입력 9개 + 계산값 1개 + 고정값 7개
  -> 17개 feature 벡터
  -> XGBClassifier 다중분류(multi:softprob)
  -> SKT / KT / LG U+ 이동 확률
→ LabelEncoder inverse_transform으로 통신사명 복원
  -> 현재 가입사는 0%로 강제 처리하고 나머지 통신사 확률을 카드 UI에 표시
```
## 모델 학습 결과서
### 이탈 예측 모델 
데이터 전처리 및 모델링 과정에서 진행한 실험과 ROC-AUC 성능 변화를 정리한 표입니다. LightGBM을 기준 지표로 사용했으며, ★ 표시는 최종 채택 실험을 의미합니다.

  | 실험 | 핵심 변경 | ROC-AUC (LightGBM) |
| :--- | :--- | :--- |
| 001 | 베이스라인 (로지스틱) | 0.6520 |
| 002 | LightGBM + 추가 변수 | 0.6266 |
| 003 | 변수 정리 + 범주형 처리 | 0.6393 |
| **004** | **전체 패널 + 파생 변수** | **0.6715 ★** |
| 005 | area 범주형 + 변수 제거 | 0.6687 |
| 006 | Optuna 튜닝 | 0.6703 |
| 007 | 결측 전부 채움 | 0.6669 |
| 008 | past_churn / bill_change만 0 채움 | 0.6693 |
| 009 | area 피처 제거 | 0.6682 |
| 010 | XGBoost 모델 추가 | 0.6583 (XGB 기준) |
| 011 | learning_rate 낮춤 | 0.6590 (미학습, 무효) |
| 012 | 신규 변수 4개 추가 | 미확인 (lr 문제 겹침) |
| 013 | SMOTE 오버샘플링 | 0.6669 (하락) |
| 014 | 앙상블 3모델 평균 | 0.6684 (LR 포함 시 하락) |
| **015** | **앙상블 LGB+XGB** | **0.6715 ★** |

총 15회의 실험을 통해 베이스라인(로지스틱 회귀, ROC-AUC 0.6520) 대비 최종 0.6715까지 성능을 향상시켰다.

**주요 인사이트**

- **변수 설계가 가장 큰 영향:** 실험 004에서 전체 패널 데이터를 활용한 파생변수(통신비 변화율, 누적 변경 횟수 등)를 추가했을 때 가장 큰 성능 향상(0.6393 → 0.6715)이 나타났다.
- **결측치 처리 방식의 영향 제한적:** 실험 007 ~ 008에서 결측치 처리 방식을 변경했으나 성능 차이가 미미(0.6669 ~ 0.6693)하여 유의미한 개선으로 보기 어렵다.
- **SMOTE 오버샘플링 효과 없음:** 실험 013에서 클래스 불균형 보정을 위해 SMOTE를 적용했으나 오히려 성능이 하락(0.6669)하여 채택하지 않았다.
- **단일 모델보다 앙상블이 안정적:** 실험 015에서 LightGBM과 XGBoost를 소프트 보팅으로 결합했을 때 단일 모델과 동일한 최고 성능(0.6715)을 유지하면서 예측 안정성이 높아져 최종 채택하였다. 로지스틱 회귀를 포함한 3모델 앙상블(실험 014)은 오히려 성능이 하락하여 제외하였다.
- **하이퍼파라미터 튜닝 효과 제한적:** Optuna를 활용한 튜닝(실험 006)은 0.6703으로 소폭 개선되었으나 최고 성능에는 미치지 못했다.


### 다음 통신사 예측 모델
검증 데이터셋 기준 LightGBM, XGBoost, Voting Ensemble 세 모델의 성능을 비교하였습니다.

| 지표 | LightGBM | XGBoost | Voting Ensemble |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **62.22%** | 61.34% | 62.00% |
| SKT Precision | 0.66 | 0.66 | 0.66 |
| SKT Recall | **0.98** | 0.92 | 0.96 |
| SKT F1 | **0.79** | 0.77 | 0.78 |
| KT Precision | 0.59 | **0.60** | **0.60** |
| KT Recall | **0.54** | 0.46 | 0.49 |
| KT F1 | **0.57** | 0.52 | 0.54 |
| LGU+ Precision | 0.53 | **0.50** | 0.52 |
| LGU+ Recall | 0.21 | **0.37** | 0.29 |
| LGU+ F1 | 0.30 | **0.42** | 0.37 |
| macro avg F1 | 0.55 | **0.57** | 0.56 |
| weighted avg F1 | 0.58 | **0.59** | **0.59** |

#### 주요 인사이트

- **전체 정확도는 LightGBM이 소폭 우위:** LightGBM이 62.22%로 세 모델 중 가장 높았으나 XGBoost(61.34%), Voting Ensemble(62.00%)과 큰 차이는 없다.
- **LightGBM은 SKT 이동 예측에 특화:** SKT Recall 0.98로 실제 SKT 이동 고객 3,327명 중 3,266명(98.2%)을 정확히 예측했으나, LGU+ Recall이 0.21로 낮아 LGU+로 이동하는 고객 대부분을 SKT/KT로 오분류하는 한계가 있다.
- **XGBoost는 3사 균형 예측에 강점:** LGU+ Recall 0.37, F1 0.42로 세 모델 중 LGU+ 예측 성능이 가장 높으며, macro avg F1도 0.57로 가장 우수하다. 단일 통신사에 편향되지 않고 3사를 고르게 예측하는 특성이 있다.
- **Voting Ensemble은 두 모델의 중간:** SKT Recall(0.96)은 LightGBM에 가깝고, LGU+ Recall(0.29)은 두 모델 사이에 위치한다. 특정 클래스에서 압도적인 강점은 없으나 전반적으로 안정적인 성능을 보인다.
- **최종 모델 선정 — XGBoost:** 전체 정확도는 LightGBM보다 소폭 낮지만, 차기 통신사 예측 서비스의 목적상 SKT뿐 아니라 KT와 LGU+ 이동 고객도 고르게 예측하는 것이 중요하다. macro avg F1 기준 최고 성능(0.57)과 LGU+ 예측 우위를 근거로 XGBoost를 최종 모델로 채택하였다.

## 회고
### 정세환

수업을 통해 이론과 기본적 실습으로만 접했던 머신러닝을 이번 프로젝트를 통해 직접 적용해보면서, 쉽지는 않았지만 개념이 훨씬 명확하게 와닿았습니다. 아직 부족한 부분이 많다는 것도 느꼈고, 앞으로 더 공부해야겠다는 동기가 생긴 프로젝트였습니다. 마지막까지 팀원들이 각자 맡은 역할을 잘 수행하고, 함께여서 완성할 수 있었습니다.

---

### 정재희

수업때 이해하기 어려웠던 머신러닝을 프로젝트에 직접 구동시키면서 잘 습득하였던거같습니다, 페이스 로그인 작업에서 cnn모델을 활용하면서 정말 다양한 모델들이 있고 이런 모델들을 잘 활용하여 좋은 제품 및 프로그램을 만들어야겠다는 목표의식도 생겼습니다. 두번째 프로젝트를 진행하면서 ui 파트를 대부분 담당하였는데 1차프로젝트보다  코드 이해도가 높아졌고 작업 진행 속도가 빨라졌음을 알게되면서 머리속에 배운 지식들이 습득이 되고있음을 다시 느꼈습니다.

---

### 이근준

두 번째 단위 프로젝트를 진행해보니 완벽하지는 않더라도 어느정도의 흐름은 저번보다는 잘 이해가 갔습니다. 
그리고 제가 기존에 공부한다고 설치 해두었던 로컬 LLM을 연동하여 활용도 할 수 있어서 만족한 부분입니다.
아직 많이 모자라지만 남은 교육기간에도 교육 집중 및 예복습을 진행하여 다음 프로젝트에는 팀에 더욱 기여하도록 노력해야겠습니다.

---

### 박회종

이번 프로젝트에서는 원시 데이터 추출부터 전처리까지 직접 맡으며, 데이터를 다루는 과정 자체의 중요성을 크게 느꼈습니다. 정제된 데이터셋을 사용하는 것과 달리, 원시 데이터를 직접 가공하는 과정에서는 결측 처리, 변수 정리, 시계열 기준 정렬 등 기본적인 작업 하나하나가 결과에 큰 영향을 미친다는 점을 체감했습니다.
또한 파생 변수를 설계할 때는 데이터의 특성과 의미를 충분히 고려해야 한다는 점을 배웠습니다. 잘 설계된 파생 변수는 모델의 성능을 높일 수 있지만, 반대로 잘못 만든 변수는 오히려 잡음을 키우고 예측 성능을 떨어뜨릴 수 있다는 사실을 확인했습니다. 비록 기대했던 성능에는 도달하지 못했지만, 여러 전처리 방법과 모델링 방식을 시도하면서 많은 것을 배울 수 있었습니다.

---

### 홍기백

이번 프로젝트 과정에서 팀원들과의 소통과 적극적인 질문이 다소 부족했던 점이 아쉽습니다. 특히 초기 단계에서 개발 흐름을 파악하는 데 어려움이 있었음에도, 이를 혼자 해결하려다 보니 적시적인 협업이 이루어지지 못했습니다. 이번 경험을 계기로 향후에는 모르는 부분을 즉각적으로 공유하고, 팀원들과 긴밀히 소통하며 함께 문제를 해결하는 능동적인 협업을 실천하겠습니다.
