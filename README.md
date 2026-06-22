# Signal-T

> 고객이 이탈하기 전에 보내는 신호를 데이터로 포착하는 통신사 고객 이탈 예측 서비스

Signal-T는 얼굴 인식 기반 로그인과 머신러닝 모델을 결합한 Streamlit 웹 애플리케이션입니다. 사용자는 아이디/비밀번호 인증과 얼굴 2차 인증을 통과한 뒤, 고객 정보를 입력해 이탈 확률과 위험도를 확인할 수 있습니다.

## 목차

- [팀 소개](#팀-소개)
- [프로젝트 개요](#프로젝트-개요)
- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [데이터 전처리 결과서](#데이터-전처리-결과서)
- [학습된 인공지능 모델](#학습된-인공지능-모델)
- [프로젝트 구조](#프로젝트-구조)
- [설치 및 실행](#설치-및-실행)
- [사용 방법](#사용-방법)
- [모델 교체 방법](#모델-교체-방법)
- [향후 개선 계획](#향후-개선-계획)

## 팀 소개

| 구분 | 내용 |
| :--- | :--- |
| 팀명 | SKT |
| 팀원 | 정세환, 정재희, 이근준, 박회종, 홍기백 |

## 프로젝트 개요

### 프로젝트명

**Signal-T**

고객이 이탈하기 전에 보내는 **시그널**을 데이터로 포착해 내겠다는 의미를 담았습니다. 여기서 `T`는 `Telecom`을 의미합니다.

### 프로젝트 배경

#### 알뜰폰(MVNO)의 공세와 통신 시장 변화

- **합리적 소비 트렌드 확산:** 5G 서비스에 대한 피로감과 자급제 단말기 보급 활성화로 인해 가성비를 중시하는 고객층을 중심으로 알뜰폰(MVNO) 이탈률이 증가하고 있습니다.
- **가입자 사수 경쟁 심화:** 번호이동 시장이 과열되면서 통신 3사(SKT, KT, LGU+) 간 기존 가입자 방어와 신규 가입자 유치 경쟁이 치열해지고 있습니다.

#### 통신사 마케팅 비용 지출의 딜레마

- **출혈 경쟁으로 인한 실적 압박:** 가입자 유지를 위한 마케팅비 지출은 불가피하지만, 이는 기업의 영업이익 감소로 이어질 수 있습니다.
- **비효율적 자원 배분:** 모든 고객에게 동일한 혜택을 제공하는 방식은 마케팅 비용 대비 효과를 낮출 수 있습니다.

#### 데이터 기반 타겟 리텐션의 필요성

- **비용 절감 효과:** 신규 고객 유치 비용은 기존 고객 유지 비용보다 높기 때문에, 이탈 가능성이 높은 고객을 조기에 선별하는 것이 중요합니다.
- **이탈 시그널 포착:** 고객은 이탈 전 결합 상품 해지, 요금 부담률 변화, 서비스 이용량 급감 같은 데이터상 징후를 남길 수 있습니다.

### 프로젝트 목표

통신사의 고객 데이터와 머신러닝 모델을 활용해 이탈 가능성이 높은 고객을 선제적으로 탐지합니다. 이를 통해 불필요한 마케팅 비용을 줄이고, 고위험 고객에게 집중된 리텐션 전략을 제공하는 것을 목표로 합니다.

## EDA
### 타겟 변수 분포
<img width="928" height="409" alt="image" src="https://github.com/user-attachments/assets/b869f701-efc5-4931-aa69-9cc416ff6554" />

### 연도별 이탈 현황
<img width="1189" height="489" alt="image" src="https://github.com/user-attachments/assets/acbb2d23-c363-4113-82f3-caf3c4639e49" />

### 나이대 X 통신사별 이탈율
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/e80971d3-88e9-42d7-949e-79d84a75e72c" />

### 모바일 결합 여부별 이탈율
<img width="990" height="409" alt="image" src="https://github.com/user-attachments/assets/7f8c5776-f9b8-409a-9d81-7e030d070ee7" />

## 주요 기능

### 얼굴 2차 인증 로그인

- **1차 인증:** 아이디와 비밀번호 기반 로그인
- **2차 인증:** 카메라 촬영 얼굴과 등록 얼굴 비교
- **회원 등록:** 아이디, 비밀번호, 이름, 얼굴 이미지 등록
- **인증 기술:** InsightFace와 OpenCV 기반 얼굴 검출 및 임베딩 비교

### 머신러닝 기반 이탈 예측

- **예측 모델:** 
- **예측 결과:** 
- **입력 항목:** 

### 결과 시각화

- Streamlit 기반 대시보드 제공
- 이탈 확률과 위험도 결과 표시
- 위험도에 따른 맞춤형 안내 메시지 제공

## 기술 스택

| 분류 | 기술 및 도구 | 활용 목적 |
| :--- | :--- | :--- |
| 언어 | Python 3.10+ | 데이터 전처리, 모델링, 웹 서비스 구현 |
| 프론트엔드 | Streamlit | 고객 이탈 위험도 스코어링 대시보드 구현 |
| 얼굴 인증 | InsightFace, OpenCV | 얼굴 검출 및 2차 인증 체계 구축 |
| 데이터 처리/EDA | Pandas, NumPy, Matplotlib, Seaborn | 데이터 정제, 파생 변수 생성, 시각화 |
| 머신러닝 | Scikit-learn, Optuna, SMOTE | 베이스라인 모델링, 하이퍼파라미터 최적화, 불균형 데이터 처리 |
| 최종 알고리즘 | LightGBM, XGBoost | 트리 기반 예측 모델 및 앙상블 실험 |
| 데이터베이스 | MySQL | 회원 정보와 얼굴 임베딩 데이터 저장 |
| 협업/문서화 | GitHub, Notion | 버전 관리, 일정 관리, 문서화 |

## 데이터 전처리 결과서

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

### 실험 요약


## 학습된 인공지능 모델

학습 결과로 생성된 모델 아티팩트는 `models/` 디렉터리에 저장되어 있으며, 용도와 구성 방식에 따라 다음과 같이 분류됩니다.

### 모델 아티팩트 목록

| 구분 | 파일명 | 알고리즘 | 용도 |
| :--- | :--- | :--- | :--- |
| 베이스 모델 | '' |  |  |
| 보조 모델 | `` |  |  |
| 파이프라인 | `` |     |  |

### 카테고리별 분류

#### 1. 단일 학습 모델 (Single Model)

- **`churn_model.joblib`**
  - LightGBM 단일 모델
  - 실험 004 ~ 009 구간에서 도출된 핵심 모델
  - 빠른 추론과 가벼운 배포가 필요한 경우 사용

- **`xgb_model.joblib`**
  - XGBoost 단일 모델
  - 실험 010에서 도출 (XGB 기준 ROC-AUC 0.6583)
  - 앙상블 구성 시 LightGBM과 결합되는 보조 모델

#### 2. 전처리 파이프라인 모델 (Pipeline Model)


#### 3. 최종 채택 모델 (Final Ensemble)



## 프로젝트 구조

```text
SKN32-2nd-1Team/
├── app.py                         # 메인 실행 파일 (로그인 + 이탈 예측 화면 진입점)
├── tab_telecom_churn.py                  # 이탈 예측 UI 뷰 컴포넌트
├── config.toml                    # Streamlit 테마 설정
├── requirements.txt               # Python 의존성 목록
├── README.md
├── app/                           # 핵심 비즈니스 로직 패키지
│   ├── __init__.py
│   ├── db.py                      # MySQL 연결 및 회원 정보 관리
│   ├── face_auth.py               # 얼굴 검출, 등록, 로그인 인증 (InsightFace)
│   ├── churn_service.py           # 이탈 예측 모델 로딩 및 추론
│   ├── telecom_churn_service.py   # 통신 고객 이탈 예측 서비스 로직
│   └── ui.py                      # 로그인 세션 및 UI 상태 관리
├── view/                          # 화면 단위 뷰 모듈
│   ├── login.py                   # 로그인 화면
│   ├── register.py                # 회원 등록 화면
│   └── tab_telecom_churn.py       # 이탈 예측 탭 화면
├── data/                          # 데이터 및 분석 노트북
│   ├── data.md                    # 데이터셋 설명 문서
│   ├── churn_modeling.ipynb       # 이탈 예측 모델링 노트북
│   ├── dataset.ipynb              # 데이터셋 탐색 노트북
│   ├── dataset_setting_model.ipynb # 데이터셋 구성 및 모델 셋업
│   ├── model.ipynb                # 모델 실험 노트북
│   └── preprocess.ipynb           # 전처리 노트북
├── models/                        # 학습된 모델 아티팩트
│   ├── churn_model.joblib         # 베이스 LightGBM 모델
│   ├── lgb_churn_model.joblib     # LightGBM 단일 모델
│   ├── xgb_churn_model.joblib     # XGBoost 단일 모델
│   ├── xgb_model.joblib           # XGBoost 보조 모델
│   ├── xgb_pipeline.joblib        # XGBoost + 전처리 파이프라인
│   ├── gb_churn_model.joblib      # Gradient Boosting 모델
│   ├── rf_churn_model.joblib      # Random Forest 모델
│   ├── lr_churn_model.joblib      # Logistic Regression 베이스라인
│   └── voting_churn_model.joblib  # 보팅 앙상블 모델
└── registered_faces/              # 등록된 얼굴 이미지 저장 폴더 (자동 생성)
```

## 설치 및 실행

### 요구 사항

- Python 3.10 이상
- MySQL 서버
- pip

### 1. 가상환경 생성 및 활성화

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

Windows 환경에서 얼굴 인식 관련 패키지 설치 중 `dlib` 빌드 오류가 발생하면 아래 명령을 참고합니다.

```bash
pip uninstall dlib face_recognition -y
pip install dlib-bin
pip install face_recognition --no-deps
pip install git+https://github.com/ageitgey/face_recognition_models
```

현재 프로젝트는 `face_recognition` 대신 InsightFace를 사용하므로, 위 내용은 향후 라이브러리 변경 시 참고용입니다.

### 3. MySQL 접속 정보 설정

`app/db.py`에서 본인의 MySQL 비밀번호를 환경에 맞게 설정합니다.

'''python
DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "본인의 MySQL 비밀번호")
'''

비밀번호를 모르거나 접속이 실패하는 경우 MySQL Workbench에서 먼저 접속 테스트를 진행하는 것을 권장합니다.

### 4. 앱 실행

```bash
streamlit run app.py
```

실행 후 브라우저에서 `http://localhost:8501`로 접속합니다.

## 개발 모드

로그인과 얼굴 인증 과정을 건너뛰고 이탈 예측 화면만 테스트하려면 `app.py` 상단의 개발 모드 값을 변경합니다.

```python
DEV_SKIP_LOGIN = True   # 개발 중 로그인 자동 통과
DEV_SKIP_LOGIN = False  # 발표/제출 전 권장 설정
```

`True`로 설정하면 사이드바에 개발 모드 경고가 표시되며, 로그인 없이 예측 화면을 확인할 수 있습니다.

> 발표 또는 제출 전에는 반드시 `False`로 되돌려야 합니다.

## 사용 방법

1. **회원 등록**
   - 아이디, 비밀번호, 이름을 입력하고 카메라로 얼굴을 촬영해 등록합니다.

2. **로그인**
   - 아이디와 비밀번호로 1차 인증을 진행한 뒤, 카메라 얼굴 인증으로 2차 인증을 진행합니다.

3. **고객 정보 입력**
   - 로그인 후 인적 사항, 통신 이용 행태, 결제 및 요금 정보를 입력합니다.

4. **이탈 예측 실행**
   - 예측 버튼을 클릭해 이탈 확률, 위험도 등급, 권장 대응 전략을 확인합니다.

## 모델 교체 방법

기본 모델은 `models/` 디렉터리의 joblib 파일을 기준으로 로딩됩니다. 새 모델을 적용하려면 기존 모델과 동일한 입력 컬럼을 사용하도록 학습한 뒤, 동일한 파일명으로 `models/` 디렉터리에 배치합니다.

입력 컬럼명이 달라지는 경우에는 `app.py`의 입력값 처리 로직과 모델 추론 코드도 함께 수정해야 합니다.

## 향후 개선 계획

- [ ] 자체 전처리 데이터 기반 모델로 교체
- [ ] 로그인 화면과 예측 화면의 UI 디자인 통일
- [ ] 모델 성능 지표(정확도, F1-score 등) 대시보드 표시
- [ ] 배치 예측 기능 추가
- [ ] 고객 위험도별 리텐션 전략 고도화
