# 📊 고객 이탈 예측 서비스 (Face Login + Churn Prediction)

## 프로젝트 개요

**고객 이탈 예측 서비스**는 얼굴 인식 로그인을 거쳐 접근하는 머신러닝 기반 통신사 고객 이탈 위험도 분석 시스템입니다. 아이디/암호 1차 인증과 얼굴 2차 인증을 모두 통과해야 이탈 예측 기능을 사용할 수 있으며, 고객의 인구통계 정보·서비스 이용 패턴·결제 이력을 분석하여 이탈 가능성을 예측합니다.

---

## 🎯 주요 기능

### 1. 얼굴 2차 인증 로그인
- **1차 인증:** 아이디/암호 (MySQL에 해시 저장)
- **2차 인증:** 카메라로 얼굴 촬영 → 등록된 얼굴과 비교 (InsightFace + 코사인 유사도)
- **회원 등록:** 아이디/암호/이름 + 얼굴 사진 동시 등록

### 2. 머신러닝 기반 이탈 예측
- **알고리즘:** scikit-learn 파이프라인 모델 (`models/churn_model.joblib`)
- **예측 결과:** 이탈 확률(%), 위험도 판정(이탈 위험 / 잔류 가능성)
- **입력 항목:** 인적 사항, 통신 이용 행태, 결제 및 요금 (3단 컬럼 입력 폼)

### 3. 결과 시각화
- **카드형 결과 화면:** 위험도 등급 + 이탈 확률 + 진행 바
- **맞춤형 안내 메시지:** 위험도에 따른 대응 전략 제안

---

## 📁 프로젝트 구조

```
SKN32-2nd-1Team/
├── app.py                      # 메인 실행 파일 (로그인 + 이탈예측 화면 전체)
├── app/                         # 핵심 로직 패키지
│   ├── __init__.py
│   ├── db.py                    # MySQL 연결, 회원/얼굴 정보 저장·조회
│   ├── face_auth.py             # 얼굴 검출·등록·인증 (InsightFace)
│   ├── churn_service.py         # 이탈 예측 모델 로딩 및 추론
│   └── ui.py                    # 로그인 세션 상태 관리
├── models/
│   └── churn_model.joblib       # 학습된 이탈 예측 모델
├── registered_faces/            # 등록된 얼굴 원본 이미지 저장 폴더 (자동 생성)
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml              # 다크모드 테마 설정
```

---

## 🛠️ 기술 스택

| 분야 | 기술 |
| :--- | :--- |
| **프론트엔드** | Streamlit |
| **백엔드** | Python 3.11 |
| **얼굴 인식** | InsightFace (buffalo_l), OpenCV |
| **데이터베이스** | MySQL |
| **머신러닝** | Scikit-learn, joblib |
| **데이터 처리** | Pandas, NumPy |

---

## 🚀 설치 및 실행

### 1. 필수 요구사항
- Python 3.11
- MySQL 서버 (설치 및 실행 중이어야 함)
- pip

### 2. 설치 단계

#### Step 1: 가상환경 생성 및 활성화
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

#### Step 2: 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

> ⚠️ **Windows에서 `dlib` 빌드 오류가 발생하는 경우 (face_recognition 사용 시)**
>
> 한글 Windows 환경에서 `dlib` 소스 빌드 중 인코딩 오류가 발생할 수 있습니다.
> (`UnicodeDecodeError: 'cp949' codec can't decode byte...`)
>
> ```bash
> pip uninstall dlib face_recognition -y
> pip install dlib-bin
> pip install face_recognition --no-deps
> pip install git+https://github.com/ageitgey/face_recognition_models
> ```
>
> 현재 버전은 `face_recognition` 대신 **InsightFace**를 사용하므로 이 문제가 발생하지 않지만,
> 추후 라이브러리를 변경할 경우를 대비해 기록해 둡니다.

#### Step 3: MySQL 접속 정보 설정
`app/db.py` 파일에서 본인의 MySQL 비밀번호로 수정합니다.
```python
DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "본인의 MySQL 비밀번호")
```

비밀번호를 모르거나 접속이 안 될 경우, MySQL Workbench로 먼저 접속 테스트를 해보는 것을 권장합니다.

### 3. 실행

```bash
streamlit run app.py
```

브라우저에서 자동으로 `http://localhost:8501`이 열립니다.

---

## 🧪 개발 모드 (로그인 건너뛰기)

매번 얼굴 등록·인증을 거치지 않고 이탈예측 화면만 테스트하고 싶을 때, `app.py` 상단의 스위치를 사용합니다.

```python
DEV_SKIP_LOGIN = True   # 개발 중: 로그인 자동 통과 (MySQL 연결도 건너뜀)
DEV_SKIP_LOGIN = False  # 발표/제출 전: 반드시 False로 변경
```

`True`로 설정하면 사이드바에 "⚙️ 개발 모드" 경고가 표시되어, 실수로 켜둔 상태인지 바로 확인할 수 있습니다.

> ⚠️ **발표나 제출 전에는 반드시 `False`로 되돌려야 합니다.**

---

## 📋 사용 방법

### 1. 회원 등록 (최초 1회)
"1. 얼굴 등록" 탭에서 아이디·암호·이름을 입력하고, 카메라로 얼굴을 촬영한 뒤 등록합니다.

### 2. 로그인
"2. 로그인" 탭에서 아이디/암호를 입력해 1차 인증을 통과한 뒤, 카메라로 얼굴을 촬영해 2차 인증을 진행합니다.

### 3. 고객 정보 입력
로그인 후 화면에서 3단 컬럼(인적 사항 / 통신 이용 행태 / 결제 및 요금)에 고객 정보를 입력합니다.

### 4. 예측 실행 및 결과 확인
"🔍 이탈 여부 예측하기" 버튼을 클릭하면 이탈 확률과 위험도 등급, 권장 대응 전략이 표시됩니다.

---

## 🔧 모델 교체 방법

`models/churn_model.joblib`은 파일명 기준으로 자동 로딩됩니다. 직접 전처리·학습한 모델을 동일한 파일명으로 `models/` 폴더에 덮어쓰기만 하면, 코드 수정 없이 새 모델이 적용됩니다.

단, 모델이 기대하는 입력 컬럼명이 다르면 `app.py`의 입력 폼과 `values` 딕셔너리도 함께 수정해야 합니다.

---

## 📈 향후 개선 계획

- [ ] 자체 전처리 데이터 기반 모델로 교체
- [ ] 로그인 화면 디자인을 이탈예측 화면과 동일한 톤으로 통일
- [ ] 모델 성능 지표(정확도, F1-Score 등) 화면 내 표시
- [ ] 배치 예측 기능 추가

---

**마지막 업데이트:** 2026년 6월 19일