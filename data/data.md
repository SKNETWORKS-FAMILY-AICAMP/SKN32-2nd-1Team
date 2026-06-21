# 고객 이탈 정의
- 다음 조사에서 가입 이동 통신사 변경이 확인되면 이탈로 정의한다.

# 데이터
- 출처: 한국미디어패널조사 https://stat.kisdi.re.kr/kor/contents/ContentsList.html
- 2010년 ~ 2024년 기간 매년 가구방문 면접조사한 데이터

# 공통 변수 (전체 조사에 결측이 없거나 적어 사용이 가능)
- pid : 개인 통합 ID
- p__age : 나이 카테고리
- p__gender : 성별
- p__income : 소득 카테고리
- p__school : 학력
- p__area : 지역
- p__hhldsiz : 가구원 수 (11년~24년)
- p__job1 : 직업 유무
- p__mar : 결혼
- 이동 통신사
    - p__a02014 : 일반휴대폰 가입 이동 통신사 (15년~24년)
    - p__a03008 : 스마트폰 가입 이동 통신사 (15년~24년)
    - p__a01027 : 가입 이동 통신사 (10년~14년)
- p__c01001 : 월평균 휴대폰 이용 총 금액
- p__c01003 : 월평균 기기 할부금
- p__c02001 : 휴대폰 요금 부담자

# 후보 변수 (조사 결측 연도가 있어 사용에 어려움)
- p__a02032 : 일반휴대폰 사용기간(월 기준 환산) (21년~24년)
- p__a03038 : 스마트폰 사용기간(월 기준 환산) (21년~24년)
- p__a01054 : 휴대폰 예상 교체시기 (14년)
- p__a02022 : 일반휴대폰 예상 교체시기 (15년~20년)
- p__a03018 : 스마트폰 예상 교체시기 (15년~20년)
- p__l01001 : 태블릿 PC 보유 여부 (19년~24년)
- p__j01001 : 웨어러블 기기 보유 여부 (17년~24년)

# 1차 가공 후 변수 (연도별 데이터를 시계열 형태로 붙이고 변수명 변경)
- id : 개인 통합 ID
- year : 조사 연도
- age : 나이 카테고리
- gender : 성별
- income : 소득 카테고리
- school : 학력
- area : 지역
- household_size : 가구원 수 (11년~24년)
- job : 직업 유무
- marriage : 결혼
- provider : 가입 이동 통신사
  - cellular_provider : 일반휴대폰 가입 이동 통신사 (15년~24년)
  - smartphone_provider : 스마트폰 가입 이동 통신사 (15년~24년)
  - mobile_provider : 가입 이동 통신사 (10년~14년)
- monthly_total_cost : 월평균 휴대폰 이용 총 금액
- monthly_installment : 월평균 기기 할부금
- cost_payer : 휴대폰 요금 부담자

# 이동 통신사 변수 통합 기준
- 10년~14년 데이터는 mobile_provider 변수를 사용한다.
- 15년~24년 데이터는 smartphone_provider 변수를 사용한다.
  - smartphone_provider 변수가 없으면 cellular_provider 변수를 사용한다.
  - smartphone_provider과 cellular_provider 변수가 결측이 없으면서 다른 데이터가 전체 중 30건으로 영향이 매우 미미하다.

# 최종 가공 후 변수
- id : 개인 통합 ID
- year : 조사 연도
- age : 나이
- gender : 성별
- income : 소득
- school : 학력
- area : 지역
- household_size : 가구원 수
- job : 직업 유무
- marriage : 결혼
- provider : 가입 이동 통신사
- monthly_total_cost : 월평균 휴대폰 이용 총 금액
- monthly_installment : 월평균 기기 할부금
- cost_payer : 휴대폰 요금 부담자
- churn : 다음 조사에 가입 이동 통신사가 바뀌면 이탈이므로 1, 아니면 0
  - 마지막으로 참여한 조사는 다음 조사 데이터가 없으므로 결측으로 처리

# 학습 제외 변수
- id : 개인을 구분하는 식별자일 뿐 예측 신호가 아니다. 고객의 고유번호를 외워서 과적합할 수 있고, 새 고객에는 일반화되지 않을 가능성이 크다.
- year : 고객의 이탈 원인을 직접 설명하는 변수라기보다 단순한 시점 표시자이다.

# 학습 변수
- age : 나이
- gender : 성별
- income : 소득
- school : 학력
- area : 지역
- household_size : 가구원 수
- job : 직업 유무
- marriage : 결혼
- provider : 가입 이동 통신사
- monthly_total_cost : 월평균 휴대폰 이용 총 금액
- monthly_installment : 월평균 기기 할부금
- cost_payer : 휴대폰 요금 부담자
- churn : 다음 조사에 가입 이동 통신사가 바뀌면 이탈이므로 1, 아니면 0

# 문제 변수
area 연도마다 카테고리에 차이가 있어 확인 필요

# 추후 확인 사항
- 순서형 범주를 수치로 보는게 좋은지, 범주로 보아 One-Hot 인코딩이 좋은지 비교
- 딥러닝 모델 학습 시에 train 데이터 셔플 시도
- 시계열을 무시한 랜덤 셔플 분할을 시도
- 고객 기준 데이터 분할을 시도 (동일 고객 누수를 완전 차단하지만, 실제 미래 예측 상황과 다른 문제)
