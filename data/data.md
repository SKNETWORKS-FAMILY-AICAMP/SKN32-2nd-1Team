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
- p__c02003 : 휴대폰 결합상품 가입 여부

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
- household_size : 가구원 수 (11년~24년)
- job : 직업 유무
- marriage : 결혼
- provider : 가입 이동 통신사
  - cellular_provider : 일반휴대폰 가입 이동 통신사 (15년~24년)
  - smartphone_provider : 스마트폰 가입 이동 통신사 (15년~24년)
  - mobile_provider : 가입 이동 통신사 (10년~14년)
- monthly_total_cost : 월평균 휴대폰 이용 총 금액 (단위: 천)
- monthly_installment : 월평균 기기 할부금 (단위: 천)
- cost_payer : 휴대폰 요금 부담자
- is_mobile_bundled : 휴대폰 결합상품 가입 여부

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
- household_size : 가구원 수
- job : 직업 유무
- marriage : 결혼
- provider : 가입 이동 통신사
- monthly_total_cost : 월평균 휴대폰 이용 총 금액 (단위: 천)
- monthly_installment : 월평균 기기 할부금 (단위: 천)
- cost_payer : 휴대폰 요금 부담자
- is_mobile_bundled : 휴대폰 결합상품 가입 여부
- churn : 다음 조사에 가입 이동 통신사가 바뀌면 이탈이므로 1, 아니면 0
  - 마지막으로 참여한 조사는 다음 조사 데이터가 없으므로 결측으로 처리

# 학습 제외 변수
- id : 개인을 구분하는 식별자일 뿐 예측 신호가 아니다. 고객의 고유번호를 외워서 과적합할 수 있고, 새 고객에는 일반화되지 않을 가능성이 크다.

# 학습 변수
- year : 조사 연도 (10년 ~ 24년)
- age : 나이 (1: 만 10세 미만, 2: 만 20세 미만, 3: 만 30세 미만, 4: 만 40세 미만, 5: 만 50세 미만, 6: 만 60세 미만, 7: 만 70세 미만, 8: 만 70세 이상)
- gender : 성별 (1: 남, 2: 여)
- income : 소득 (1: 소득 없음, 2: 50만원 미만, 3: 100만원 미만, 4: 200만원 미만, 5: 300만원 미만, 6: 400만원 미만, 7: 500만원 미만, 8: 500만원 이상)
- school : 학력 (1: 미취학, 2: 초졸 이하, 3: 중졸 이하, 4: 고졸 이하, 5: 대졸 이하, 6: 대학원 재학 이상)
- household_size : 가구원 수
- job : 직업 유무 (1: 예, 2: 아니오)
- marriage : 결혼 (1: 미혼, 2: 결혼, 3: 사별, 4: 이혼)
- provider : 가입 이동 통신사 (1: SKT, 2: KT, 3: LG U+, 4: 알뜰폰)
- monthly_total_cost : 월평균 휴대폰 이용 총 금액 (단위: 천)
- monthly_installment : 월평균 기기 할부금 (단위: 천)
- cost_payer : 휴대폰 요금 부담자 (1: 본인, 2: 회사가 전액 부담, 3: 회사가 일부 지원, 4: 가족이나 타인이 전액 부담, 5: 가족이나 타인이 일부 부담, 6: 기타)
- is_mobile_bundled : 휴대폰 결합상품 가입 여부 (1: 가입, 2: 미가입)
- churn : 다음 조사에 가입 이동 통신사가 바뀌면 이탈이므로 1, 아니면 0 (라벨 변수)


Result

[LR Train]
Accuracy : 0.6066
Precision: 0.4762
Recall   : 0.5819
F1-score : 0.5238
ROC-AUC  : 0.6432

[LR Validation]
Accuracy : 0.589
Precision: 0.4515
Recall   : 0.6324
F1-score : 0.5269
ROC-AUC  : 0.6316

[LR Test]
Accuracy : 0.5496
Precision: 0.4293
Recall   : 0.5205
F1-score : 0.4705
ROC-AUC  : 0.5661

[RF Train]
Accuracy : 0.6563
Precision: 0.5317
Recall   : 0.6336
F1-score : 0.5782
ROC-AUC  : 0.7031

[RF Validation]
Accuracy : 0.6163
Precision: 0.4746
Recall   : 0.5614
F1-score : 0.5143
ROC-AUC  : 0.641

[RF Test]
Accuracy : 0.573
Precision: 0.4434
Recall   : 0.4335
F1-score : 0.4384
ROC-AUC  : 0.5689

[GB Train]
Accuracy : 0.6644
Precision: 0.6077
Recall   : 0.274
F1-score : 0.3777
ROC-AUC  : 0.6734

[GB Validation]
Accuracy : 0.6643
Precision: 0.6157
Recall   : 0.1929
F1-score : 0.2938
ROC-AUC  : 0.6463

[GB Test]
Accuracy : 0.6178
Precision: 0.5905
Recall   : 0.0198
F1-score : 0.0384
ROC-AUC  : 0.5801

[LGB Train]
Accuracy : 0.7107
Precision: 0.5963
Recall   : 0.6864
F1-score : 0.6382
ROC-AUC  : 0.7818

[LGB Validation]
Accuracy : 0.6158
Precision: 0.4736
Recall   : 0.5524
F1-score : 0.51
ROC-AUC  : 0.6409

[LGB Test]
Accuracy : 0.5659
Precision: 0.4316
Recall   : 0.4066
F1-score : 0.4188
ROC-AUC  : 0.5689

[XGB Train]
Accuracy : 0.6541
Precision: 0.5299
Recall   : 0.6159
F1-score : 0.5697
ROC-AUC  : 0.7037

[XGB Validation]
Accuracy : 0.6142
Precision: 0.4727
Recall   : 0.5699
F1-score : 0.5167
ROC-AUC  : 0.647

[XGB Test]
Accuracy : 0.5678
Precision: 0.4377
Recall   : 0.4361
F1-score : 0.4369
ROC-AUC  : 0.5742

[Random Forest 중요 변수]
                         feature  importance
0      cat__provider_changed_0.0    0.125178
1      cat__provider_changed_1.0    0.124218
2              cat__provider_1.0    0.105384
3                      num__year    0.065314
4         cat__prev_provider_1.0    0.042806
5              cat__provider_3.0    0.035655
6              cat__provider_2.0    0.027367
7          num__cost_roll_mean_3    0.022393
8           num__cost_roll_std_3    0.020234
9              num__service_cost    0.020188
10          num__cost_per_person    0.019092
11        cat__prev_provider_3.0    0.018230
12  num__prev_monthly_total_cost    0.018051
13        cat__prev_provider_2.0    0.017558
14   num__monthly_total_cost_log    0.017325
15       num__monthly_total_cost    0.017216
16              num__cost_change    0.017000
17  num__disposable_income_proxy    0.015994
18      num__monthly_installment    0.015283
19  num__monthly_installment_log    0.015114

[Gradient Boosting 중요 변수]
                          feature  importance
0       cat__provider_changed_0.0    0.224355
1       cat__provider_changed_1.0    0.160508
2               cat__provider_1.0    0.153534
3                       num__year    0.124938
4          cat__prev_provider_1.0    0.045647
5                        num__age    0.032587
6           num__cost_roll_mean_3    0.019614
7   num__prev_monthly_installment    0.018250
8            num__cost_roll_std_3    0.016851
9    num__monthly_installment_log    0.014119
10           num__cost_per_person    0.014002
11              num__service_cost    0.013465
12   num__prev_monthly_total_cost    0.012414
13   num__disposable_income_proxy    0.011351
14    num__installment_per_person    0.009701
15       num__monthly_installment    0.009486
16        num__installment_change    0.009353
17              cat__provider_5.0    0.008607
18               num__cost_change    0.006911
19              cat__provider_4.0    0.006136

[LightGBM 중요 변수]
                          feature  importance
0            num__cost_roll_std_3        4744
1           num__cost_roll_mean_3        4490
2           num__cost_change_rate        3766
3                       num__year        3525
4            num__cost_per_person        3232
5    num__prev_monthly_total_cost        2925
6               num__service_cost        2623
7                num__cost_change        2557
8         num__monthly_total_cost        2465
9         num__service_cost_ratio        2215
10         num__cost_income_ratio        2198
11   num__disposable_income_proxy        2008
12                       num__age        1940
13   num__installment_change_rate        1866
14        num__income_roll_mean_3        1839
15  num__prev_monthly_installment        1830
16         num__installment_ratio        1657
17         num__income_roll_std_3        1338
18        num__installment_change        1298
19  num__installment_income_ratio        1120

[XGBoost 중요 변수]
                          feature  importance
0       cat__provider_changed_0.0    0.202692
1       cat__provider_changed_1.0    0.184707
2               cat__provider_1.0    0.092691
3          cat__prev_provider_1.0    0.024745
4               cat__provider_3.0    0.019699
5               cat__provider_2.0    0.016902
6                       num__year    0.016600
7      cat__is_mobile_bundled_2.0    0.016237
8   cat__is_first_observation_1.0    0.011876
9                        num__age    0.011401
10              cat__provider_5.0    0.011366
11   num__monthly_installment_log    0.011235
12         cat__prev_provider_2.0    0.011139
13     cat__is_mobile_bundled_1.0    0.011052
14  cat__is_first_observation_0.0    0.009902
15       num__monthly_installment    0.009805
16         cat__prev_provider_3.0    0.009609
17              cat__marriage_1.0    0.009301
18              cat__provider_4.0    0.008046
19  num__prev_monthly_installment    0.008019
