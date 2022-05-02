# 뇌졸중 발병 가능성 예측

- 뇌졸중은 암 다음으로 흔한 사망원인으로 (단일 장기질환 중 사망률 1위), 한번 발생하면 후유증, 합병증이 나타날 가능성이 높음
- 뇌졸중이 한번 발병하면 지속적인 케어가 필요하므로, 가정적 또는 사회적으로 손실이 발생함
- 개인들이 가지고 있는 생활습관, 성별, 나이 등을 이용해 뇌졸중 발병 가능성을 예측 => ***질환으로 인한 여러 손실을 미리 방지하고 대비가 가능***
- 뇌졸중의 위험인자와 부합하는 특성을 사용해, 성인들의 뇌졸중 발병 가능성을 예측하고자 함

<br>

- 파일 구성
  - `Dataset.zip` : BRFSS 2015 Dataset (용량이 커서 압축파일로 uplaod함)
  - `EDA_Modeling` : BRFSS 2015 Dataset으로 EDA, Modling 진행한 주피터 노트북

<br>

# 뇌졸중의 위험인자 (Risk Factor)

**1. 조절 가능한 위험인자**  
  - 고혈압, 고지혈증, 당뇨병, 심장질환	 
  - 흡연, 과도한 음주 (하루에 2잔 이상)	
  - 식이 (고염분, 가공식품, 인스턴트 식품)	
  - 운동부족	
  - 비만	
  - 스트레스	
  - 사회적/경제적 요인	

**2. 조절 불가능한 위험인자**
  - 나이 (55세 이후에는 10년마다 두배 씩 유병확률이 증가함.)  
  - 인종 (African Americans)
  - 성별 (남성이 발병 확률이 높으나, 사망률은 여성이 더 높음.)  

<br>

- 참고 링크
	- [위키백과-뇌졸중](https://ko.wikipedia.org/wiki/%EB%87%8C%EC%A1%B8%EC%A4%91)  
	- [Brain Basics: Preventing Stroke](https://www.ninds.nih.gov/Disorders/Patient-Caregiver-Education/Preventing-Stroke)  
	- [Stroke Risk Factors](https://www.stroke.org/en/about-stroke/stroke-risk-factors/risk-factors-under-your-control)  
	- [충남대 병원- 뇌졸중](https://www.cnuh.co.kr/rcc/sub03_02.do)  
	- [서울아산병원 - 뇌졸중](http://www.amc.seoul.kr/asan/healthinfo/disease/diseaseDetail.do?contentId=30518)

<br>

# Dataset
![image](https://user-images.githubusercontent.com/77204538/166240368-67ba9349-7f2f-4606-92de-7d66e471f556.png)

- Behavioral Risk Factor Surveillance System (BRFSS) 2015 Dataset [(링크)](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system)
  - 미국 전역을 대상으로 행동 위험 요인을 조사하는 전화 설문조사
  - 성인들의 만성질환여부, 흡연, 신체활동, 과일/야채 섭취량 등 다양한 생활 패턴을 조사
  - 설문조사 항목 중, 뇌졸중과 관련된 특성만을 골라 편집함 [(참고링크)](https://www.kaggle.com/code/alexteboul/diabetes-health-indicators-dataset-notebook/notebook)
  - 설문조사를 거부하거나 '모른다' 라고 답한 경우 제외

<br>

# 사용한 특성
**1. Target**
  - 뇌졸중 발생 이력 여부 : `Stroke`
    
**2. Feature**
- 고혈압, 고지혈증, 심장질환, 당뇨병 : `HighBP`, `HighChol`, `HeartDisease`, `Diabetes` (Binary)
- 흡연량 : `Smoker` (비흡연, 금연, 가끔씩 흡연, 매일 흡연)
- 과음(남 – 주 14 잔 이상, 여- 주 7잔 이상) 여부 : `HvyAlcohol` (Binary)
- 하루에 1회 이상 과일, 야채 섭취 여부 : `Fruits`, `Veggies` (Binary)
- 30일 이내 운동 여부, 보행에 대한 어려움 여부 : `PhysActivity`, `DiffWalk` (Binary)
- 비만 (BMI) : BMIGroup (`Underweight`, `Normalweight`, `Overweight`, Obese)
- 지난 30일 동안 스트레스/우울증 등의 문제를 겪은 일 수 : `MentHlth` 
- 소득 : `Income` (Less than $10,000, Less than $15,000, Less than $20,000, Less than $25,000, Less than $35,000, Less than $50,000, Less than $75,000, $75,000 or more)
- 학력 : `Education` (Never/Kindergarten, Elementary, SomeHighSchool, HighSchoolGraduate, SomeCollege/TechnicalSchool, CollegeGraduate)
- 혼인 상태 : `Marital` (NeverMarried, Married, Divorced, Widowed, Separated, MemberOfAnUnmarriedCouple)
- 의료보험 가입 여부, 의료비용에 의한 치료거부 여부 : `AnyHealthcare`, `NoDocbcCost` (Binary)
- 나이 : `AgeGroup` (18-24, 25-29, 30-34,35-39, 40-44, 45-49,50-54, 55-59,60-64, 65-69,70-74, 75-79, 80-)
- 인종 : `Race` (White, Black, AmericanIndian/AlaskanNative, Asian, NativeHawaiian/OtherPacificIslander, Other, Multiracial, Hispanic)
- 성별 : `Sex` (Male, Female)

<br>

# Model
- Logistic Regression, Decision Tree, Random Forest, XGBoost 모델들의 성능을 비교하여, 최적의 모델을 선택
- AUC score 기준, `XGBoost` 모델이 0.825로 가장 높은 성능을 나타냄
- 최적의 모델로 `XGBoost` 선택
---
- Target 데이터의 불균등 보정
  - 모델의 Target이 되는 `Stroke`에서 압도적으로 정상 데이터(`Stroke = 0`)가 많음
  - Oversampling 방법 (SMOTE)을 이용하여 해결하려 하였으나, XGBoost 모델 기준, AUC score가 **0.74**을 기록함
  - Class weight를 주는 방법을 사용하였을 때 **0.825**를 기록하여, 최종적으로 ***Class weight를 주는 방법***을 사용함

<br>

# HyperParameter tuning
- Grid Search cross-validation 사용
- AUC score 증가 (0.825 => 0.826)





