# 뇌졸중 발병 가능성 예측

## **프로젝트 진행배경**
- 뇌졸중은 암 다음으로 흔한 사망원인으로 (단일 장기질환 중 사망률 1위), 한번 발생하면 후유증, 합병증이 나타날 가능성이 높음
- 뇌졸중이 한번 발병하면 지속적인 케어가 필요하므로, 가정적 또는 사회적으로 손실이 발생함
- 개인들이 가지고 있는 생활습관, 성별, 나이 등을 이용해 뇌졸중 발병 가능성을 예측 => ***질환으로 인한 여러 손실을 미리 방지하고 대비가 가능***
> ✔ 성인들의 생활 패턴 중 **뇌졸중 위험인자**와 부합하는 특성을 사용하여, 뇌졸중 발병 질환을 예방할 수 있도록함


<br>

## **개발 환경**
- Google Colab
- 사용된 library & Tools
  - `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `eli5` ,`pdp`,`shap`

<br>

## **파일 구성**
  - `1_Dataset` : BRFSS 2015 Dataset (용량이 커서 압축파일로 uplaod함) [[Googld_Drive]](https://drive.google.com/drive/folders/1lE4dTk9JfI1p7V6zonN9LLMJxxfDsiJP?usp=sharing)
  	- `1_1_2015.csv`
    	- Behavioral Risk Factor Surveillance System (BRFSS) 2015 Dataset
  	- `1_2_Stroke_BRFSS2015.csv`
    	- 설문조사 항목 중, 뇌졸중과 관련될 것이라 생각되는 특성들만 골라 편집한 dataset


  - `2_데이터_전처리.ipynb`
    - `2015.csv`에서 일부 특성들만 선택하고 전처리한 후,  `Stroke_BRFSS2015.csv`로 저장
  - `3_EDA_Modeling.ipynb`
    - 편집한 BRFSS 2015 Dataset으로 EDA, Modling 진행한 주피터 노트북
  

<br>

## **1. Dataset**
<br>

![image](https://user-images.githubusercontent.com/77204538/166240368-67ba9349-7f2f-4606-92de-7d66e471f556.png)

- Behavioral Risk Factor Surveillance System (BRFSS) 2015 Dataset [(링크)](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system)
  - 미국 전역을 대상으로 행동 위험 요인을 조사하는 전화 설문조사
  - 성인들의 만성질환여부, 흡연, 신체활동, 과일/야채 섭취량 등 다양한 생활 패턴을 조사
  - 다양한 생활패턴이 뇌졸중 발병에 어떠한 영향을 끼치는지 살펴보기 위하여 해당 dataset을 선택하였음
  - 설문조사 항목 중, 뇌졸중과 관련된 특성만을 골라 편집함 [(참고링크)](https://www.kaggle.com/code/alexteboul/diabetes-health-indicators-dataset-notebook/notebook)
  - 설문조사를 거부하거나 '모른다' 라고 답한 경우 제외


### 데이터셋 구성
> 뇌졸중 위험인자 (Risk Factor)를 참고하여 데이터셋을 구성함

<details>
<summary>뇌졸중 위험인자 (Risk Factor)</summary>
<div markdown="1">

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

</div>
</details>

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

## **2. 데이터 전처리**
- 설문조사 데이터셋 중, 필요한 것만 선택하여 편집
- 각 특성에서 "모른다" 라고 답하거나, 답을 거부한 경우를 Nan으로 변경
- category가 3개 이상인 특성은 추후 시각화를 위하여 값을 String 형태로 변환함

<br>

## **3. EDA 및 Feature Engineering**

- Target 데이터의 불균등 보정
  - 모델의 Target이 되는 `Stroke`에서 압도적으로 정상 데이터(`Stroke = 0`)가 많음

    <img src="https://user-images.githubusercontent.com/77204538/175293794-ea27f32d-009c-488f-937c-d171bcceb7f2.png" width=300 height=200>

  - Oversampling 방법 (SMOTE)을 이용하여 해결하려 하였으나, XGBoost 모델 기준, AUC score가 **0.74**을 기록함
  - Class weight를 주는 방법을 사용하였을 때, AUC score 가 **0.825**를 기록하여, 최종적으로 ***Class weight를 주는 방법***을 사용함

<br>

## **4. Modeling**

### **4.1 Base 모델 선정**
- Logistic Regression, Decision Tree, Random Forest, XGBoost 분류 모델들의 성능을 비교하여, 최적의 모델을 선택
- 데이터 불균형을 고려하면서 뇌졸중 환자를 정확하게 예측해야 하므로, **AUC score**를 평가 기준으로 선택
  - AUC score는 Recall(양성 데이터 중에 양성이라 예측한 경우)과 위양성률(음성 데이터 중에 양성이라 예측한 경우)을 동시에 고려함
- `XGBoost` 모델의 AUC socre가 **0.825**로 가장 높은 성능을 나타냄
  | 분류 모델 성능 	| Logistic Regression 	| Decision Tree 	| Random Forests 	| XGBoost 	|
  |:--------------:	|:-------------------:	|:-------------:	|:--------------:	|:-------:	|
  |    Accuracy    	|         0.77        	|      0.91     	|      0.95      	|   0.74  	|
  |    Precision   	|         0.11        	|      0.11     	|      0.05      	|   0.11  	|
  |     Recall     	|         0.68        	|      0.09     	|      0.01      	|   0.77  	|
  |    AUC score   	|         0.81        	|      0.52     	|      0.76      	|   0.82  	|

<br>

### **4.2 HyperParameter tuning**
- Grid Search cross-validation 사용
  | Hyperparameter tuning 	| Tuning 전 	| Tuning 후 	|
  |:---------------------:	|:---------:	|:---------:	|
  |        Accuracy       	|    0.74   	|    0.73   	|
  |       Precision       	|    0.11   	|    0.11   	|
  |         Recall        	|    0.77   	|    0.77   	|
  |       AUC score       	|   0.8250  	|   0.8251  	|

    <img src="https://user-images.githubusercontent.com/77204538/175301556-c14456b3-5210-4869-8642-c04bf621c4b3.png" width=450 height=250>

- 성능지표 상으로는 크게 차이는 없지만, Stoke환자를 정상군으로 예측하는 경우가 감소함

<br>

### **4.3 Feature Importance**

  <img src="https://user-images.githubusercontent.com/77204538/175463176-ae10a0c0-fa05-48ac-a6fd-f4240921f609.png" width=300 height=450>

  - 뇌졸중 예측에 영향을 크게 주는 특성 TOP 5
    - 심장질환, 고혈압 여부 (`HeartDisease`, `HighBP`)
    - 연령대 (`AgeGroup`)
    - 보행장애 (`DiffWalk`)
    - 소득 (`Income`)

<br>

### **4.4 PDP plot & SHAP value**
- **PDP plot**
  - 여러 category로 이루어진 특성의 영향력을 살펴볼 수 있음
  
  <img src="https://user-images.githubusercontent.com/77204538/175465771-638927d6-8ebd-4c2f-b011-8a871033bf9e.png" width=850 height=250>

  - 연령대
    - 25~29, 35~39세에서 가장 뇌졸중 위험이 높음
    - 55세 이상부터는 연령대가 높아질 수록 뇌졸중 위험이 감소
  - 소득
    - 소득이 25000$ 이상일 때, 소득이 증가할 수록 뇌졸중 위험이 높아짐



<br>

- **SHAP value**
  - 모델 예측 결과에 대한 특성들의 영향력을 설명함
  
  <img src="https://user-images.githubusercontent.com/77204538/175463662-01edd8d7-edbc-46e3-ad40-001602a3223c.png" width=500 height=450>

  - 뇌졸중 질환에 가장 영향을 많이 끼치는 특성은 **심장질환, 보행장애, 고혈압, 연령대, 소득** 이다.
  - 심장질환, 보행장애가 있거나 정상혈압을 가지고 있을 때 뇌졸중 위험도가 상승한다.
  - 연령대는 젋은 나이일 수록 뇌졸중 위험도가 증가한다. (노인의 경우 오히려 뇌졸중 위험이 감소한다)
  - 소득이 많을 수록 뇌졸중 위험도가 증가한다.

<br>

## **5. 결론**

- 만성질환의 유무가 뇌졸중 확률을 높인다. (뇌졸중 위험정도 :심장질환 > 고지혈증 > 당뇨병)
  - 알려진 바와 다르게, 고혈압이 있으면 오히려 뇌졸중 위험도가 낮아진다
  - 뇌졸중 환자 중, 고혈압이 있는 사람이 더 많음에도 불구하고 나타남 🡪 **데이터 불균형**에 의한 가능성
- 걷는 것이 어려운 사람은 뇌졸중 확률이 높다
  - 보행장애가 생기면, 신체활동의 정도가 급격히 줄어들기 때문인 것으로 예상됨
- 청년층(25~39세) 에서의 뇌졸중 확률이 높다. 🡪 노년층에서는 확률이 낮아진다
  - 알려진 바와 다르게, 노년층이 아닌 **청년층에서의 뇌졸중 위험이 높다**
  - 전화설문이라는 방식으로 데이터가 수집 🡪 노년층에서 뇌졸중이 심한 사람들이 제외되었을 가능성이 있음
- 소득이 높을 수록 뇌졸중 확률이 높다

<br>

## **6. 한계점 및 개선사항**

### 한계점

- Target variable의 데이터 불균형
  - SMOTE와 같은 Oversampling 방법과 Target variable의 ratio를 통해 해결하고자 하였으나,
  정상군을 뇌졸중 환자로 분류한 경우가 많음
  - 데이터 불균형에 의해, 정상군/뇌졸중 환자에 많이 보이는 특성을 구분하기 어려움
- 알려진바와 다른 결과가 나타남
  - 연령대의 경우, 오히려 젊을수록 뇌졸중 위험도가 높아짐 🡪 데이터 불균형, 또는 연령대를 너무 세부적으로 나누어서 발생된 문제로 생각됨

### 개선사항
  - Down Sampling 시도
    - 정상군의 수가 20여 만명, 뇌졸중 환자가 1만 여명 이므로, 뇌졸중 환자 수에 맞추어 정상군의 수를 조정하였어도 예측 모델 생성이 가능했을 것으로 생각됨

  - 연령대 수정
    - 연령대를 20대, 30대와 같이 category를 변경할 것

<br>

## **후기**

- 데이터 불균형 문제를 해결하려 하였으나, 제대로 해결하지 못해 아쉽다.
  - Oversampling 방법과 가중치를 주는 방법을 시도하였으나, 데이터 불균형이 너무 심하여 제대로 해결되지 못한것 같다.
  - 손실될 데이터가 아까워서 Down sampling을 시도해보진 않았는데, 뇌졸중 환자 수에 맞추어도 약 2만여개의 데이터가 남아있으므로 모델을 형성하기엔 충분한 데이터였을 것 같다.
  - 의학 데이터는 데이터 수집하는 것 자체가 쉽지 않아 데이터 하나하나가 소중한데, 이러한 버릇이 아직 남아있는 것 같다. 데이터를 아까워하는 행동도 경우에 따라 어느정도 타협을 해야할 것 같다.

- PDP plot과 SHAP value를 처음 사용해보았는데, Model의 해석에 유용하다는걸 알 수 있었다.
  - Feature importance의 경우, 특성이 예측에 얼마나 영향을 주는지는 알 수 있지만, 음성 또는 양성으로 분류하는데 영향을 주는지는 알지 못한다. 
  - PDP plot과 SHAP value를 통해 뇌졸중 위험도에 영향을 끼치는 특성이 어떤 것인이 구체적으로 해석할 수 있어 편리하였다.

[[프로젝트 상세] 발표 PPT File 링크](https://drive.google.com/file/d/1h63JIlSbAJNnv0xOYYF4DaSQBdq3t2BP/view?usp=sharing)