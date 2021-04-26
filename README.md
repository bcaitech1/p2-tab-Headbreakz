# 부스트캠프 - AI Tech 

## P stage 2 - Tabular data classification

> `P stage 2 대회 진행 과정과 결과를 기록한 Git repo 입니다. 대회 특성상 수정 혹은 삭제된 부분이 존재 합니다`

---

### 대회 설명

​	본 대회는 `주어진 고객 데이터`를 사용하여 `특정 날의 소비량을 예측`하는 대회로 총 2주간 진행되었고, Baseline 바탕으로 Tabular data 이론에 대한 수업을 수강하면서 대회를 진행하였습니다.

 

### 목차

[📖대회 요약 정리](#📖대회요약정리)

[📂Code 설명](#📂Code 설명)

[📕대회 과정](#📕대회 과정)

* [DataSet구성](#DataSet구성)
* [Data전처리](#Data전처리)
* [DataFeature](#DataFeature)
* [UseModel](#UseModel)
* [새로운시도](#새로운시도)

[📈성공](#📈성공)

[📉실패](#📉실패)



---

---

### 📖대회요약정리

![img_1](https://github.com/bcaitech1/p2-tab-Headbreakz/blob/master/Image/img_1.png?raw=true)

![img_2](https://github.com/bcaitech1/p2-tab-Headbreakz/blob/master/Image/img_2.png?raw=true)

---

---

### 📂Code설명

[자세한 설명](https://github.com/bcaitech1/p2-tab-Headbreakz/tree/master/Code)

* EDA
  * Competition에 대한 설명
  * Competition Data 분석
* Baseline
  * Data 불러오기
  * Model (lightgbm, xgb, catboost)
  * Data Feature 추가
  * Model Train
  * Use Optuna



---

---

### 📕대회과정

#### 1. DataSet구성

 데이터는 고객 별로 2년간 기록된 데이터로, 예측에 사용하기 위해 1년 단위로 데이터를 나누었습니다.`"소비량의 차이는 있지만, 소비 패턴은 변하지 않았을 것이다"`의  가정을 가지고 Train set은 2009년 12월 ~ 2010년 11월, Test set은 2010년 12월 ~ 2011년 11월로 1년 단위로 나누어 사용하였습니다. 

  똑같은 모델과 Feature를 사용하여 Baseline에서 사용한 Train set, Test set 방식과 비교를 하였고, 더 좋은 성능을 확인하였습니다. 그리고 전체 데이터에서 상품 구입량에 따라 월 단위로 그래프를 그리게 될 경우, 연도 별로 비슷한 패턴의 구입량을 확인하였습니다.

![img_3](https://github.com/bcaitech1/p2-tab-Headbreakz/blob/master/Image/img_3.jpg?raw=true)

  

#### 2. Data전처리

​	데이터의 Product ID(상품 번호)에서 특이한 경우를 발견하였는데, 이는 실제 예측 데이터에도 충분히 나올 수 있는 경우로 생각되어 제거하지 않고, 유사한 Product ID 값(ex. 2014A, 2014B,2014C)을 통일 시켰습니다. 그리고 5000개 이상있는 Description(상품 설명)을  단어별로 분리해서 많이 등장한 단어를 중심으로 제품류 60종+ @로 바꾸었습니다.

​	 또한, 주어진 데이터에서 `Quantity(상품 수량),Price(상품 가격)은 Total(총 구매액)에 포함되어 오히려 예측에 방해가 된다고 생각하고 삭제`를 하였습니다. 



#### 3. DataFeature

​	Data feature 선택은 대회 동안 대부분의 시간을 소비하였습니다. Feature는 Baseline에서 제공한 값과  `Total 과 Time`에 대한 값을 중점적으로 사용하였습니다.

* Baseline
  * Total (고객 별 총합) - Aggregation(Mean,Sum,Std,Skew)
  * Customer ID Total Cumsum (고객 ID 별 누적 총합) - Aggregation(Mean,Sum,Std,Skew)
  * Order ID Total Cumsum (주문 ID 별 누적총합) - Aggregation(Mean,Sum,Std,Skew)
  * Order Time diff (주문 시간 차이) - Aggregation(Mean,Sum,Std,Skew)
  * Order ID nunique (주문 ID 별 횟수 )
  * Product ID nuinque (상품 ID 별 횟수)
* 추가  
  * Order Time (주문한 시간) - Aggregation(Mean,Sum,Std,Skew) ,
  * First bought month ( 처음 주문한 달) - category
  * Last bought month (마지막으로 주문한 달) - category
  * Order month Count (전체 구매한 달의 횟수)
  * Description nuinque (상품 설명 별 횟수) - 전처리 데이터

이외의 다른 Feature 경우 점수가 향상되지 않아 사용하지 않았습니다.

* 제거
  * Order Time mean diff (평균 주문 시간 차이)
  * Order Time max min diff (첫주문과 마지막 주문 차이)
  * Total sum / Order month count (총 소비량 / 주문한 달 횟수)
  * Total count / Order month count (총 주문 횟수 / 주문한 달 횟수)
  * Totla cumsum / Order month count (총 누적소비량 / 주문한 달 횟수)
  * First bought month - Last bought month (첫 구매달 - 마지막 구매달)
  * Last bought month - First bought month (마지막 구매달 - 첫 구매달)
  * Bought month diff mean (구매한 달의 평균 간격)



#### 4. UseModel

##### 	4.1 LSTM & CNN

​	데이터 분석 이후 사용한 방법은 CNN과 LSTM으로 시작을 하였습니다. 제일 익숙한 CNN 모델에서 2D conv이 아닌 1D conv를 사용하여 모델을 만들었습니다. CNN 모델에 데이터를 넣기 위해, 고객 별 월 소비 총합으로 데이터 셋을 구성하여 Input size로 (24,1)을 사용 하였습니다. 그리고 `filter = 64, Kernel size =  3, Padding = 0,stride = 1`으로 2층을 쌓고,  `AvgPool1D, fc층`으로 구성하였습니다.

​	LSTM은 CNN에서 사용한 데이터셋을 사용하여, LSTM과 GRU 사용하여 모델을 구성하였습니다. 



##### 4.2 Lightgbm

​	제일 처음에 사용하고, 최종에 사용한 Decision tree 계열 모델입니다. Decision tree 계열에 모델에서는 `모델의 구조보다 사용한 데이터의 Feature에 중점`을 두고 Baseline에 주어진 Code를 사용하였습니다. 



##### 4.3 Xgb , catboost

​	category 계열에서 catboost가 성능이 우수하다는 이야기를 듣고  Lightgbm과의 성능 비교를 위해 사용한 모델입니다. 



#### 5. 새로운시도

​	학습에 Test set을 사용하면 안되지만,  `Pseudo labeling`의  방식처럼 Test set을 예측하여 Train set에 포함시켜 모델 학습을 진행하였습니다.  예측한 Test set을 Train set에 포함 시킨 이유는 `"Decision tree 계열을 사용 할 경우, 데이터 값과 양에 따라서 다른 모델이 만들어 질 것이다"` 라고 생각 했습니다. 즉, 새로운 데이터를 넣을 경우 기존과는 다른 기준으로 모델이 형성 될 것이고, 이전과는 새로운 예측 값이 나올 것으로 예상하였습니다.

![img_4](https://github.com/bcaitech1/p2-tab-Headbreakz/blob/master/Image/img_4.png?raw=true)



#### 6. 전체 과정

![img_5](https://github.com/bcaitech1/p2-tab-Headbreakz/blob/master/Image/img_5.png?raw=true)



---

---

### 📈성공

#### 1. Feature 설정

​	어떤 Feature를 사용하는가에 따라서 성능의 큰차이를 보였습니다. `Total,Time`값을 직접적으로 사용한 Feature에 대해서는 성능 향상을 보였으며, 이 Feature를 가공하여 만든 2차적 Feature값에 대해서는 성능 향상을 보이지 않았습니다.

​	Feature 선택시에는 모델에서 측정한 Feature importance값 혹은 Feature 간의 correlation값을 사용하여 도출된 낮은 Feature에 대해서는 제거를 하지 않았습니다. 오히려 제거를 할 경우, 기존보다 안 좋은 결과를 보였으며, Feature간의 관계보다는 각각의 Feature가 가진 성격이 중요한 요소로 생각되었습니다.



#### 2. Data set Fold

​	Fold를 통해 Train set과 Valid set으로 나누어 학습을 진행시, Fold 값에 따라서 예측 값의 차이가 있는 것을 확인하였습니다. Train set과 Valid set의 비율에 따라 AUC score 차이가 생겨서 제일 높은 Out of fold 의 AUC score을 가지는 Fold값을 사용하여 모델 학습을 진행하였습니다. 

​	Fold 값에 따른  AUC score의 차이가 발생하는 이유에 대해서는 특정 지을 수 없었습니다. Fold의 값이 커진다고해서 AUC score가 일정하게 증가하거나 감소하는 것은 아니였으며, 일정한 패턴을 가지고 있다고 판단하기 어려웠습니다. 



#### 3. Optuna

​	Optuna를 사용하여 하이퍼파라미터 조정을 하였습니다. Feature가 결정이 되면서 그 이후 하이퍼 파라미터 값을 탐색하였고, 그 결과 단시간에 가장 높은 점수 향상을 보였습니다. 



---

---

### 📉실패

#### 1.LSTM & CNN

​	LSTM과 CNN모델을 사용하는데 들어간 시간에 비해 아주 좋지 않은 성적을 보였습니다. `사용된 데이터에 0의 값이 많다`는 점이  LSTM과 CNN 모델에서 가장 큰 실패 요인으로 생각하고 있습니다. 데이터의 사이즈를 줄여서 학습을 진행 할 경우, 대부분의 값이 0이 나오는 것을 확인하였습니다. 

​	모든 고객이 항상 구매를 하는 것이 아니기 떄문에 중간에 0 값을 가지는 경우가 대부분이였으며, 이는 학습이 진행되면 될수록 0으로 값을 예상했으며, 가장 최근에 사용된 값에 대해서 가장 큰 영향을 받고 값을 예측하는 것 같았습니다. 

​	결론적으로, 서로 다른 값을 모델에 넣게 되더라도 예측된 값은 비슷한 값으로 예측을 하였습니다. 



#### 2. Feature 

​	Order Time mean diff (평균 주문 시간 차이), Order Time max min diff (첫주문과 마지막 주문 차이) 경우에는 명확한 값의 차이가 있기 모델 학습에 있어서 중요한 Feature값으로 생각되었으나, 오히려 점수를 감소시켰습니다. 

​	실제 점수를 높이는데 사용된 Feature의 경우에는 실험을 통해 점수을 향상시키는 Feature로 지정되었으며,  향상 시키는 이유에 되어서는 명확한 답변을 할 수가 없습니다. Feature간의 관계나 , Featrue importance에서도 해석 결과로는 중요하다고 생각되는 Feature가 오히려 안좋고, 그 반대인 경우도 발생하기 때문에 Feature 선택에선 실험을 통해 선택하여 사용했다고 할 수 있습니다.    



#### 3. Pseudo labeling

​	1번의 점수 향상과 1번의 점수 감소를 보였습니다. 점수 감소를 보인 경우를 살펴보면,  Test set를 예측하여 다시 Test set를 예측하기 때문에  처음에 예측한 값에 따라서 모델에 큰 변화를 시킨 것 같습니다. 그리고 너무 많은 데이터를 추가 시켜서 오히려 Train set에 큰 영향을 미치게 되어 전혀 다른 모델이 만들어 진 것 같습니다.

​	추가적인 실험을 하지 않았지만, 데이터 추가에서 20~ 30%만 추가시켜서 학습을 진행 시켰으면, 점수 향상의 결과를 보였을 것으로 예상합니다.   



---

---

