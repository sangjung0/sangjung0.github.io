---
title: 목적 함수, 최적화 함수, 비용 함수, 손실 함수
date: 2025-01-18 22:14:00 +0900
categories: [AI, Objective Function]
tags: [
    AI, Objective Function, Optimization Function, Cost Function, Loss Function
]
pin: false
math: true
mermaid: true
img_path: /img/ai/optimizer/optimizer
---

## Objective Function 목적 함수
  모델 학습의 목표를 정의하는 함수이다. 손실 함수, 비용 함수를 포함한다.  
  손실 함수와, 비용 함수는 일반적으로 혼용되어 사용된다. 그러나 엄밀히 말하자면, 손실 함수는 단일 샘플의 오류를 말하는 것이고, 비용 함수는 전체 데이터셋에 대한 비용을 말하는 것이다.  
  - Error: 실제값 - 예측값(모집단 최적 예측 함수에서의)
  - Residual: 실제값 - 예측값(표본집단 최적 예측 함수에서의) 


### MSE (Mean Squared Error) 평균 제곱 오차
  오차의 제곱의 평균  
  수식  
  $$ MSE = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 $$  
  사용: 회귀, 생성 모델  
  장점  
  + 제곱으로 인하여 오차에 더 큰 패널티 부여  
  + 계산 효율적  
  + 대부분 회귀 문제에서 사용 가능  
  단점  
  + 이상값에 민감  
  + 실제 오차 크기가 아님  


### MAE (Mean Absolute Error) 평균 절대 오차
  오차의 절대값의 평균  
  수식  
  $$ MAE = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i| $$  
  사용: 회귀(이상값이 많은 데이터셋에서 사용)  
  장점  
  + 이상값에 덜 민감  
  + 오차를 직관적으로 해석 가능  
  단점  
  + 기울기가 일정하여, 수렴 속도가 느림  
  + 미분 불연속성으로, 최적화 알고리즘에서 계산이 까다로울 수 있음  


### Huber Loss  
  임의의 Threshold에 따라 MSE 또는 MAE로 오류 계산, 작은 오차에서는 MSE처럼, 큰 오차에서는 MAE처럼 작동  
  수식  
  $$ L = \begin{cases} 
    \frac{1}{2}(y_i - \hat{y}_i)^2 & \text{if } |y_i - \hat{y}_i| \leq \delta \\
    \delta \cdot |y_i - \hat{y}_i| - \frac{1}{2} \delta^2 & \text{otherwise}
  \end{cases} $$  
  사용: 회귀(이상값이 어느 정도 있는 데이터셋에서 사용)  
  장점  
  + MSE와 MAE의 장점 결합  
  + 이상값 문제를 완화  
  단점  
  + 임계값이 적절해야 함, 데이터에 따른 튜닝 필요  
  + 계산량 많음  


### Log-Cosh Loss
  오차에 하이퍼볼릭 코사인을 취한 다음 사용 로그를 취한 합을 오류로 함, 작은 오차에서는 MSE와 유사하며 큰 오차에서는 MAE와 유사 함  
  수식  
  $$ L = \sum \log(\cosh(\hat{y}_i - y_i)) $$  
  사용: 회귀(이상값에 민감하지 않은 안정적인 회귀)  
  장점  
  + 이상값에 덜 민감  
  + Huber Loss와 비슷한 역할 하면서, 미분 가능  
  단점  
  + 계산량이 많음  
  + 특정 상황에서 MSE와 성능 차이가 크지 않음  


### Binary Crossentropy (이진 교차 엔트로피)
  엔트로피는 불확실성의 정도를 측정하는데 사용되는 함수이다. 이진 교차 엔트로피는 이것을 이용한 것이며, 두 확률 분포 간의 차이를 측정하는 것이다. 이진 교차 엔트로피는 해당 식의 오차의 합의 평균을 오류로 한다.    
  수식  
  $$ BCE = - \frac{1}{N} \sum_{i=1}^N \Big( y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \Big) $$  
  사용: 이진 분류, 생성 모델  
  장점  
  + 이진 분류 문제에서 확률 기반 예측을 효과적으로 학습  
  + 출력값이 [0, 1] 범위에 제한되므로 안정적  
  단점  
  + 잘못된 확률 값에 대해 무한대 손실이 발생할 수 있다.  
  + 클래스 불균형 문제에서는 성능 저하 가능  


### Categorical Crossentropy
  크로스 엔트로피 식을 각 클래스마다 실행하며, 각 클래스마다의 확률을 가진 일 종의 벡터로 확률을 나타내어, 각 클래스에 해당하는 확률의 평균을 오류로 함  
  수식  
  $$ CCE = - \sum_{i=1}^N \sum_{j=1}^C y_{ij} \log(\hat{y}_{ij}) $$  
  사용: 분류  
  장점  
  + 다중 클래스 분류 문제에서 확률 기반 하습에 효과적  
  + Softmax 활성화 함수와 잘 결합되어 동작  
  단점  
  + 클래스 불균형 문제에서 성능 저하 가능  
  + 예측 확률 값이 0에 가까운 경우 계산이 불안정할 수 있음  


### KL-Divergence (Kullback-Leibler Divergence)
  실제 분포에 예측 분포를 나눈 로그값에 실제 분포를 곱한 것의 합을 오류로 하는 것  
  수식  
  $$ D_{KL}(P || Q) = \sum P(x) \log\left(\frac{P(x)}{Q(x)}\right) $$  
  사용: 확률 분포 학습 모델(분류)  
  장점  
  + 두 확률 분포 간의 차이를 정량화  
  + 확률 모델 평가에 효과적  
  단점  
  + 예측 분포가 0인 경우에 계산 불가능


### Hinge Loss
  1에 예측값과 실제값의 곱을 뺀 다음, 음수면 0으로 대체한 것의 합의 평균을 오류로 하는 것  
  수식  
  $$ L = \frac{1}{N} \sum_{i=1}^N \max(0, 1 - y_i \cdot \hat{y}_i) $$  
  사용: 분류  
  장점  
  + SVM에서 효과적으로 사용 됨  
  + 마진 기반 학습으로 과적합 방지  
  단점  
  + 다중 클래스 문제에서 직접 사용이 어려움  
  + 실제 값이 -1 또는 1로 구분되어야 한다  


### Wasserstein Loss
  두 분포간 차이를 오류로 한다.  
  수식  
  $$ W(D, G) = \mathbb{E}_{x \sim P_r}[D(x)] - \mathbb{E}_{z \sim P_g}[D(G(z))] $$  
  사용: 생성 모델  
  장점  
  + GAN에서 학습 불안정 개선  
  + 두 분포 간의 거리를 직접적으로 측정  
  단점  
  + 계산량이 많음  
  + Lipschitz 조건을 만족하기 위해 추가적인 제약 필요  
