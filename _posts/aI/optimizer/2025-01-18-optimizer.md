---
title: 최적화 알고리즘
date: 2025-01-18 21:34:00 +0900
categories: [AI, Optimizer]
tags: [AI, Optimizer, SGD, MBGD, BGD, Adagrad, Adadelta, RMSProp, Adam, Nadam]
pin: false
math: true
mermaid: true
img_path: /img/ai/optimizer/optimizer
---

## Optimizer
  기울기를 기반으로 최적값을 찾는 것. 알고리즘.  
  최소 제곱법 왜 안 씀? 못 쓰는 거임. 최소 제곱법은 선형적인 데이터에 사용되며, 딥러닝은 비선형 데이터를 고려하는 것. 거기다 대형 모델에서 최소 제곱법을 적용하면 많은 연산량이 필요  


### SGD (Stochastic Gradient Descent) 경사하강법
  확률적 경사하강법이라고도 하며, 매번 하나의 데이터 샘플을 사용하여 손실 함수의 기울기를 계산하고 가중치를 업데이트한다.  
  장점  
  + 빠른 업데이트
  + 노이즈를 통한 지역 최솟값 탈출
  + 메모리 효율적
  단점  
  + 진동 (노이즈로 인하여 최적값 근처에서 진동하는 것)
  + 학습 속도 느림
  + 재현성 부족 (결과가 다르다)

### MBGD (Mini-Batch Gradient Descent) 미니 배치 경사하강법
  데이터를 작은 배치로 나누어, 배치 단위로 기울기를 계산하고 가중치 업데이트한다.  
  장점  
  + 균형 잡힌 접근 (SGD와 BGD의 중간)
  + 배치 단위 병렬 처리 가능  
  + 노이즈 감소
  + 효율적인 메모리 사용
  단점  
  + 배치 크기 설정이 중요
  + 복잡성 증가
  + 배치 크기가 작으면 진동할 가능성 있음

### BGD (Batch Gradient Descent) 배치 경사하강법
  전체 데이터셋을 사용하여 손실 함수의 기울기를 계산하고 한 번에 가중치를 업데이트한다.  
  장점  
  + 안정적 수렴
  + 재현성 보장
  + 정확한 기울기 계산 
  단점  
  + 계산량 증가
  + 메모리 사용량 큼
  + 학습 속도 느림

### Adagrad (Adaptive Gradient Algorithm)
  가중치마다 학습률을 다르게 조정한다 (자주 업데이트 되면 학습률 낮게, 드물게 업데이트 되면 학습률 크게)  
  장점  
  + Sparse Data 처리에 효과적
  + 매개변수별로 학습률 조정하므로, 수렴이 빠를 수 있음.
  + 학습률 설정이 덜 민감 함.
  단점  
  + 학습률이 계속 감소하여 장기 학습에서는 성능이 저하됨
  + 기울기 업데이트가 거의 이루어지지 않는 상태에 빠질 수 있음
  + 대규모 데이터셋에서는 비효율적일 수 있음

### Adadelta
  Adagrad의 개선, 최근 기울기 정보만 고려하여 학습률을 조정한다.  
  장점  
  + Adagrad 학습률 감소 문제 해결
  + 학습률 설정 불필요
  + 학습이 장기적으로 안정적, Sparse Data에서도 효과적
  단점  
  + 복잡한 문제에서는 부족할 수 있음  
  + 학습 초기 단계에서는 Adagrad보다 느림
  + 단순한 문제에서는 성능 향상이 겅의 없음

### RMSProp (Root Mean Square Propagation)
  Adagrad의 개선, 최근 기울기를 더 강조하기 위해 지수 이동 평균(Exponential Moving Average)을 사용  
  장점  
  + 시계열 데이터와 RNN에서 매우 효과적
  + Adagrad 학습률 감소 문제 해결
  + 기울기가 큰 방향으로는 학습 속도를 줄이고, 작은 방향으로는 더 빠르게 학습
  단점  
  + 손실 함수가 매우 복잡한 경우에는 불안정할 수 있음
  + 학습률 초기 설정에 따라 성능이 크게 달라질 수 있음
  + 데이터셋에 따라 과적합이 발생할 가능성이 있음

### Adam (Adaptive Moment Estimation)
  Momentum과 RMSProp의 장점을 결합, 1차 모멘트(기울기 평균)과 2차 모멘트(기울기 제곱 평균)을 모두 사용하여 학습률 조정  
  장점  
  1. 대부분 딥러닝 문제에서 안정적이고 빠르게 수렴
  2. 학습률을 자동으로 조정
  3. 희소 데이터와 대규모 데이터 모두에 적합
  4. SGD에 비해 과적합 가능성이 낮고 수렴 속도가 빠름
  단점  
  1. 학습률 초기 설정이 여전히 결과에 영향을 줄 수 있음
  2. 일부 문제에서는 과적합을 초래할 가능성이 있음
  3. 모멘텀 매개변수 설정이 복잡한 문제에서는 민감할 수 있음

### Nadam (Nesterov-accelerated Adaptive Moment Estimation)
  Adam에 Nesterov Momentum 추가, 기울기 계산 전 미리 한 단계 앞으로 이동하여 기울기의 변화를 더 잘 반영   
  장점  
  1. Adam보다 더 빠르게 수렴하며 안정적.
  2. Nesterov Momentum 덕분에 기울기 방향을 더 정확히 예측.
  3. 대부분의 데이터셋에서 높은 성능과 빠른 학습 속도 제공
  4. Adam의 장점을 계승하면서도 더욱 효과적
  단점  
  1. 계산량이 Adam보다 약간 더 많음
  2. Nadam이 항상 Adam보다 좋은 것은 아니다
  3. 초기 학습률 설정이 여전히 중요하다