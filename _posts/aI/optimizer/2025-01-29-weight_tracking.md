---
title: 가중치 추적 기법들
date: 2025-01-29 15:58:00 +0900
categories: [AI, Optimizer]
tags: [AI, Weight Tracking]
pin: false
math: true
mermaid: true
---


## SMA (Simple Moving Average, 단순 이동 평균)
  - 특정한 window 크기 N에서 그 window 안에 있는 가중치의 Arithmetic Mean을 계산하는 방식이다.  
  $$ SMA_t = \frac{1}{N} \sum_{i=t-N+1}^{t} \theta_i $$
    + $ SMA_t $: 현재시점 t에서의 SMA 가중치
    + $ N $: 이동 평균을 계산할 윈도우 크기
    + $ \theta_i $: 시점 i에서의 모델 가중치
  - 장점
    + 이상치에 덜 민감하다.
    + 예측이 안정적이다.
  - 단점
    + 과거 데이터를 완전히 버린다.
    + 최신 데이터와 과거 데이터를 동등하게 취급한다.
    + 윈도우 크기 N값이 중요하다.


## EMA (Exponential Moving Average, 지수 이동 평균)
  - 가중치 업데이트를 부드럽게 만들어 학습의 안정성을 높이는 기법이다. 가중치를 구한 후에 이전 가중치들의 영향력을 일정부분 첨가하는 것이다.  
  $$ \theta_t^{EMA} = \alpha \theta_t + (1 - \alpha) \theta_{t-1}^{EMA} $$
    + $ \theta_t^{EMA} $: 현재 시점 t에서의 EMA 가중치
    + $ \theta $: 현재 시점 t에서의 실제 모델 가중치
    + $ \theta_{t-1}^{EMA} $: 이전 시점 t-1에서의 EMA 가중치
    + $ \alpha $: EMA decay factor
  - 특징
    + 첫 번째 EMA 값을 구할 때는 이전 EMA값이 없는 문제가 있다.
      * 첫 번째 값을 그대로 사용할 수 있다.
      * 초기 몇번은 EMA를 건너뛸 수도 있다.
      * 첫 번째 데이터로 대체할 수도 있다.
      * 초기 몇번은 이동 평균 없이 가중치를 구하고, 이후 SMA를 이용하여 SMA 가중치를 구한 후 그것을 초기 EMA로 할 수도 있다. 
  - 장점
    + 최신 데이터에 더 높은 가중치를 부여하기에 변화 감지가 빠르다.
    + 모든 데이터를 고려하며, 오래된 데이터의 영향력은 감소시킨다.
    + 계산량이 작고 효율적이다.
  - 단점
    + 이상치에 민감하다.
    + 감쇠 계수 $ \alpha $값 설정이 중요하다.
    + 첫 번째 EMA값이 실제 데이터와 차이가 있을 수 있다.
