---
title: 학습률 스케줄
date: 2025-01-20 20:40:00 +0900
categories: [AI, Optimizer]
tags: [AI, Optimizer, Learning Rate Schedule]
pin: false
math: true
mermaid: true
---


## Learning rate scheudule
  더 빠른 수렴을 달성하고 진동을 방지하며 바람직하지 않은 국소 최솟값에 갇히는 것을 방지하기 위해 학습률 스케줄링 하는 것


### Cosine learning rate schedule
  - learning rate를 학습 과정에서 학습 과정에서 코사인 함수의 하프 사이클($ 0 \sim \pi $) 형태로 점진적으로 감소시키는 방식  
  $$\eta_t = \eta_{\text{min}} + \frac{1}{2} (\eta_{\text{max}} - \eta_{\text{min}}) \left(1 + \cos\left(\frac{t}{T} \pi\right)\right)$$  
  - 수식은 위와 같다. 기호의 설명은 아래와 같다.
    + $ \eta $: 현재 학습률 (t번째 스탭에서의 학습률) 
    + $ \eta_{\text{max}} $: 초기 학습률 (최대 학습률, 학습 초기에 설정)
    + $ \eta_{\text{min}} $: 최소 학습률 (학습 끝날 때의)
    + $ t $: 현재 스탭 (0부터 시작)
    + $ T $: 총 학습 스탭 수
  - $ \eta_{\text{max}} $와 $ \eta_{\text{min}} $ 그리고 T는 하이퍼파라미터이다.
  - 스탭이 진행될수록 학습률이 초기에는 빠르게, 갈수록 천천히 $ \eta_{\text{min}} $으로 감소하는 것을 알 수 있다.

  - Warm Restarts와 결합  
  $$\eta_t = \eta_{\text{min}} + \frac{1}{2} (\eta_{\text{max}} - \eta_{\text{min}}) \left(1 + \cos\left(\frac{t \bmod T_i}{T_i} \pi\right)\right)$$
    + 위 식은 Warm Restarts와 결합한 형태이며, 기호의 설명은 라애와 같다.
      * $ T_i $: 반복 주기
    + $ T_i $는 하이퍼파라미터로, 스탭이 진행되면 특정 주기마다 학습률이 비선형적으로 급격하게 다시 강해지는 것을 알 수 있다.

  
### Linear Warmup
  - 초기에 학습률을 선형적으로 점진적으로 증가시키는 방법이다. 초기 학습 단계에서 학습률이 높으면 모델이 불안정할 수 있기 때문에 사용한다.  
  $$ \eta_t = \eta_{\text{max}} \cdot \frac{t}{T_w} $$
    + $ t $: 현재 스탭
    + $ T_w $: Warmup 단계가 끝나는 스탭 수
    + $ \eta_{\text{max}} $: Warmup 이후의 최대 학습률
  - 설정된 수치까지 천천히 학습률이 증가하며, 이후 스케줄은 다른 학습률 스케줄로 변경된다.
  - Pretrained 가중치를 사용하는 경우에 안정적으로 사용할 수 있다.
