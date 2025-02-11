---
title: 컴퓨터 비전 기초
date: 2025-01-05 19:18:00 +0900
categories: [Computer Vision]
tags: [Computer Vision, Basics]
pin: false
math: true
mermaid: true
---

## Computer Vision
  [**Computer Vision**](https://ko.wikipedia.org/wiki/%EC%BB%B4%ED%93%A8%ED%84%B0_%EB%B9%84%EC%A0%84)은 기계의 시각에 해당하는 부분을 연구하는 컴퓨터 과학의 연구 분야 중 하나이다.  

  - Computer Graphics: 컴퓨터에서 만들어서 보여주는 것  
  - Computer Vision: 컴퓨터가 인식하는 것  
    + AM (Automobile): 자율주행  
    + Object detection: 특정 범위에 존재하냐  
    + Object segmentation: 정확한 범위를 찾아내는 것  
    + 생성형 데이터  
  
## 용어
  - Kernel: Mask, Filter라고도 불리며, 이미지를 처리하는 작은 행렬을 의미한다.  
  - Spatial Frequency  
    공간주파수로, 이미지나, 패턱 등 공간적 변화를 나타낸다.  
    즉, 공간을 축으로 둔다.  
    주파수가 높다라는 것은, 단위 간격동안 빠른 변화를 가지는 것이며, 주파수가 낮다라는 것은 단위 간격동안 완만한 변화를 가지는 것이다.  
  - Temporal Frequency  
    시간주파수로, 공간주파수의 반대 의미라고 할 수 있다.  
    음파, 전파, 동영상 등 신호의 시간적 변화를 나타낸다.  
    즉, 시간 축에서 주기적인 변화가 얼마나 발생하는지를 측정한 값이다.  
    높은 시간 주파수는 빠르게 변하는 신호를 말하며, 느린 시간 주파수는 천천히 변하는 신호를 말한다.  
  - Mach band (마하밴드): 색이 섞이지 않았음에도 경계가 회색처럼 보이는 것이다.  
  - False Contouring (가짜 윤곽선 현상): 주로 저해상도에서 밝기 변화가 계단처럼 불연속적으로 보이는 현상을 말한다.
  - Salt-and-Pepper Noise: 0 또는 255 값이 흩뿌려져 있는 것을 말한다.
  - Filter
    + Smooth(soft) filter: 블러 효과, 노이즈 제거로 쓰인다.  
      * Mean filter
        커널을 평균값으로 바꾼다.  
        Salt-and-Pepper Noise 문제가 극단적인 값으로 인해 발생한다.  
        edge 영역에서 blurring 효과가 발생한다.  
      * Median filter
        커널을 중앙값으로 만든다.  
        Salt-and-Pepper Noise 문제에 강하다.  
        edge 영역에서 blurring 효과가 발생하지 않는다.
    + Hard filter
      Threshold를 기준으로, 데이터를 제거하거나 유지한다. 이미지 화질을 조금 높이며, 뾰족하게 만든다.  
      * Laplacian of Caussian (LoG) Filter (라플라시안 가우시안 필터)
        이미지의 엣지 검출에 효과적이다.  
        Gaussian Filter로 이미지의 노이즈를 제거한 후, Laplacian Filter을 적용하여 이미지의 2차미분 성분을 계산한다.  
        $$ \text{LoG}(x, y) = \nabla^2 \left[ G(x,y) * I(x, y) \right] $$  
        $$ G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}} $$  
  - Morphological operation (형태학적 연산)
    + Dilation (팽창): 객체를 확장하거나 두껍게 만듦, 객체 크기 증가, 객체 간의 작은 간격 연결  
    + Erosion (침식): 객체를 축소하거나 얇게 만듬, 객체 간 연결성 분리, 얇은 선이나 작은 객체 제거  
    + Opening: Erosion 후 Dilation 수행. 노이즈 제거, 객체 경계 매끄럽게 함, 객체 분리  
    + Closing: Dilation 후 Erosion 수행. 작은 구멍 메우기, 객체 경계 연결  
  - Convolution (합성곱)  
    연속적이지 않은 것의 넓이를 구할 때, Convolution을 한다.  
    식: $ (f * g)(t) = \int_{-\infty}^\infty f(\tau) g(t - \tau) \, d\tau = = \sum_{\tau} f(\tau) \cdot g(t - \tau) $  
    계산 예시
      $$ \begin{align*}
        &\textbf{0. 가정} \\
          & \quad 
          \begin{aligned}
            &\text{입력 신호: } \quad f(\tau) = [1, 2, 3, 4] \\
            &\text{필터: } \quad g(\tau) = [1, 0, -1] \\ 
            &\text{시작 위치: } \quad t = 0 \\
          \end{aligned}
          \\[10pt]
        &\textbf{1. 시간 축 반전:} \quad t - \tau \text{는 시간 축 반전을 의미한다. 따라서 필터를 뒤집어 준다.} \\
          & \quad g(-\tau) = [-1, 0, 1] 
          \\[10pt]
        &\textbf{2. 유효한 슬라이딩 위치 수 계산} \\
          & \quad 
          \begin{aligned}
            \text{슬라이딩 거리} &= \text{입력 신호 길이} - \text{필터 길이} + 1 \\
                                  &= 4 - 3 + 1 = 2
          \end{aligned}
          \\[10pt]
        &\textbf{3. 첫 번째 계산:} \quad \boldsymbol{\tau = 0} \\
          & \quad 
          \begin{aligned}
            f(0)g(0) + f(1)g(1) + f(2)g(2) &= (1 \cdot -1) + (2 \cdot 0) + (3 \cdot 1) \\
                                            &= -1 + 0 + 3 \\
                                            &= 2
          \end{aligned}
          \\[10pt]
        &\textbf{4. 두 번째 계산:} \quad \boldsymbol{\tau = 1} \\
          & \quad 
          \begin{aligned}
            f(1)g(0) + f(2)g(1) + f(3)g(2) &= (2 \cdot -1) + (3 \cdot 0) + (4 \cdot 1) \\
                                            &= -2 + 0 + 4 \\
                                            &= 2
          \end{aligned}
          \\[10pt]
        &\textbf{5. 최종 결과} \\
          & \quad \text{Result} = [2, 2]
      \end{align*} $$
  - Cross-Correlation (Cross-Convolution)  
    두 신호의 유사성을 알아보기 위하여 탄생했다.  
    Convolution 연산과 유사하다. g 함수는 필터 또는 다른 연속 함수가 될 수 있다.  
    식: $ (f \star g)(t) = \int_{-\infty}^\infty f(\tau) g(t + \tau) \, d\tau $
  - Discrete Convolution (이산 합성곱)  
    디지털 신호 처리를 위하여 탄생 했다.  
    계산 과정은 일반 합성곱과 유사하다.  
    식: $ (f * g)[n] = \sum_{k=-\infty}^\infty f[k] \cdot g[n - k] $  
  - Linear Time-Invariant System (선형시불변 시스템)  
    입력에 따른 소요 시간이 일정한 것을 말한다.  
    자료구조에서 Time complexity에서 O(n)을 만족하는 자료구조를 생각하면 이해가 쉽다...  
  - Laplace Transform (라플라스 변환)
    시간 영역에서 정의된 함수를 주파수 영역으로 변환하는 도구이다.  
    식: $ F(s) = \mathcal{L}\{f(t)\} = \int_{0}^{\infty} f(t) e^{-st} \, dt $  
  - Furier & Laplace transform (FT, 퓨리에 변환)  
    시간에서 공간, 공간에서 시간으로 변환 가능하다.  
    Fast FT도 있다.  
    Furier 식: $ F(\omega) = \mathcal{F}\{f(t)\} = \int_{-\infty}^{\infty} f(t) e^{-j\omega t} \, dt $  
    Furier 와 Laplace 관계식: $ F(\omega) = \mathcal{F}\{f(t)\} = \mathcal{L}\{f(t)\} \big|_{s = j\omega} $  

    