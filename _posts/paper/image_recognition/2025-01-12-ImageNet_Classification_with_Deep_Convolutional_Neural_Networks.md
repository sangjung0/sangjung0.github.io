---
title: ImageNet Classification with Deep Convolutional Neural Networks
date: 2025-01-12 20:21:00 +0900
categories: [Paper, Image Recognition]
tags: [Paper, Machine Learning, Image, Recognition, AlexNet]
pin: false
math: true
mermaid: true
img_path: /assets/img/paper/image_recognition/ImageNet_Classification_with_Deep_Convolutional_Neural_Networks/
---


![Desktop View](/assets/img/paper/image_recognition/ImageNet_Classification_with_Deep_Convolutional_Neural_Networks/subject.png)


### Abstract
  - 모델 소개: AlexNet은 6천만개의 파라미터, 65만 개의 뉴런으로 구성되어 있으며, 5개의 Convolution Layer와 일부의 max-pooling layer 그리고 3개의 Fully-connected layer 그리고 output layer의 1000-way softmax로 이루어져 있다.  
  - 학습 방법: 훈련 속도를 높이기 위해 non-saturating neurons 사용과 효율적인 GPU를 사용 했다. 
  - 과적합 문제: 과적합 문제 해결을 위해 Dropout이라 불리는 정규화 방법을 사용했다.
  - 성과: ILSVRC-2012 대회에서 15.3%의 top-5 테스트 오류율을 기록하여 우승했다.
  

### 1 Instruction
  - 객체 인식을 위해서는 현재 머신 러닝이 필수적이다. 성능 향상을 위해서는 큰 데이터셋을 수집하고, 더 강력한 모델을 학습하며, 과적합을 방지하는 더 나은 기법을 사용해야 한다.
  - 최근까지도 라벨링된 이미지 데이터 셋의 크기는 작았다. 일반적으로 수만개(NORB, Caltech-101/256, CIFAR-10/100)의 이미지로 구성되었으며, 이는 단순한 인식에서는 데이터 증강 기법 등을 통해 좋은 성능을 보이나, 현실적인 환경의 객체 인식은 좋은 성능이 보이지 않는다. 이러한 상황에서 더 좋은 성능을 보이기 위해서는 큰 데이터셋이 필요하다. 최근에서야 수백만 개의 라벨이 지정된 이미지를 포함하는 데이터셋을 수집할 수 있게 되었다. (LabelMe, ImageNet이 있다.)

  - 수백만 개의 이미지를 학습하려면 학습 용량이 큰 모델이 필요하다. 그러나, 이미지 인식의 복잡성 때문에 이를 명확히할 수 없다. 따라서 모델은 우리가 가지고 있지 않은 데이터를 보완할 수 있도록 많은 사전 지식(이미지의 특성들이 모델에 내제화되어야 한다. 통계적 정적성이라든가 픽셀간 로컬리티 같은)이 필요하다. 이런 것을 반영한 모델 중 하나가 CNN이다. CNN은 깊이와 너비로 용량을 조정할 수 있으며, 이미지 특성에 강하고 거의 올바른 가정을 할 수 있다. 추가적으로 비슷한 크기의 Feedforward neural networks에 비해 파라미터 수와 edge가 적어 훈련이 훨씬 쉽다.

  - 현재의 GPU는 고도로 최적화된 2D convolution과 결합되어 충분한 성능을 제공한다.

  - ILSVRC-2010 및 ILSVRC-2012 대회에서 사용된 ImageNet의 하위 집합을 대상으로 현재까지 가장 큰 CNN을 훈련했고, 가장 좋은 성능을 달성했다.
  - 최종 네트워크는 5개의 컨볼루션과 3개의 Fully Connected Layer로 구성되어 있으며, 깊이가 매우 중효함을 깨달았다. 컨볼루션 레이어는 아주 적은 파라미터를 포함하지만 이를 제거하면 성능이 저하되는 결과를 보였다.

  > 이미지 인식을 효과적으로 할 수 있는 CNN을 채택하며, 높은 성능의 연산을 위하여 GPU 사용.
  이 덕분에 ILSVRC-2010 및 ILSVRC-2012 대회에서 가장 좋은 성능을 달성 함
  {: .prompt-info}


### 2 The Dataset
  - ImageNet은 22000개의 카테고리에 속하는 1500만 개 이상의 라벨이 지정된 고해상도 이미지로 구성되어 있다.
  - ImageNet은 가변 해상도로 구성되어 있으나, 고정된 해상도인 256x256로 다운샘플링 했다. 직사각형 이미지일 경우 짧은 변의 길이를 256으로 하여 중앙을 잘라 냈다. 이미지 정규화를 제외하고는 이미지 전처리는 추가적으로 시행하지 않았다.  


### 3 The Architecture

#### 3.1 ReLU Nonlinearity
![Desktop View](/assets/img/paper/image_recognition/ImageNet_Classification_with_Deep_Convolutional_Neural_Networks/figure1.png) 
_Figure 1._

  - f(x) = max(0, x)로, 전통적인 hyperbolic tangent 함수와 시그모이드 함수보다 같은 네트워크 학습에서 6배 빠르다(Figure 1). 또한 saturating nonlinearity가 아닌 non-saturating nonlinearity 이다.

#### 3.2 Training on Multiple GPUs
  - GTX 580 GPU 두 개를 사용하여 분산처리 하였다. 각 GPU에 절반의 커널(채널을 절반씩 담당하게 된다)을 배치하며, 특정 레이어에서만 통신이 이루어 지게 했다. 이러한 연결 패턴은 교차 검증의 과제가 될 수 있지만, 이 방법을 통해 계산량을 효율적으로 조정할 수 있다. (이전 레이어의 모든 파라미터를 가중치로 활용할 수 없다.)

#### 3.3 Local Response Normalization
  - ReLU는 Sigmoid와 달리 saturating이 없어, 입력값의 정규화가 필요하지 않는 특성이 있다. 그러나, ReLU를 적용한 후에 Lateral Inhibition의 정규화 방식을 사용하면 더 좋은 결과가 나오는 것을 발견했다. 

  $$ \text{Lateral Inhibition} \quad b^i_{x,y} = \frac{a^i_{x,y}}{\left( k + \alpha \cdot \sum_{j=\max(0, i-\frac{n}{2})}^{\min(N-1, i+\frac{n}{2})} \left( a^j_{x,y} \right)^2 \right)^\beta} $$

  * $ k, n, a, b $: 하이퍼 파라미터 (본 논문에서는 각각 2, 5, $ 10^{-4} $, 0.75 사용)
  * $ a^i_{x, y} $: i번째 커널의 x, y위치의 값
  * $ N $: 전체 커널 수
  * $ n $: 인접 커널 범위
  - LRN얼 적용한 후 top-1 오류율과 top-5 오류율이 각각 1.4%와 1.2% 감소하였다.
  
#### 3.4 Overlapping Pooling
  - 전통적인 풀링과 다르게 풀링을 겹쳐지게 시행했고, 풀링에서 Stride는 2, 커널 크기는 3으로 하여 적용하였다.
  
#### 3.5 Overall Architecture
![Desktop View](/assets/img/paper/image_recognition/ImageNet_Classification_with_Deep_Convolutional_Neural_Networks/figure2.png)
_Figure 2._

  - Figure 2에서 나타난 바와 같이, 네트워크는 8개의 층으로 구성되어 있으며, 처음 다섯 층은 컨볼루션 층, 나머지 세 층은 fully connected 층이다. 마지막은 1000-way 소프트맥스에 입력된다.
  - 두 번째, 네 번째, 다섯 번째 convolution layer의 커널은 동일한 GPU에 위치한 이전 커널맵과 연결 된다.
  - 세 번째 convolution layer은 두 번째 layer 의 모든 커널 맵과 연결된다.
  - fully connected layer는 이전 계층의 모든 뉴런과 연결된다.
  - 첫 번째와 두 번째 convolution layer 뒤에는 Response-normalization(LRN)이 적용된다.
  - 첫 번째와 두 번째의 LRN뒤에, 다섯 번째 레이어 뒤에 Pooling layer가 온다.
  - ReLU는 모든 convolution layer와 fully connected layer의 출력에 사용된다. 
  
  - 모델 구조 (논문에서 224로 표기되었으나, 오타이므로 정정한 값을 따라 계산 함. 논문과 값이 다를 수 있음)
    1. 입력: $ 227 \times 227 \times 3 $ 이미지를 입력 받는다. 총 $ 154,587 $ 차원이다.
      * Tensor Size: $ 227 \times 227 \times 3 = 154,587 $
      * Weights: $ 0 $
      * Biases: $ 0 $
      * Parameters: $ 0 $
    2. 첫 번째 레이어: 두 개의 GPU가 각각 $ 11 \times 11 \times \ 3 $ 구조의 커널 48개(총 96개), stride 4로 convolution 한다. 뉴런 개수는 각 $ 145,200 $개로 총 $ 290,400 $개 이다.  
      * Tensor Size: $ 55 \times 55 \times 96 = 290,400 $
      * Weights: $ 34,848 $
      * Biases: $ 96 $
      * Parameters: $ 34,944 $
    3. 첫 번째 레이어 풀링: 첫 번째 convolution layer에서는 Max Pooling이 적용되며, 커널 크기는 3, Stride는 2이다. 따라서 뉴런 개수는 각 $ 34,992 $개로 총 $ 69,984 $개 이다.
      * Tensor Size: $ 27 \times 27 \times 96 = 69,984 $
      * Weights: $ 0 $
      * Biases: $ 0 $
      * Parameters: $ 0 $
    4. 두 번째 레이어: 두 개의 GPU가 각각 $ 5 \times 5 \times 48 $ 구조의 커널 128개(총 256개)를 가지고 있다. Stride는 1, padding은 2로하여 출력 크기를 유지하였다. 뉴런 개수는 각 $ 93,312 $개로 총 $ 186,624 $개 이다.
      * Tensor Size: $ 27 \times 27 \times 256  = 186,624 $
      * Weights: $ 307,200 $
      * Biases: $ 256 $
      * Parameters: $ 307,456 $
    5. 두 번째 레이어 풀링: 두 번째 convolution layer에서는 Max Pooling이 적용되며, 커널 크기는 첫 번째 레이어에서와 같다. 따라서 뉴런 개수는 각 $ 21,632 $개로 총 $ 43,264 $개 이다.
      * Tensor Size: $ 13 \times 13 \times 256 = 43,264 $
      * Weights: $ 0 $
      * Biases: $ 0 $
      * Parameters: $ 0 $
    6. 세 번째 레이어: 세 번째는 각 GPU가 이전의 모든 커널 맵을 전달 받기에, $ 3 \times 3 \times 256 $의 커널 구조가 적용되며, Stride 1에 padding 1로 출력 크기를 유지하였다. 커널 개수는 384이다. 따라서 뉴런 개수는 각 $ 32,488 $개로 총 $ 64,896 $이다.  
      * Tensor Size: $ 13 \times 13 \times 384  = 64,896 $
      * Weights: $ 884,736 $
      * Biases: $ 384 $
      * Parameters: $ 885,120 $
    7. 네 번째 레이어: 네 번째 레이어는 384개의 커널에 $ 3 \times 3 \times 196 $의 커널 구조가 사용되며, Stride 1에 Padding 1로 출력 크기를 유지하였다. 따라서 뉴런 개수는 각 $ 32,488 $개로 총 $ 64,896 $이다.
      * Tensor Size: $ 13 \times 13 \times 384  = 64,896 $
      * Weights: $ 677,376 $
      * Biases: $ 384 $
      * Parameters: $ 677,760 $
    8. 다섯 번째 레이어: 다섯번째 레이어는 256개의 커널에 $ 3 \times 3 \times 192 $의 커널 구조가 사용되었으며, Stride 1에 padding 1로 출력 크기를 유지하였다. 따라서 뉴런 개수는 각 $ 21,632 $개 이며 총 $ 43,264 $개 이다.
      * Tensor Size: $ 13 \times 13 \times 256  = 43,264 $
      * Weights: $ 442,368 $
      * Biases: $ 256 $
      * Parameters: $ 442,624 $
    9. 다섯 번째 레이어 풀링: 다 섯번째 convoultion layer에서는 Max Pooling이 적용 되며, 구조는 이전과 같다. 따라서 뉴런 개수는 각 $ 4,608 $이며 총 $ 9,216 $개 이다.
      * Tensor Size: $ 6 \times 6 \times 256 = 9,216 $
      * Weights: $ 0 $
      * Biases: $ 0 $
      * Parameters: $ 0 $
    10. 여섯 번째 레이어: 여섯 번째는 Fully Connected layer로써, 각 2048개(총 4096개)의 뉴런을 가진다.
      * Tensor Size: $ 2048 \times 2 = 4,096 $
      * Weights: $ 37,794,736 $
      * Biases: $ 4096 $
      * Parameters: $ 37,752,832 $
    11. 일곱 번째 레이어: 일곱 번째는 Fully Connected layer로써, 각 2048개(총 4096개)의 뉴런을 가진다.
      * Tensor Size: $ 2048 \times 2 = 4,096 $
      * Weights: $ 16,777,216 $
      * Biases: $ 4096 $
      * Parameters: $ 16,781,312 $ 
    12. 여덟 번째 레이어: 여덟 번째는 Fully Connected layer로써, 총 $ 1,000 $개의 뉴런을 가진다.
      * Tensor Size: $ 1,000$
      * Weights: $ 4,096,000 $
      * Biases: $ 1000 $
      * Parameters: $ 4,097,000 $

  > sturating이 없는 ReLU, GPU 2개, LRN을 사용하였으며, Pooling은 일부 영역이 겹치게 진행 함.
  {: .prompt-info}


### 4 Reducing Overfitting

#### 4.1 Data Augmentation

  - 두가지 형태의 데이터 증강을 사용한다. 이 두가지 기법은 적은 연상으로 이미지를 변환하며, 따라서 디스크에 저장할 필요가 없다. 그러므로 CPU가 이미지를 변환할 때, GPU가 이전 배치의 이미지를 학습하는 방향으로 설계된다.
  - 첫 번째 데이터 증강은 이미지 변환 및 수평 반사를 생성하는 것이다. $ 256 \times 256 $ 크기의 이미지에서 $ 224 \times 224 $ 크기의 패치를 추출한다(각 모소리 + 중앙). 그리고 이것들을 수평 반사하여 총 10개의 데이터를 학습하며, 추론시에도 동일한 과정을 따라 예측한 후 10개의 이미지의 추론 값에 대해 평균을 내어 최종 예측 한다.
  - 두 번째 데이터 증강은 이미지 RGB 채널의 강도를 변경하는 것이다. 전체 이미지 데이터 셋의 RGB의 값의 주성분을 구하여, 평균이 0이고 표준편차가 0.1인 가우시안 분포에서 랜덤 변수를 하나 뽑아 각 이미지 별로 적용하여 훈련한다. AlexNet에서는 SVD가 아닌 공분산을 활용한 PCA 추출을 활용하여 연산하였다.

  > 특징이 적을때는 공분산을 활용한 PCA 추출이 적합할 수 있다. 일반적으로는 SVD 추출 방법을 사용한다.
  {: .prompt-tip}


#### 4.2 Dropout

  - 앙상블의 효과를 얻을 수 있는 Dropout을 설정하여, Complex codaptation 을 낮추는 동시에 향상된 정확도와 일반화 성능을 얻을 수 있다.

  > 이미지 자체의 변환과 RGB 채널의 강도 변환을 이용하여 데이터셋을 증강한다. 또한 Dropout 기법을 사용하여 더 정확한 추측과 일반화를 달성했다.
  {: .prompt-info}


### 5 Details of learning
  - 모델을 훈련할때 SGD를 사용했으며, 배치 사이즈는 128로 하였다. 모멘텀은 0.9로 하였으며 weight decay는 0.005로 하였다.
  $$ \begin{align*}
    v_{i+1} &:= 0.9 \cdot v_i - 0.0005 \cdot \epsilon \cdot w_i - \epsilon \cdot \left\langle \frac{\partial L}{\partial w} \middle| w_i \right\rangle_{D_i} \\
    w_{i+1} &:= w_i + v_{i+1}
  \end{align*}  $$
  - 각 레이어의 가중치는 평균이 0이고 표준편차가 0.01인 가우시안 분포를 따라 초기화 하였고, ReLU가 있는 레이어는 바이어스를 1로, 없는 레이어는 0으로 초기화했다.
  - 학습률은 0.01로 초기화 하였으며, 학습률 조정은 검증 에러률 증가가 멈출때마다 직접 0.1씩 기존 학습률에 곱한 값을 넣어주었다.

### 6 Results

  | Model | Top-1 | Top-5 |
  |------:|:------|:--------|
  | _Sparse coding [2]_ | _47.1%_ | _28.2%_|
  | _SIFT_ + _FVs [24]_ | _45.7$_ | _25.7%_ |
  | CNN | **37.5%** | **17.0%** |

  - ILSVRC-2010의 결과는 위와 같다.
  
  | Model | Top-1 (val) | Top-5 (val) | Top-5 (test) |
  |-----:|:-----|:-----|:-----|
  | SIFT + FVs [7]| - | - | 26.2% |
  | 1 CNN | 40.7% | 18.2% | - |
  | 5 CNNs | 38.1% | 16.4% | 16.4%|
  | 1 CNN* | 39.0% | 16.6% | - |
  | 7 CNNs* | 36.7% | 15.4%| 15.3%|

  - ILSVRC-2012 결과는 위와 같다.


#### 6.1 Qualitative Evalutions

![Desktop View](/assets/img/paper/image_recognition/ImageNet_Classification_with_Deep_Convolutional_Neural_Networks/figure3.png)
_Figure 3_

  - 그림 3은 첫번째 레이어의 커널을 시각화 한 것이다. GPU 1은 색상에 무관심하며, GPU 2는 색상에 특화된 것을 알 수 있다. 이러한 현상은 모든 실행에서 발생하며, 가중치 초기화와 독립적이다. GPU의 번호 배치와 관련있다.

![Desktop View](/assets/img/paper/image_recognition/ImageNet_Classification_with_Deep_Convolutional_Neural_Networks/figure4.png)
_Figure 4_

  - 그림 4를 보면, 맞든 틀리든 Top-5 예측에서 유의미한 클래스를 예측했다.

  - 네트워크를 시각적으로 판단하는 방법은, 최종 FC layer가 유도하는 특징 activation vector을 분석하는 것이다. 그림 4의 우측은 가장 좌측 이미지와 비교했을 때, 가장 유사한 특징 activation vector을 가지는 이미지들이다. 
  
  - 4096 차원의 실수 값 벡터를 유클리드 거리를 사용하여 유사성을 비교하는 것은 비효율적이다. auto-encoder를 사용하면 효율성을 높일 수 있다.


### 7 Discussion
  - 실험을 단순화하기 위해 비지도 학습을 사용하지 않았다. 연산력이 충분하다면, 네트워크 크기와 깊이를 크게 증가시키면서도, 데이터양을 충분히 확보하지 못할 경우에는 비지도 학습이 도움이 될 것이다.
