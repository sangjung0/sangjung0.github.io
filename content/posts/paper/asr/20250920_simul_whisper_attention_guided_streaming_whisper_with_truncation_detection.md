---
title: "Simul-Whisper: Attention-Guided Streaming Whisper with Truncation Detection 번역"
subtitle: ""
draft: false
date: 2025-09-20 15:36:49 +0900
categories: [Paper, Translation, ASR]
tags: [Paper, Machine Learning, ASR, Whisper]
math: true
mermaid: true
image:
  path: /images/paper/asr/simul_whisper_attention_guided_streaming_whisper_with_truncation_detection/Subject.png
---


{{< figure src="/images/paper/asr/simul_whisper_attention_guided_streaming_whisper_with_truncation_detection/Subject.png" alt="subject" class="center" width="80%" >}}

## Abstract

강건하고 대규모 다국어 음성 인식 모델인 Whisper는 다양한 low-resource하고 out-of-distribution 시나리오에서 인상적인 결과를 보여줬다.
하지만, Whisper의 인코더-디코더 구조는 실시간 음성 인식에 적용하기 어렵다.
본 논문에서는 Simul-Whisper을 소개한다.
Simul-Whisper는 auto-regressive decoding을 유도하기 위해  Whisper의 cross-attention 내의 시간 정렬을 사용하고, 청크 기반 streaming ASR을 fine-tuning 없이 달성한다.
또한, 우리는 디코딩 결과에서 청크 경계 절단 단어의 부정적인 효과를 관측했고, 이를 해결하기 위해 integrate-and-fire-based truncation detection model을 제안한다.
다국어 언어와 Whisper 아키텍처에서 수행한 실험은 Simul-Whisper가 청크 사이즈 1초에서, 평균 절대 WER 저하 1.46%를 달성한것을 보여준다.
이는 현재 SOTA baseline을 능가한다.  
*Index Terms: streaming speech recognition, Whisper, decision policy, truncation detection*

## 1. Introduction

대규모 데이터셋의 사전 훈련은 자동 음성 인식(ASR)에서 상당한 발전을 가져왔다.[^1] [^2]
최근에, 680,000시간의 대규모 데이터셋에서 훈련 받은 weakly-supervise된 encoder-decoder 모델 Whisper[^3]은 다양한 언어와 복잡한 개방형 환경에서 놀라울 정도로 강건한 성능을 보였다[^4] [^5] [^6] [^7].
그러나, Whisper은 streaming ASR에 대해서 사전 훈련이 수행되지 않았으며, 회의 전사, 동시 통역, 실시간 스트리밍과 같은 시나리오에서 적용되는데 장애물이 된다.
이러한 맥락에서, Whisper을 Streaming ASR에 활용하는 것은 매력적인 제안이 된다.

오프라인 ASR과 비교했을 때, 추론 동안의 전체 문맥을 활용 못하기 때문에 streaming ASR은 더 도전적이다.
Transformer 기반 스트리밍 ASR 모델은 통상적으로 시간 제한적[^8], 청크 기반[^9], 메모리 기반[^10] 어텐션을 사용한다.
하지만, 스트리밍 어텐션 매커니즘을 사용하기 위해 사전 훈련된 파라미터들을 수정하는 것은 매우 복잡한 자원[^11]을 요구한다.
게다가, HuBERT[^12] 또는 WavLM[^13]과 같은 인코더만 사전훈련된 모델과 비교해서 Whisper 모델은 encoder-decoder 구조 때문에 streaming ASR에 사용할 때 추가적인 어려움에 직면한다.
encoder-decoder 구조에서, 인코더는 입력 오디오 전체를 잠재 표현 변환하고, 이후에 디코더는 "\<eos\>" 토큰이 출력되기 까지 반복적인 auto-regressive 디코딩을 시작한다.
Streaming ASR에서, 모델의 입력은 종종 고정된 길이의 짧은 세그먼트로 잘려있다.
이 무작위적인 잘림은 전사의 마지막 부분에서 신뢰할 수 없는 출력으로 이끌 수 있다.
그리고 종종 attention에 실패하여, 의미없는 반복 출력을 생성하는 경우도 발생한다.
이러한 오류들은 경계 이전에서 디코딩을 멈추는 것으로 피할 수 없다.
왜냐하면 encoder-decoder cross-attention이 비단조 특성을 가지므로, 출력 토큰들에 대응되는 시작과 끝 타임스탬를 찾을 수 없다.

이 문제를 해결하기 위해, Macháček et al.는 Whisper에서 Local Agreement[^14] 정책을 제안했다.
이들은 청크 기반의 추론 접근법을 사용했는데, 즉 오디오 청크가 들어올때마다 Whisper가 non-streaming 디코딩을 수행하고, 최종 전사는 이전 청크 결과의 가장 긴 공통 접두사를 기준으로 결정했다.
이 기법은 청크 경계에서의 신뢰할 수 없는 전사를 완화하면서도 Whisper의 파라미터를 수정하는 것을 피한다.
하지만, 모델이 오직 가장 긴 접두사가 확장될 때만 전사하기 때문에 latency를 예측하기 어렵다.
추가적으로, 반복적으로 non-streaming 디코딩을 수행하기 때문에 계산량이 매우 크다.

encoder-decoder 모델의 Streaming 추론은 음성 번역 분야에서도 중요한 주제이다[^15] [^16] [^17].
최근에, 몇몇 연구들은 음성 번역에 직접적으로 non-streaming 모델을 사용하는 것을 제안했으며[^18] [^19] [^20] [^21], 스트리밍 학습을 별도로 수행한 모델과 비교해도 근사하거나 더 나은 성능을 달성했다.
이러한 연구들은 적절히 학습된 encoder-decoder 모델 내의 거의 단조적인 cross-attention에 주목하여, 디코딩을 언제 멈출지 제어할 수 있는 방법을 고안했다.
예를 들어, EDATT[^22]는 마지막 몇 오디오 프레임에 대한 어텐션이 특정 임계치에 가까워졌을 때 디코딩을 멈춘다.
반면, AlignAtt[^19]는 가장 attention된 지점을 추적하거나, 그것이 오디오 끝부분에 충분히 가까워지면 디코딩을 중단한다.
Local Agreement 기법과 비교하여, 이러한 종류의 기법은 더 적은 계산량을 요구하고 더 나은 디코딩 latency를 가진다.
Whisper의 cross-attention은 좋은 시간적 정렬을 보여주고, 이러한 fine-tuning 없는 기법이 Whisper의 streaming 추론에도 적용할 수 있을 것이라 기대한다.

{{< figure src="/images/paper/asr/simul_whisper_attention_guided_streaming_whisper_with_truncation_detection/Figure1.png" alt="Figure1" title="Figure1" class="center" width="50%" caption="Figure 1: 제안하는 기법의 개요. 서로 다른 색은 오디오 청크와 그에 대응하는 전사를 나타낸다. cross attention 행렬에서 더 어두운 블록은 현재 토큰이 가장 집중하고 있는 오디오 프레임을 의미한다. 상단: 가장 집중된 오디오 프레임이 청크 경계에서 나타날 때 디코딩을 중단한다. 하단: 잘림이 감지되면, 신뢰할 수 없는 마지막 단어를 전사에서 삭제하고 모델을 다음 청크가 도착할때까지 대기한다">}}

그러나, 이러한 기법의 문제들은 cross-attention이 디코더에게 주로 정보를 제공하는 것이다.
그림 1을 보듯, 청크 경계에서 절단된 단어는 잘못된 전사로 이끌 수 있고, 오직 디코더 정보에만 의존하여 이러한 상황을 충분히 구별하긴 어렵다.
신뢰할수없는 전사를 제외하기 위하여, 인코더의 정보는 매우 중요하다.
최근에, Dong et al.[^23]과 Zhang et al.[^24]는 단어 경계를 탐지하기 위해 integrate-and-fire (IF) 모델[^25] [^26] [^27]을 제안했다.
이와 유사한 기법을 단어 잘림 감지에도 적용할 수 있다.
다시 말해, 단어 경계가 청크 경계에서 탐지되지 않았을 때, 이는 단어가 잘려 나간 것과 동일하다고 볼 수 있다.

본 논문에서는, Simul-Whisper을 제안한다.
이는 fine-tuning이 필요없는 Whisper의 실시간 추론 전략이다.
cross-attention과 잘림 탐지 모듈로부터 encoder와 decoder의 정보를 통합하는 것으로 디코딩 과정을 추적하고, 적절한 시점에 디코딩을 멈추고, 신뢰할 수 없는 전사를 제거할 수 있도록 했다.
우리의 실험은 다국어 언어와 Whisper 아키텍처에서 시연되었다.
실험 결과 제안된 기법은 청크 사이즈 1초에서 최소 절대 성능 감소 0.77%를 streaming 추론에서 달성했다.
이는 SOTA 기준인 Local Agreement를 크게 능가했다.

## 2. Methods

이 절에서는, fine-tuning 없이 non-streaming으로 Whisper를 활용하여 streaming 추론을 수행하는 방법을 소개한다.
먼저, 대략적인 순서를 얻고 적절한 시점에서 디코딩을 중단하기위해 Whisper의 cross-attention 메커니즘을 활용한다.
다음으로, 청크 경계에서 신뢰할 수 없는 전사를 제거하기 위해 Integrate-and-Fire (IF) 메커니즘을 기반으로한 Truncation Detection Module (TDM)을 도입한다.

### 2.1. Attention-Guided Decoding Policy

Whisper는 여러가지 모델 아키텍처를 가지고 있으며, 이들 모두 2-layer CNN과 multi-layer 트랜스포머 인코더와 디코더로 구성되어 있다.
encoder-decoder attention은 다음과 같이 계산된다.

$$
Q = X_t W^{i}_{Q} \tag{1}
$$

$$
K = X_a W^{i}_{K} \tag{2}
$$

$$
S^{i} = \mathrm{softmax}\left(\frac{Q K^{\top}}{\sqrt{d_{K}}}\right), \tag{3}
$$

여기서 $ X_a \in \mathbb{R}^{N_a \times D_a} $ 는 인코딩된 오디오이고, $ X_t $ 는 디코더의 이전 출력값을 의미한다.
$ N_a $ 와 $ D_a $ 는 각각 인코딩된 오디오의 길이와 은닉 차원이다.
이와 유사하게, $ N_t $ 와 $ D_t $ 는 디코더 출력의 길이와 차원을 나타낸다.
$ S^i \in \mathbb{R}^{N_t \times N_a} $ 는 multi-head cross-attention 모듈의 $ i $ 번째 어텐션 헤드 출력이다.

Whisper의 대규모 weakly supervised training 과정에서, cross-attention 모듈에서 몇 attention 헤드들은 시간 정렬에서 유리한 특성을 보인다.
구체적으로 말하자면, attention 행렬의 $t$-번째 행 $s_t^i$를 보면, 토큰 $t$가 인코딩된 오디오 $X_a$에 대해 강하게 attention하는 구간이 실제 오디오와 상당히 겹치는 경우가 많다.
이러한 attention 헤드들은 수동으로 선택되며, 이를 alignment head라고 부른다.
Whisper가 생성하는 타임스탬프는 실제로 이 alignment heads에 대해 Dynamic Time Warping (DTW)을 적용하여 얻어진다.
aliment 행렬의 결과는 다음과 같이 표현될 수 있다.

$$
S = f_{m}\left(\sum_{i\in H} s^{i}\right), \tag{4}
$$

여기서 $f_n$은 창 크기 7의 중앙값 필터이고, $H$는 aliment head의 집합을 의미한다.
디코딩 과정을 안내하기 위해, alignment head로부터 현재 토큰이 오디오 상에서 집중하는 위치를 찾아야 한다.
Papi et al.의 아이디어를 참고하여, 우리는 최대 어텐션이 발생하는 위치를 기준점으로 사용한다.
구체적으로, 디코딩 과정은 다음이 성립할 때 종료된다.

$$
N_a - \arg\max(S_t) < l, \tag{5}
$$

이는 잘못되거나 반복적인 토큰을 피하기 위함이다.
여기서 $l$은 사전에 정의된 임계값이다.
DTW와 비교하여, 이러한 접근법은 각 토큰을 독립된 단위로 취급하여 auto-regressive decoding 과정에서 누적 오류를 줄여준다.
또한 최대값을 선택하는 방식은 모델의 무작위성으로 인해 발생하는 잡음을 줄여준다.
따라서 이 방법은 학습과 테스트 환경이 크게 불일치하는 스트리밍 추론 과정에서 더 적합하다.

### 2.2. IF-Based Truncation Detection Module

모델에 입력되는 고정 길이의 오디오 청크는 불완전한 발화 단위가 포함될 수 있으며, 이는 신뢰할 수 없는 전사로 이어진다.
더 나아가, 이러한 에러는 누적되며 전체 문장에 영향을 미친다.
이 문제를 다루기 위해, 우리는 IF 기반의 truncation detection module을 설계했다.
만약 디코딩 과정에서 잘림이 감지 되지 않는다면, 해당 토큰은 유지된다.
반대로 잘림이 감지되면, 신뢰할 수 없는 토큰은 제거되며, 완전한 단어가 다음 청크에서 입력될 때 다시 생성한다.

IF 뉴런은 신호를 지속적으로 받아들이고 통합한다.
누적된 신호가 특정 임계값을 초과하면, 뉴런은 발화한다.
다시 말해서, 출력을 만들어내고 누적값을 초기화 한다.
제안한 TDM에서 신호는 Whisper의 인코더 출력을 선형 계층과 시그모이드 함수를 거쳐 생성된다.
학습 목표는 IF 뉴런의 발화 횟수가 오디오 내 단어 수와 일치하도록 하는 것이다.
추론 동안, 만약 뉴런이 오디오의 끝에 발화하지 않으면 잘림이 탐지된다.
주어진 IF 임계값 $f$, 신호 시퀀스 $ a \in \mathbb{R}^{N_a} $, 누적 신호 $I$에 대해 TDM은 IF 뉴런이 마지막으로 발화한 위치 $p$를 기록하여 잘림을 감지한다. 이 과정은 아래 연산에 의해 정의된다.

{{< figure src="/images/paper/asr/simul_whisper_attention_guided_streaming_whisper_with_truncation_detection/Algorithm1.png" alt="Algorithm1" title="Algorithm1" class="center" width="50%">}}

Whisper는 추론 과정에서 입력을 30초로 패딩하기 때문에, 인코더 특성의 마지막 프레임은 실제로 음성 내용에서 무음 패딩으로 넘어가는 전환 지점이다.
이 지점에서 IF 뉴런은 항상 발화된다.
따라서, IF 결과의 정확성을 보장하기 위해, 우리는 인코더 특성의 마지막 프레임은 제거한다.

## 3. Experimental Settings

### 3.1. Data

제안 기법은 LibriSpeech 데이터셋[^28] 과 네덜란드어(nl), 프랑스어(fr), 폴란드어(pl), 독일어(de), 이탈리아어(it), 포르투칼어(pt), 스페인어(es) 7개 언어로 구성되어 있는 다국어 LibriSpeech (MLS) dataset [^29]에서 평가되었다.
LibriSpeech 데이터셋에서는 test-clean과 dev-clean, test-other, dev-other을 평가 대상으로 사용했고, MLS에서는 test 셋만 사용했다.

### 3.2. Truncation Detection Training Setup

TDM의 objective로는 RMSE(root mean square error)를 사용했다.
훈련 데이터로는 100시간 분량의 Librispeech train-clean 일부를 사용했다.
최적화에는 Adam 옵티마이저를 사용했으며, 학습 초기에 불안정한 동작을 방지하기 위해 warm-up을 적용했다.
TDM은 총 10 epoch 동안 학습되었으며, 배치 크기는 4,500 MFCC 프레임 이다.
처음 3 epoch는 워밍업 단계로 학습률을 $0$에서 $1\times10^{-6}$까지 선형적으로 증가시켰다.
훈련은 24GB 메모리를 탑재한 NVIDIA GeForce RTX 3090 GPU 1개로 수행되었다.

### 3.3. Inference and Evaluation

생성된 전사와 정답 레이블은 Whisper의 오픈 소스 텍스트 정규화 도구를 이용해 정규화 했으며, 파이썬 라이브러리 editdistance[^30]를 WER 평가를 위해 사용했다.
attention-guided decoding에서 threshold $l$은 12 frames(240ms)로 설정하였으며, TDM의 발화 threshold $f$는 0.999로 설정했다.
베이스라인으로는, [오픈 소스 코드](https://github.com/ufal/whisper_streaming)을 기반으로 Local Agreement 정책을 재현 하였고, 원 논문에 설명된 실험 설정을 따랐습니다.
이 설정에서는 $n=2$개의 연속된 최장 공통 접두사를 사용합니다.
지연 평가는 Differentiable Average Lagging (DAL)[^31]을 사용합니다. 모든 추론은 메모리 24GB의 NVIDIA GeForce RTX 3090 GPU 한 대에서 수행됩니다.

## 4. Results

{{< figure src="/images/paper/asr/simul_whisper_attention_guided_streaming_whisper_with_truncation_detection/Table1.png" alt="Table1" title="Table1" class="center" width="50%" caption="Table 1: 청크 길이 1초에서 Librispeech 와 MLS 데이터셋에서 스트리밍 디코딩 정책의 WER (%) 이다. $\bar{\delta}$는 평균 성능 감소이다. 제안된 기법은 최소 성능 감소 0.09%를 달성했으며, baseline인 Local Agreement를 능가했다. TDM은 거의 모든 모델 아키텍처와 데이터셋에 긍정적인 영향을 보여줬다.">}}

표 1은 스트리밍 청크 길이가 1초일 때, Librispeech와 MLS 데이터셋에서의 오프라인 디코딩과 스트리밍 디코딩 정책에서의 WER을 보여준다.
실험에서, 제안 기법은 0.09%의 최소 절대 성능 감소 스트리밍 디코딩을 달성했다.
전반적으로, Medium과 Large 아키텍처는 Base와 Small 아키텍처보다 적은 성능 감소를 보여준다.
더 나아가, Librispeech test-clean과 dev-clean 데이터셋에서 스트리밍 WER은 보통 MLS보다 낮게 나타나는 경우가 많았다.
이러한 관찰은 스트리밍 디코딩에서 성능 감소가 아마 모델의 초기 성능과 관련되어있음을 암시한다.

우리는 또한 평균 성능 저하율 $\bar{\delta}$를 비교했다.
우리가 제안한 기법은 baseline과 비교하여 특히 적은 $\bar{\delta}$를 보여줬다.
우리는 Local Agreement 정책을 사용한 전사에서 종종 삽입 에러와 삭제 에러가 문장 끝에서 발생한다는 것을 관찰했다.
이는 이 정책의 문맥 관리 방식과 관련이 있을 수 있습니다.
Local Agreement는 다양한 길이의 오디오 문맥을 유지하며, 마침표 등의 구분자를 출력하면 Whisper의 타임스탬프를 기준으로 해당 위치에서 버퍼를 잘라낸다.
그러나, 이 타임스탬프가 충분히 정확하지 않으면 오류가 발생할 수 있다.
반면, attention-guided 정책은 정확한 타임스탬프를 필요로 하지 않는다.
우리는 이전 오디오 구간과 해당 전사 결과를 보존한 뒤, 이를 큐에 삽입합니다.
보존된 문맥의 길이가 일정 임계값을 초과하면, 큐의 맨 앞에 있는 오디오 및 전사 결과를 제거합니다.
비록 남아있는 전사가 현재 오디오보다 약간 앞설 수 있지만, 이 추가적인 문맥은 잘못 잘린 오디오보다 부정적인 영향을 적게 가진다.
또한 우리는 이 문맥을 디코더에 조건 입력으로 제공하여, Local Agreement에서 프롬프트로 문맥을 사용하는 것보다 더 나은 디코딩 제어가 가능하다.

게다가, 제안된 IF 기반 TDM은 대부분의 모델 아키텍처와 언어에서 정확도 향상에 긍정적인 효과를 보여주었으며, 이는 제안된 방법의 일반화 능력을 시사한다.
특히 Whisper의 Medium 및 Large 모델에서, Simul-Whisper은 Librispeech의 test-clean 및 dev-clean 데이터셋에서 유사한 성능을 보여주었으며, 이는 해당 데이터셋에서의 성능 저하가 TDM이 없을 경우 잘린 단어들에 의해 주로 발생함을 나타낸다.

우리는 스트리밍 정책에서 레이턴시를 추정하기 위해 Differentiable Average Lagging (DAL)을 사용했다.
DAL은 모든 토큰에 대해 평균을 낸, 이상적인 스트리밍 시스템에 비해 상대적인 지연이다.
입력 오디오의 길이를 $N_a$, 출력 토큰 수를 $N_t$라고 가정하면, 이상적인 스트리밍 정책은 매 $d = \frac{N_a}{N_t}$초마다 하나의 토큰을 생성한다고 볼 수 있다.
토큰 $t$가 시간 $g(t)$에 생성된다고 가정할 때, DAL은 다음과 같이 계산된다.

$
g'_d(t) = \begin{cases} g(t) & \text{if } t = 1 \max\left(g(t),\, g'_d(t - 1) + d\right) & \text{if } t > 1 \end{cases}
$

$
\mathrm{DAL} = \frac{1}{N_t} \sum_{t=1}^{N_t} \left( g'_d(t) - (t - 1)d \right)
$

함수 $g(t)$를 $g'_d(t)$로 조정함으로써, 의미 없는 음수값을 피할 수 있습니다.
우리는 computation-unaware 지연과 computation-aware 지연 두 가지 방식으로 지연을 평가합니다.
computation-unaware latency는 연산 시간을 무시하고 청크를 받자마자 즉시 출력이 가능하다고 가정한다.
반면에 computation-aware latency는 모델의 하드웨어 연산 시간을 포함한다.

그림 2는 0.5~1.0초 까지의 다양한 청크 길이와 함께 Simul-Whisper와 baseline 모델들의 latency를 보여준다.
이 그림에서는 같은 지연시간 기준에서 제안된 모델의 WER이 Local Agreement 보다 상당히 낮다는 것을 볼 수 있다.
Local Agreement는 최장 공통 접두사가 확정 될때만 토큰을 생성하기 때문에 높은 레이턴시를 가진다.
또한, 레이턴시는 청크 길이로 직접적으로 조정될 수 없다.
반면에, Simul-Whisper의 레이턴시는 일반적으로 청크 길이의 1~2배 정도이다.
이는 대부분의 토큰이 현재 청크 이후 또는 다음 청크까지 기다렸다가 생성되기 때문이다.
최종적으로, TDM은 연산 시간을 약간 증가시키지만, 동일한 computation-aware latency에서 더 낮은 성능 저하를 달성한다.

## 5. Conclusions and Discussions

본 논문에서, 우리는 사전 학습된 Whisper 모델을 추가적인 파인튜닝 없이 사용할 수 있는 스트리밍 디코딩 정책인 Simul-Whisper에 대해 소개했다.
우리는 cross-attention을 활용하여 디코딩 과정을 유도하며, 청크 경계에서의 신뢰할 수 없는 전사를 방지하기 위해 TDM을 학습시켰다.
여러 데이터셋에 대한 실험 결과, 제안된 방법은 1초 청크 길이에서 평균 1.46%의 성능 저하만으으로 스트리밍 ASR을 구현할 수 있음을 보여줬다.

하지만 Simul-Whisper에는 여전히 일부 문제점이 존재한다.
파인튜닝 없는 방식을 지향하기 때문에, Whisper의 입력 요건을 만족시키기 위해 입력을 30초로 패딩해야 한다.
이로 인해 computation-aware latency가 증가하게 된다.
향후 연구에서는 self-distillation 등의 방법을 통해 이 문제를 해결하는 방안을 탐색할 예정이다.

[^1]: A. Mohamed, H.-y. Lee, L. Borgholt, J. D. Havtorn, J. Edin,C. Igel, K. Kirchhoff, S.-W. Li, K. Livescu, L. Maaløe, T. N.Sainath, and S. Watanabe, “Self-supervised speech representation learning: A review,” IEEE J. Sel. Topics Signal Process., vol. 16,no. 6, pp. 1179–1210, 2022.
[^2]: J. Zhao and W.-Q. Zhang, “Improving automatic speech recognition performance for low-resource languages with self-supervised models,” IEEE J. Sel. Topics Signal Process., vol. 16, no. 6, pp.1227–1241, 2022.
[^3]: A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, andI. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proc. Int. Conf. on Mach. Learn., 2023, pp. 28 492–28 518.
[^4]: S. Radhakrishnan, C.-H. Yang, S. Khan, R. Kumar, N. Kiani,D. Gomez-Cabrero, and J. Tegn ́er, “Whispering LLaMA: A cross modal generative error correction framework for speech recognition,” in Proc. Conf. Empirical Methods Natural Language Process., 2023, pp. 10 007–10 016.
[^5]: Y. Gong, S. Khurana, L. Karlinsky, and J. Glass, “Whisper-AT:Noise-robust automatic speech recognizers are also strong generalaudio event taggers,” in Proc. INTERSPEECH, 2023, pp. 2798–2802.
[^6]: R. Ma, A. Liusie, M. J. F. Gales, and K. M. Knill, “Investigating the emergent audio classification ability of ASR foundation models,” arXiv:2311.09363, 2024.
[^7]: S. Rathod, M. Charola, and H. A. Patil, “Noise robust whisper features for dysarthric severity-level classification,” in Proc. Pattern Recog. Mach. Intell., 2023, pp. 708–715.
[^8]: Q. Zhang, H. Lu, H. Sak, A. Tripathi, E. McDermott, S. Koo, and S. Kumar, “Transformer transducer: A streamable speech recognition model with transformer encoders and RNN-T loss,” in Proc.IEEE Int. Conf. Acoust. Speech Signal Process., 2020, pp. 7829–7833.
[^9]: Z. Tian, J. Yi, Y. Bai, J. Tao, S. Zhang, and Z. Wen, “Synchronous transformers for end-to-end speech recognition,” in Proc. IEEE Int. Conf. Acoust. Speech Signal Process., 2020, pp. 7884–7888.
[^10]: Y. Shi, Y. Wang, C. Wu, C.-F. Yeh, J. Chan, F. Zhang, D. Le, and M. Seltzer, “Emformer: Efficient memory transformer based acoustic model for low latency streaming speech recognition,” inProc. IEEE Int. Conf. Acoust. Speech Signal Process., 2021, pp.6783–6787.
[^11]: Y. Fu, Y. Kang, S. Cao, and L. Ma, “DistillW2v2: A small and streaming wav2vec 2.0 based ASR model,” arXiv:2303.09278, 2023.
[^12]: W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “HuBERT: Self-supervised speech representation learning by masked prediction of hidden units,”IEEE/ACM Trans. Audio, Speech, Language Process., vol. 29, pp.3451–3460, 2021.
[^13]: S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, N. Kanda, T. Yoshioka, X. Xiao et al., “WavLM: Large-scale selfsupervised pre-training for full stack speech processing,” IEEE J.Sel. Topics Signal Process., vol. 16, no. 6, pp. 1505–1518, 2022.
[^14]: D. Mach ́aˇcek, R. Dabre, and O. Bojar, “Turning whisper into realtime transcription system,” in Proc. Int. Joint Conf. on Natural Language Process. Conf. Asia-Pacific Chapter Assoc. for Comput. Linguistics: Syst. Demonstrations, 2023, pp. 17–24.
[^15]: A. B ́erard, O. Pietquin, C. Servan, and L. Besacier, “Listen and translate: A proof of concept for end-to-end speech-to-text translation,” in Proc. NIPS Workshop End-to-End Learni. Speech Audio Process., 2016.
[^16]: R. J. Weiss, J. Chorowski, N. Jaitly, Y. Wu, and Z. Chen, “Sequence-to-sequence models can directly translate foreign speech,” in Proc. INTERSPEECH, 2017, pp. 2625–2629.
[^17]: X. Ma, J. Pino, J. Cross, L. Puzon, and J. Gu, “Monotonic multihead attention,” in Proc. Int. Conf. Learn. Representations, 2020, pp. 27-41
[^18]: S. Papi, M. Gaido, M. Negri, and M. Turchi, “Does simultaneous speech translation need simultaneous models?” in Findings Assoc. Comput. Linguistics: EMNLP, 2022, pp. 141–153.
[^19]: S. Papi, M. Turchi, and M. Negri, “AlignATT: Using attentionbased audio-translation alignments as a guide for simultaneous speech translation,” Proc. INTERSPEECH, pp. 3974–3978, 2023.
[^20]: M. Ma, L. Huang, H. Xiong, R. Zheng, K. Liu, B. Zheng, C. Zhang, Z. He, H. Liu, X. Li, H. Wu, and H. Wang, “STACL: Simultaneous translation with implicit anticipation and controllable latency using prefix-to-prefix framework,” in Proc. Annu. Meeting Assoc. Comput. Linguistics, 2019, pp. 3025–3036.
[^21]: P. Pol ́ak, N.-Q. Pham, T. N. Nguyen, D. Liu, C. Mullov, J. Niehues, O. Bojar, and A. Waibel, “CUNI-KIT system for simultaneous speech translation task at IWSLT 2022,” in Proc. Int. Conf. Spoken Language Transl., 2022, pp. 277–285.
[^22]: S. Papi, M. Negri, and M. Turchi, “Attention as a guide for simultaneous speech translation,” in Proc. Annu. Meeting Assoc. Comput. Linguistics, 2023, pp. 13 340–13 356.
[^23]: Q. Dong, Y. Zhu, M. Wang, and L. Li, “Learning when to translate for streaming speech,” in Proc. Annu. Meeting Assoc. Comput. Linguistics, 2022, pp. 680–694.
[^24]: S. Zhang and Y. Feng, “Information-transport-based policy for simultaneous translation,” in Proc. 2022 Conf. Empirical Methods Natural Language Process., 2022, pp. 992–1013.
[^25]: L. F. Abbott, “Lapicque’s introduction of the integrate-and-fire model neuron (1907),” Brain research bulletin, vol. 50, no. 5-6, pp. 303–304, 1999.
[^26]: L. Dong and B. Xu, “CIF: Continuous integrate-and-fire for endto-end speech recognition,” in Proc. IEEE Int. Conf. Acoust. Speech Signal Process. IEEE, 2020, pp. 6079–6083.
[^27]: C. Yi, S. Zhou, and B. Xu, “Efficiently fusing pretrained acoustic and linguistic encoders for low-resource speech recognition,”IEEE Signal Process. Lett., vol. 28, pp. 788–792, 2021.
[^28]: V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Librispeech: an ASR corpus based on public domain audio books,” inProc. IEEE Int. Conf. Acoust. Speech Signal Process., 2015, pp. 5206–5210.
[^29]: V. Pratap, Q. Xu, A. Sriram, G. Synnaeve, and R. Collobert, “MLS: A large-scale multilingual dataset for speech research,” inProc. Interspeech, 2020, pp. 2757–2761.
[^30]: L. Yujian and L. Bo, “A normalized levenshtein distance metric,”IEEE Trans. Pattern Anal. Mach. Intell., vol. 29, no. 6, pp. 1091–1095, 2007.
[^31]: N. Arivazhagan, C. Cherry, W. Macherey, C.-C. Chiu, S. Yavuz, R. Pang, W. Li, and C. Raffel, “Monotonic infinite lookback attention for simultaneous machine translation,” in Proc. Annu. Meeting Assoc. Comput. Linguistics, 2019, pp. 1313–1323.
