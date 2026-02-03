---
title: Robust Speech Recognition via Large-Scale Weak Supervision 번역
subtitle: ""
draft: true
date: 2025-09-20 15:36:49 +0900
categories: [Paper, Translation, ASR]
tags: [Paper, Machine Learning, ASR, Whisper]
math: true
mermaid: true
image:
  path: /images/paper/asr/robust_speech_recognition_via_large_scale_weak_supervision/Subject.png
---


{{< figure src="/images/paper/asr/adaption_whisper_for_streaming_speech_recognition_via_two_pass_decoding_translation/Subject.png" alt="subject" class="center" width="80%" >}}

## Abstract

우리는 인터넷에 있는 방대한 오디오 전사본을 단순히 예측하도록 학습된 음성 처리 시스템의 능력에 대해 연구한다.
multitask 지도 학습을 680,000 시간의 다국어 데이터셋으로 확장하여 훈련한 모델은 표준 벤치마크에서 뛰어난 일반화 성능을 보이며, 기존의 완전 지도학습 결과와도 경쟁력이 있다.
이러한 결과는 fine tuning 필요없이 zero-shot transfer 환경에서 달성했다.
인간과 비교해서도 모델의 정확성과 강건성은 상당히 근접하다.
우리는 강건한 음성 처리에 추가 연구의 기반이 될 수 있도록 모델과 추론 코드를 공개한다.

## 1. Introduction

음성 인식의 발전은 Wav2Vec 2.0[^5]으로 대표되는 비지도 사전 학습 기법의 개발로 촉진되었다.
이러한 기법은 사람의 라벨링 없이 원본 오디오로부터 직접적으로 학습하기 때문에, 라벨링되지 않은 큰 음성 데이터셋을 사용할 수 있으며 빠르게 1,000,000 시간의 훈련 데이터로 확장되었다[^84].
표준 벤치마크에 대해 fine-tuning 됐을 때, 이러한 접근법은 특히 데이터가 적은 환경에서 SOTA를 크게 향상시켰다.

이러한 사전 학습된 오디오 인코더들은 고품질의 음성 표현을 학습하지만, 순수하게 비지도학습만으로는 그 표현을 활용 가능한 출력으로 매핑하는 동등한 성능의 디코더가 부족하기 때문에, 실제로 음성 인식과 같은 작업을 수행하기 위해서는 fine-tuning 단계가 필요하다. ([^6]은 흥미로운 예외이다.)  
이는 유감스럽게도 해당 기법의 유용성과 영향력을 제한하는데, 이는 fine-tuning이 여전히 전문적인 기술이 요구되는 복잡한 연산 과정에는 필요하기 때문이다.
fine-tuning의 필요성은 추가적인 위험이 따른다.
머신러닝 기법은 특정 학습 데이터셋 내에서 동일한 데이터셋에서 보류된 데이터에 대해 성능을 높여주는 패턴을 찾는데 능숙하다.
하지만, 몇몇 패턴들은 취약하고, 우연적이며 다른 데이터셋이나 분포에서 일반화되지 못한다.
특히 충격적인 예시로, [^61]에서는 ImageNet 데이터셋[^65]에서 컴퓨터 비전 모델을 fine-tuning 했을 때, 객체 분류 정확도가 9.2^ 증가한 반면에 다른 8개의 자연 이미지 데이터셋에서 분류할 때 평균 정확도가 향상되는 것이 관측되지 않았음을 보고했다.
모델이 하나의 데이터셋에서 '초인간적'인 성능을 달성했더라도 다른 데이터셋에서 평가됐을때 여전히 많은 기초적인 오류를 범할 수있음을 보여준다.
이는 인간이 감지하지 못하는 데이터셋 특유의 특성을 악용하고 있기 때문일수도 있다[^20].

이는 비지도 사전 학습이 오디오 인코더의 성능을 드라마틱하게 향상시켰음에도 불구하고, 동등한 고품질의 사전 훈련된 디코더의 부재와 데이터셋별 fine-tuning 과정을 추천하는 것에 대해서 중요한 약점이며, 유용성과 강건성을 제한한다는 점을 암시한다.
음성 인식 시스템의 목표는 매 배포때마다 디코더를 지도 학습으로 fine-tuning하는 과정 없이 다양한 환경에서 바로 신뢰성 있게 동작하는 것이다.

[^48] [^39] [^10]의 연구에서 보여주듯이 다양한 데이터셋과 도메인에 걸쳐 지도 학습하는 방법으로 사적학습된 음성 인식 시스템은 높은 강건성을 보이고, 단일 데이터셋에서 훈련된 모델보다 보류된 데이터셋에서 더 효과적인 일반화를 보인다.
이 작업의 성과는 가능한 많은 고품질 음성 인식 데이터셋을 결합함으로 달성된다.
하지만, 쉽게 사용할 수 있는 데이터의 양은 여전히 제한적이다.
SpeechStew[^10]는 7개의 기존에 존재하는 데이터셋을 합쳐 5,140시간의 지도학습 데이터를 구성했다.
이것은 적은 양이 아니지만, [^84]에서 사용한 1,000,000 시간의 라벨링되지 않은 음성 데이터에 비하면 여전히 작은 수준이다.

기존의 높은 퀄리티의 지도학습 데이터셋의 크기에 대한 한계를 인식하고, 최근에는 음성 인식에서 더 큰 데이터셋을 만드려고 노력한다. [^11]과 [^19]에서는 'gold-standard human-validation transcripts' 기준을 완화하고, 약하게 지도된 음성 인식을 10,000시간, 30,000 시간의 노이즈 있는 데이터셋으로 확장하기 위해 정교한 자동화 파이프라인을 사용했다.
이러한 질과 양에서의 trade-off는 종종 올바른 선택일 수 있다.
음성 인식에서는 아직 충분히 연구되지 않았지만, 최근 컴퓨터 비전 분야의 연구에서는 ImageNet[^65] 같은 정답 수준의 크라우드소싱 데이터셋을 넘어, 훨씬 더 크지만 약하게 지도된 데이터셋을 활용하는 것이 모델의 강건성과 일반화 성능을 크게 향상시킨다는 점이 입증되었다[^42] [^35].

그러나 이러한 새로운 데이터셋은 기존 고품질 데이터셋의 총합보다 몇 배 크긴 하지만, 여전히 이전의 비지도 학습 연구에서 사용된 규모에 비하면 훨씬 작은 수준에 머물러 있다.
본 연구에서는 이 격차를 해소하기 위해, 약하게 감독된 음성 인식을 한단계 더 확장하여 680,000 시간의 라벨링된 오디오 데이터를 사용한다.
우리는 이러한 접근법을 Whisper라고 한다. (이름의 약허나 기반이 필요한 경우, Web-scale Supervised Pretraining for Speech Recognition을 의미하는 WSPSR을 사용할 수 있다.)
이 규모에서 학습된 모델은 기존의 데이터셋에서 zero-shot으로 전이될 수 있음을 보여주며, 데이터셋별 미세 조정 없이도 고품질의 결과를 달성할 수 있음을 입증한다.

규모 확장 뿐만 아니라, 우리의 연구는 약하게 지도된 사전 학습의 범위를 영어 음성 인식에 국한되지 않고 다국어와 다중 작업으로 확장하는데에도 초점을 맞추고 있다.
680,000 시간의 오디오에서, 117,000 시간은 96개의 다른 언어로 구성되어 있다.
데이터셋은 또한 126,000 시간의 교차 번역 데이터도 포함하고 있다.
우리는 충분히 큰 모델의 경우 이것이 단점이 아니라 이점이 있음을 발견했다.

우리의 연구는, 음성 인식 분야에서 약하게 지도된 사전 학습의 단순한 규모 확장이 지금까지 충분히 주목받지 못했다는 점을 시사한다.
우리는 이러한 결과를 자기 지도 학습이나 자기 훈련 기법을 사용하지 않고도 달성했다.
강건한 음성 인식에 대한 추가 연구의 기반이 될 수 있도록, 우리는 추론 코드와 모델을 다음 URL에서 공개한다: [https://github.com/openai/whisper]

## 2. Approach

### 2.1. Data Processing

{{< figure src="/images/paper/asr/adaption_whisper_for_streaming_speech_recognition_via_two_pass_decoding_translation/Figure1.png" alt="Figure1" title="Figure1" class="center" width="80%" caption="통합된 이중 패스(Unified Two-pass) 디코딩 프레임워크를 활용한 하이브리드 토크나이저 기반 스트리밍 Whisper" >}}

U2 ASR 모델[^6]은 비스트리밍과 스트리밍 ASR 모두에서 통합된 아키텍처를 제공하는 것을 목표로 한다.
U2 모델은 encoder, CTC decoder, attention decoder로 구성되어있다.
그림 1은 U2 구조를 어떻게 Whisper에 적용했는지 도식화했다.
학습 과정에서, CTC와 attention decoder 모두 하이브리드 CTC-attention loss[^10]를 사용하여 정답 전사를 생성하도록 학습한다.
손실 함수는 식 1과 같이 정의된다.

$$ \mathcal{L} = \alpha \cdot \mathcal{L}\_{\text{CTC}} + (1 - \alpha) \cdot \mathcal{L}\_{\text{Attention}} \qquad\qquad \text{(1)} $$

또한 U2는 인코더의 은닉 표현이과거 또는 일부 제한된 미래 문맥에만 의존하도록 훈련하는 동안 동적 attention masks를 적용한다.
이러한 어텐션 마스크와함께 훈련하는 것은 인코더가 추론 시간에서 스트리밍 모드로 동작할 수 있게 한다.
또한 훈련과 스트리밍 추론 사이의 일관성을 보장한다.
실험에서, 훈련하는 동안 샘플 청크 사이즈를 0.1 ~ 1.0 초 사이의 랜덤한 크기로 정했다.
이는 다양한 청크 사이즈에 대해 일반화할 수 있도록 하기 위함이다.

### 2.2. Streaming inference

추론 흐름은 그림 1의 초록색 화살표로 나타냈다.
인코더는 오디오를 청크단위로 처리하며, CTC decoder는 top-k 스트리밍 부분 전사를 생성하기 위해 prefix beam search[^4]를 수행한다.
종단점은 0.5초 동안 침묵이 유지되거나 최대 지연 제한(max delay constraint)에 도달한 경우로 정의한다.
종단점이 탐지됐을 때, attention decoder의 재정렬(rescoring)을 통해 최종 부분 전사를 전달한다.
최종 전사는 언급한 top-k CTC 가설을 재정렬(rescoring)하고 높은 점수를 선택하는 과정으로 선택된다.

모든 평가는 WeNet C++ 추론 런타임을 사용하여 수행되며, 여기에는 Whisper 추론을 지원하는 오픈소스 구현이 포함되어 있다.
이러한 설정은 end-to-end 방식으로 실제 서비스 운영 시점에서의 테스트를 가능하게 한다.
특히 롱-폼 스트리밍 전사에 대해서도 같은 방식으로 테스트할 수 있다.
이러한 비교는 모델의 성능을 원본 모델과 비교하는 과정에서 더 나은 관점을 제공한다.
Whisper는 롱-폼 전사 과정에서 과거의 긴 전사를 프롬프트로 사용하기 때문에 상당한 이점을 갖는다.

{{< admonition type=info >}}

WeNet은 실시간 및 비실시간 음성 인식을 위한 오픈소스 end-to-end ASR 프레임워크 이며, U2 (Unified Streaming & Non-Streaming) 아키텍처 기반의 모델 학습 추론 서빙을 지원하는 플랫폼이다.

{{</ admonition >}}

WeNet 툴킷은 점진적인 스트리밍 encoder 추론을 위해 효율적인 Key-value 캐시로 구현되었다.
이는 재연산 없이 이전 청크의 KV 값의 재사용을 가능하게한다.
attention 재정렬(rescoring)동안, 시스템은 오직 diagonal causal attention mask를 사용한 단일 배치 전달만으로 충분하며, autoregressive 디코딩은 필요하지 않는다.
이러한 최적화는 추론 효율성을 향상시켰으며, 상당한 사이즈 769 million 개의 파라미터를 가진 Whisper Medium 모델을 파인튜닝한 후에도 실시간 CPU 기반 연산을 가능하게 했다.

### 2.3 Hybrid tokenizer

Whisper는 GPT-2 토크나이저에서 파생된 BPE (Byte Pair Encoding)[^11] 기반의 대규모 토큰 공간을 사용하며, 이는 50,00개 이상의 토큰으로 구성되어 있다[^12].
소규모의 도메인 특화 데이터셋으로 Whisper을 파인튜닝할 때, 넓은 토큰 공간은 CTC decoder를 효과적으로 훈련하기에는 충분히 커버되지 않을 수 있다.
이로 인해, 처음부터 훈련되는 CTC 브랜치(branch)는 도메인 외 또는 희귀토큰에 대해 일반화 성능이 떨어질 수 있다.

이를 해결하기 위해, CTC decoder에서 토큰 공간을 Whisper 토크나이저의 첫 8,000개의 토큰으로 제한했다.
이는 숫자나 대문자, 소문자 단어 그리고 일반적인 서브워드(subwords) 같은 필수 요소를 포함하면서, 성능 손실 없이 효과적으로 토크나이징을 보장한다.
학습 중에는, 8,000개 토큰 기반으로 SentencePiece[^13]을 이용해 CTC 예측 타겟을 생성하며, attention decoder는 기존의 전체 토큰 세트를 유지한다.

추론 에서는 PyTorch의 TorchText[^14] 을 이용해 TorchScript와 호환가능한 retokenizer을 구현한다.
retokenizer는 CTC 가설을 문자열로 디코딩 한다.
이를 다시 Whisper 토크나이저로 리토크나이징하고, Whisper 전용 프롬프트 토큰을 추가한 뒤 어텐션 디코더로 전달하여 재정렬을 수행한다.
이는 그림 1에 나타난바와 같다.

## 3. Datasets

우리는 실험의 완전성과 다른 접근 방식 과의 비교를 위해 LibriSpeech[^15]에서도 실험을 수행하였지만, 주된 초점은 내부에서 선별한 어닝 콜(earnings calls) 데이터셋에 맞춰져 있다.
Earnings-22[^16]도 고려하였지만, 제외하였다.
왜냐하면 규모가 작아 실험에서 필요한 학습에는 부족하다고 판단했기 때문이다.
내부 어닝 콜 데이터셋은 높은 퀄리티와 서면 전사(written-from transcripts)와 함께 적절한 구두점, 대소문자, 이메일과 숫자 같은 형식이 Whisper의 출력전사와와 유사하여 선택되었다.
이 데이터셋을 중심으로, Whisper을 streaming ASR로 파인튜닝하는 것을 목적으로 하며, 이는 전사, 구두점, 대소문자 그리고 텍스트 정규화 역처리 (inverse text normalization)을 통합적 기능으로 제공할 수 있게 한다.

또한, 데이터셋은 일반화의 도전적인 테스트 환경을 제공한다.
이는 어닝 콜 마다 특화되있는 복잡한 금융 용어가 주어지기 때문이다.
이는 도메인 특화 용어와 희귀 단어들을 다루기 때문에 모델의 능력을 평가하기에 적합하다.
LibriSpeech와 비교하면, 어닝콜은 롱폼 전사 테스트를 위한 현실적으로 긴 오디오 샘플, 자연스러운 말의 멈춤, 적당한 잡음을 특징으로하여 현실적인 서비스 환경을 반영할 수 있다.
다른 장점은 풍부한 높은 퀄린티의 데이터를 가지고 있는 것이다.
이 덕분에 훈련 데이터 증가에 따라 모델의 성능이 어디까지 향상되는지 연구할 수있다.

학습 데이터는 2023년 이전의 어닝 콜 데이터를 랜덤으로 샘플링한 것이며, 5,800 시간의 오디오와 텍스트트 전사를 포함한다.
forced aligner을 통해 5~20초 길이의 클립으로 분할했다.
데이터 누수 방지를 위해, 우리의 테스트 셋은 2023년 이후의 83개의 어닝콜에서 샘플링되었으며, 총 10시간 분량의 83개 샘플로 구성되어 있다.
이러한 설정은 타겟 분포를 커버하는 표현을 제공하면서도, 테스트 샘플이 end-to-end streaming 성능 측정으로 충분하도록 보장한다.

## 4. Results

제안한 접근 방식이 데이터에 따라 어떻게 확장되는지 평가하기 위해, 어닝 콜 트레이닝 데이터의 서브셋을 사용하여 Whisper Medium 모델을 파인튜닝 했다.
각 서브셋은 725, 1450, 2900, 5800 시간의 오디오로 구성되어있고, 모두 싱글 토크나이저와 하이브리드 토크나이저 접근법으로 평가했다.
먼저, 인과정 어텐션 마스크(causal attention mask)에서 오직 attention 로스만으로 1 에포크만 훈련 함으로써 Whisper을 30초 보다 짧은 입력에 적응 시키고 인코더가 스트리밍에 적합하도록 보장시켰다.
다음으로, CTC 분류 헤드를 추가하고, 두 에포크 동안 오직 CTC 로스만으로 학습시켰다.
이 단계에서 다른 모든 파라미터는 고정(freeze) 시켰다.
훈련스텝에 관한 손실 포하가 통상적으로 첫 에포크 이후에 관찰되기 때문에, 각 단계는 1~2 에포크면 충분하다.
마지막으로, 모든 파라미터를 풀고(unfreeze) 하이브리드 로스를르 적용한다.
검증 WER 3번 연속으로 향상되지않으면 중단한다.
이러한 절차는 파라미터를 사전훈련된 모델에 가깝게 유지하면서 더 잘 일반화 한다.

모든 U2 Whisper 실험에서, CTC 디코더에서 prefix beam search 크기를 10으로 설정하고, 재정렬(rescoring)에서는 top 6 후보 가설을 사용한다.
8-bit 양화화를 채택하여 streaming latency를 감소시켰다.
양자화된 모델은 WER 기준 0.3%p WER 감소가 있어, 전체 경향에 영향을 주지 않는다.
모든 평가는 1초 청크 크기, 최대 지연시간 12초 기준으로 수행되었으며, 언어모델은 사용하지않았다.

### 4.1. Data scaling behavior of this approach

{{< figure src="/images/paper/asr/adaption_whisper_for_streaming_speech_recognition_via_two_pass_decoding_translation/Table1.png" alt="Table1" title="Table1" class="center" width="50%" caption="훈련 데이터의 다양한 크기와 이에 따른 Whisper Medium을 스트리밍으로 파인튜닝한 모델의 WER 결과">}}

표1은 서로 다른 훈련셋 크기에 대해서 싱글 토크나이저와 하이브리드 토크나이저 설정에서의 어닝 콜 테스트 셋의 WER 결과를 나타냈다.
하이브리드 토크나이즈는 일관되게 더 나은 성능을 보여주며, 특히 데이터 셋이 작을 때 더 효과적이었다.
그러나, 이러한 장점은 데이터셋 크기가 커질수록 감소된다.
같은 Whisper Medium 모델의 구조를 pretrain 가중치 없이 모든 훈련 셋에서 처음부터 훈련시켰을 때,WER 20.59%가 나타났다.
pretrained 모델에서는 17.30%가 나타났으며, 이는 pretrained 가중치의 가치를 보여준다.

### 4.2. Runtime configurations and performance

{{< figure src="/images/paper/asr/adaption_whisper_for_streaming_speech_recognition_via_two_pass_decoding_translation/Table2.png" alt="Table2" title="Table2" class="center" width="50%" caption="재정렬(rescoring)여부와 다양한 청크 크기로 스트리밍 Whisper Medium의 WER">}}

우리는 최고 성능 체크포인트를 평가했다.
이는 5,800 시간동한 학습된 모델의 체크포인트이다.
다양한 청크 크기에서 평가하였다.
데이터셋은 어닝 콜 테스트 셋을 사용했으며, 결과는 표2에서 보여준다.

청크 사이즈가 작아질수록 정확도가 낮아지는 것을 알 수 있다.
청크가 작아질수록 포매팅 에러 (e.g., "$1.3 million"이 "1.3 million dollars"로 잘못 출력)가 발생하는 것을 관측할 수 있었다.
이는 옳바른 포매팅을 위해 필요한 적절한 미래 컨텍스트가 부분 디코딩 시점에 누락되기 때문이다.
또한, CTC prefix beam search에서 옳바른 가설이 너무 일찍 제거되면 이후 단계에서는 형식 오류를 바로잡을 수 없다.

재정렬(Rescoring)은 디코더의 어텐션 정보를 활용해 적절한 가설을 선택함으로써 정확도를 향상시킬 수 있다.
하지만 표 2를 보면 그 효과는 상대적으로 제한적이다.
통상적으로, CTC가 생성하는 상위 가설들은 구두점이나 대소문자와 같은 사소한 차이만 있기 때문에, 재정렬을 통해 얻을 수 있는 이득이 제한된다.

{{< figure src="/images/paper/asr/adaption_whisper_for_streaming_speech_recognition_via_two_pass_decoding_translation/Table3.png" alt="Table3" title="Table3" class="center" width="50%" caption="최종화 시간 지연 제약에 따른 모델 성능">}}

maximum delay 파라미터 또한 성능에 상당한 영향을 준다.
표3에는 WER과 Real-Time Factor (RTF) 그리고 평균 최종화 지연 시간(average finalize latency) 보인다.
평균 최종화 지연 시간은 어텐션 재정렬과 같은 최종 연산의 평균 시간을 의미한다.
모든 U2 실험은 4개의 가상 CPU 코어를 사용하여 수행했다. (Intel Xeon 6240)

긴 지연시간은 WER을 향상시키는데 도움이 된지만, 연산 복잡도가 입력 길이에 따라 기하급수적으로 증가하기 때문에, 계산 비용 또한 증가한다.
최대 지연을 선택하는 것은 정확도와 실행 시간 효율성 간의 균형을 맞추는데 중요한 요소이다.
end-to-end 레이턴시를 고려할 때, 3가지의 주요 요소가 있다.
각각 청크 기반 버퍼링 레이턴시(연산시간 제외), 부분 전사 연산(제안 기법에서는 약 267ms가 소요되었다.), 최종화 연산 시간 (표 3)이다.
우리의 실험에서는, 최대 지연 시간 12초 설정에서도 U2 Whisper 모델이 스트리밍에서 수용 가능하다.
다만, 최종화 연산 시간이 여전히 실시간 프로그램에서 높은 장애물이며, 이는 WhisperTurbo 체크포인트같이 작은 디코더 모델을 선택하면 감소시킬 수 있다.

### 4.3 Comparing with UFAL Whisper and other variants on earnings and LibriSpeech

우리는 U2 streaming Whisper의 접근법을 UFAL Whisper과 비교했다.
그리고 다른 비 스트리밍 즉 오프라인 모델을 어닝 콜과 LibriSpeech 데이터셋에서 비교했다.
공정한 평가를 위해서, Whisper Medium을 Whisper 논문에[^1] 따라 파인튜닝 했다.
해당 가중치는 UFAL 추론 코드에 로드하여 평가했다.
UFAL whisper 추론에서는 모든 설정을 기본 값으로 사용했으며, A100GPU와 충분한 CPU를 사용했다.
반면에 모든 U2 모델들은 오직 4코어 Xeon 6240 CPU에서 실행되었다.
또한, 비교를 위해 비 스트리밍 Whisper도 포함시켰으며, "FT" 표기는 해당 데이터셋으로 파인튜닝 되었음을 의미한다.

{{< figure src="/images/paper/asr/adaption_whisper_for_streaming_speech_recognition_via_two_pass_decoding_translation/Figure2.png" alt="Figure 2" title="Figure 2" class="center" width="50%" caption="청크 사이즈에 따른 어닝 콜 데이터셋 결과">}}

{{< figure src="/images/paper/asr/adaption_whisper_for_streaming_speech_recognition_via_two_pass_decoding_translation/Figure3.png" alt="Figure 3" title="Figure 3" class="center" width="50%" caption="청크 사이즈에 따른 LibriSpeech test-clean 데이터셋 결과">}}

{{< figure src="/images/paper/asr/adaption_whisper_for_streaming_speech_recognition_via_two_pass_decoding_translation/Figure4.png" alt="Figure 4" title="Figure 4" class="center" width="50%" caption="청크 사이즈에 따른 LibriSpeech test-other 데이터셋 결과">}}

그림 2는 청크 사이즈 0.1, 0.24, 0.5, 1, 1.5, 6초 에서의 어닝 콜 테스트 셋 결과를 보여준다.
U2 Whisper은 긴 테스트 샘플을 세그먼트 없이 오프라인 추론할 수 없기 때문에, 오프라인 U2 Whisper WER은 보여주지 않는다.

우리는 LibriSpeech에서도 유사한 평가를 수행했다.
학습은 표준 학습 분할만 사용했다.
그림 3과 4에서는 test-clean과 test-other을 각각 보여준다.
동일한 구조의 모델을 사전학습 없이 처음부터 학습시켰을 경우, test-clean 에서 5.18% test-other에서 13.35%의 WER을 보였다.

UFAL의 오리지널 구현은 각 청크에 대해 확정되지 않은 부분 전사를 출력하지 않는다.
오직 두 번 연속 동일한 예측이 나왔을 때만 최종 전사를 출력한다.
실제로, UFAL이 부분 전사를 출력하도록 만들수 있다.
이러한 관점에서 보면, 두 모델을 동일한 청크 크기로 테스트하는 것은 계산 시간을 제외하고 부분 전사 출력이 대략적으로 일치하는 것을 알 수 있다.
실제로는, UFAL에서 부분 출력을 강제하는 것은 부분 출력에서 hallucinations 발생되게 만든다.
다시 말해, 청크가 작을 수록 WER이 감소하는 것을 설명한다.
하지만, 단어 경계가 여러 청크에 의해 나눠질 수 있고, 이후의 업데이트가 이전 출력을 덮어쓸 수 있기 때문에 부분 전사 품질을 평가하는 것은 어렵다.
그래서 우리는 오직 최종적으로 확정된 전사의 WER만 비교한다.

작은 청크 크기에서는, 어닝 콜과 LibriSpeech test-clean에서 U2 Whisper가 UFAL Whisper을 능가한다.
어닝 콜 시나리오와 유사하게 더 도전적인 LibriSpeech test-other에서는 UFAL을 능가하기 위해서는 더 많은 훈련 데이터가 필요하다.
청크 크기가 커질 때, UFAL Whisper가 유리해진다.
왜냐하면 비스트리밍 모드와 더 유사하기 때문이다.

연산 효율성 측면에서, U2는 효율적인 CPU에서 동작하는 반면, UFAL은 더 많은 연산량을 요구한다.
Whisper Medium을 사용하는 UFAL도 CPU에서 8-bit 양자화를 사용해도, 실시간 속도를 달성하지 못한다.
GPU 환경에서도, UFAL은 청크 크기가 LibriSpeech에서는 0.5초 이하, 어닝콜에서는 1초 이하일 경우 여전히 RTF > 1로 어려운 수준이다.

계산을 고려하지 않는 최종 지연 시간 관점에서 보면, UFAL 두번 연속 예측이 매칭했을 때만 단어를 확정하므로, 통상적으로 청크 크기의 두배 정도의 지연시간을 의미한다[^2].
만약 청크 사이즈가 1초라면, UFAL은 평균 출력 최종 지연 시간은 약 2초인 경향이 있다.
반면에, U2 Whisper은 오직 종단점(말의 멈춤)을 감지했을 때문 부분 전사를 확정하므로, 평균 지연 시간이 더 높을 수 있다.
하지만, U2 Whisper의 최대 지연 파라미터를 통해 최대 대기 시간을 명확히 제한할 수 있는 반면, UFAL은 최종 전사 출력의 최대 지연 시간에 대한 명확한 상한이 없다.

요약하면며, 충분한 도메인 내의 데이터가 제공될 경우, U2 Whisper은 낮은 지연의 부분 전사가 필수적이고 계산 효율성이 중요한 실시간 애플리케이션에 더 적합하다.
반면, UFAL Whisper은 즉각적인 부분 전사가 꼭 필요하지 않으며, GPU 자원이 풍부하게 사용 가능한ㅎ 환경에 더 적합하다.

## Conclusion

우리는 Whisper를 U2 아키텍처를 활용해 스트리밍 ASR 모델로 변환하는 방법을 제시하였으며, 이를 통해 원래 Whisper 설정과 비슷한 성능을 달성하였다. 또한, 하이브리드 토크나이저를 도입하여 특히 제한된 데이터로 파인튜닝할 때 일반화 성능을 향상시켰다.
실험을 통해 이 접근 방식의 데이터 확장 특성을 분석하였으며, 적절한 실행 구성하에서 CPU에서도 실시간 처리가 가능함을 확인했다.

하이브리드 토크나이저는 저자원 환경에서는 큰 성능 향상을 보였지만, 데이터가 많아질수록 그 이점은 점점 줄어드는 경향이 있었다. 
또한 다양한 실행 구성 실험을 통해 청크 크기, 최대 지연 시간, 계산 복잡도 간의 트레이드 오프도 설명하였다식

향후 연구에서는 사전 학습된 디코더가 갖고 있는 언어적 지식을 더욱 효과적으로 활용하는 방향으로 발전시킬 예정이다.

[^1]: Alcorn, M. A., Li, Q., Gong, Z., Wang, C., Mai, L., Ku, W.S., and Nguyen, A. Strike (with) a pose: Neural networks are easily fooled by strange poses of familiar objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4845–4854, 2019.
[^2]: Amodei, D., Anubhai, R., Battenberg, E., Case, C., Casper, J., Catanzaro, B., Chen, J., Chrzanowski, M., Coates, A., Diamos, G., et al. Deep speech 2: end-to-end speech recognition in english and mandarin. arxiv. arXiv preprint arXiv:1512.02595, 2015.
[^3]: Ardila, R., Branson, M., Davis, K., Henretty, M., Kohler, M., Meyer, J., Morais, R., Saunders, L., Tyers, F. M., and Weber, G. Common voice: A massively-multilingual speech corpus. arXiv preprint arXiv:1912.06670, 2019.
[^4]: Babu, A., Wang, C., Tjandra, A., Lakhotia, K., Xu, Q., Goyal, N., Singh, K., von Platen, P., Saraf, Y., Pino, J., et al. XLS-R: Self-supervised cross-lingual speech representation learning at scale. arXiv preprint arXiv:2111.09296, 2021.
[^5]: Baevski, A., Zhou, H., Mohamed, A., and Auli, M. wav2vec 2.0: A framework for self-supervised learning of speech representations. arXiv preprint arXiv:2006.11477, 2020.
[^6]: Baevski, A., Hsu, W.-N., Conneau, A., and Auli, M. Unsupervised speech recognition. Advances in Neural Information Processing Systems, 34:27826–27839, 2021.
[^7]: Bapna, A., Cherry, C., Zhang, Y., Jia, Y., Johnson, M., Cheng, Y., Khanuja, S., Riesa, J., and Conneau, A. mslam: Massively multilingual joint pre-training for speech and text. arXiv preprint arXiv:2202.01374, 2022.
[^8]: Barbu, A., Mayo, D., Alverio, J., Luo, W., Wang, C., Gutfreund, D., Tenenbaum, J., and Katz, B. Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. Advances in neural information processing systems, 32, 2019. 
[^9]: Caruana, R. Multitask learning. Machine learning, 28(1): 41–75, 1997.
[^10]: Chan, W., Park, D., Lee, C., Zhang, Y., Le, Q., and Norouzi, M. SpeechStew: Simply mix all available speech recognition data to train one large neural network. arXiv preprint arXiv:2104.02133, 2021.
[^11]: Chen, G., Chai, S., Wang, G., Du, J., Zhang, W.-Q., Weng, C., Su, D., Povey, D., Trmal, J., Zhang, J., et al. Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio. arXiv preprint arXiv:2106.06909, 2021.
[^12]: Chen, S., Wu, Y., Wang, C., Chen, Z., Chen, Z., Liu, S., Wu, J., Qian, Y., Wei, F., Li, J., et al. Unispeech-sat: Universal speech representation learning with speaker aware pre-training. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 6152–6156. IEEE, 2022a.
[^13]: Chen, T., Xu, B., Zhang, C., and Guestrin, C. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016.
[^14]: Chen, Z., Zhang, Y., Rosenberg, A., Ramabhadran, B., Moreno, P., Bapna, A., and Zen, H. Maestro: Matched speech text representations through modality matching. arXiv preprint arXiv:2204.03409, 2022b.
[^15]: Child, R., Gray, S., Radford, A., and Sutskever, I. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.
[^16]: Collobert, R., Weston, J., Bottou, L., Karlen, M., Kavukcuoglu, K., and Kuksa, P. Natural language processing (almost) from scratch. Journal of machine learning research, 12(ARTICLE):2493–2537, 2011.
[^17]: Conneau, A., Ma, M., Khanuja, S., Zhang, Y., Axelrod, V., Dalmia, S., Riesa, J., Rivera, C., and Bapna, A. Fleurs: Few-shot learning evaluation of universal representations of speech. arXiv preprint arXiv:2205.12446, 2022.
[^18]: Del Rio, M., Delworth, N., Westerman, R., Huang, M., Bhandari, N., Palakapilly, J., McNamara, Q., Dong, J., Zelasko, P., and Jett´e, M. Earnings-21: a practical benchmarkfor asr in the wild. arXiv preprint arXiv:2104.11348, 2021.
[^19]: Galvez, D., Diamos, G., Torres, J. M. C., Achorn, K., Gopi, A., Kanter, D., Lam, M., Mazumder, M., and Reddi, V. J. The people’s speech: A large-scale diverse english speech recognition dataset for commercial usage. arXiv preprint arXiv:2111.09344, 2021.
[^20]: Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., and Wichmann, F. A. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.
[^21]: Ghorbani, B., Firat, O., Freitag, M., Bapna, A., Krikun, M., Garcia, X., Chelba, C., and Cherry, C. Scaling laws for neural machine translation. arXiv preprint arXiv:2109.07740, 2021.
[^22]: Griewank, A. and Walther, A. Algorithm 799: revolve: an implementation of checkpointing for the reverse or adjoint mode of computational differentiation. ACM Transactions on Mathematical Software (TOMS), 26(1):19–45, 2000.
[^23]: Gunter, K., Vaughn, C., and Kendall, T. Contextualizing/s/retraction: Sibilant variation and change in washington dc african american language. Language Variation and Change, 33(3):331–357, 2021.
[^24]: Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., Fern´andez del R´ıo, J., Wiebe, M., Peterson, P., G´erard-Marchant, P., Sheppard, K., Reddy, T., Weckesser, W., Abbasi, H., Gohlke, C., and Oliphant, T. E. Array programming with NumPy. Nature, 585:357–362, 2020. doi: 10.1038/ s41586-020-2649-2.
[^25]: Hendrycks, D. and Gimpel, K. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.
[^26]: Hendrycks, D., Liu, X., Wallace, E., Dziedzic, A., Krishnan, R., and Song, D. Pretrained transformers improve out-ofdistribution robustness. arXiv preprint arXiv:2004.06100, 2020.
[^27]: Hernandez, F., Nguyen, V., Ghannay, S., Tomashenko, N. A., and Est`eve, Y. Ted-lium 3: twice as much data and corpus repartition for experiments on speaker adaptation. In SPECOM, 2018.
[^28]: Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R., and Mohamed, A. Hubert: Selfsupervised speech representation learning by masked prediction of hidden units. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3451–3460, 2021a.
[^29]: Hsu, W.-N., Sriram, A., Baevski, A., Likhomanenko, T., Xu, Q., Pratap, V., Kahn, J., Lee, A., Collobert, R., Synnaeve, G., et al. Robust wav2vec 2.0: Analyzing domain shift in self-supervised pre-training. arXiv preprint arXiv:2104.01027, 2021b.
[^30]: Huang, G., Sun, Y., Liu, Z., Sedra, D., and Weinberger, K. Q. Deep networks with stochastic depth. In European conference on computer vision, pp. 646–661. Springer, 2016.
[^31]: Jia, R. and Liang, P. Adversarial examples for evaluating reading comprehension systems. arXiv preprint arXiv:1707.07328, 2017.
[^32]: Johnson, M., Schuster, M., Le, Q. V., Krikun, M., Wu, Y., Chen, Z., Thorat, N., Vi´egas, F., Wattenberg, M., Corrado, G., et al. Google’s multilingual neural machine translation system: Enabling zero-shot translation. Transactions of the Association for Computational Linguistics, 5:339351, 2017.
[^33]: Kendall, T. and Farrington, C. The corpus of regional african american language. Version 2021.07. Eugene, OR: The Online Resources for African American Language Project. http://oraal.uoregon.edu/coraal, 2021. Accessed: 2022-09-01.
[^34]: Koenecke, A., Nam, A., Lake, E., Nudell, J., Quartey, M., Mengesha, Z., Toups, C., Rickford, J. R., Jurafsky, D., and Goel, S. Racial disparities in automated speech recognition. Proceedings of the National Academy of Sciences, 117(14):7684–7689, 2020.
[^35]: Kolesnikov, A., Beyer, L., Zhai, X., Puigcerver, J., Yung, J., Gelly, S., and Houlsby, N. Big transfer (bit): General visual representation learning. In European conferenceon computer vision, pp. 491–507. Springer, 2020.
[^36]: Kuchaiev, O., Li, J., Nguyen, H., Hrinchuk, O., Leary, R., Ginsburg, B., Kriman, S., Beliaev, S., Lavrukhin, V., Cook, J., et al. Nemo: a toolkit for building ai applications using neural modules. arXiv preprint arXiv:1909.09577, 2019.
[^37]: Lake, B. M., Ullman, T. D., Tenenbaum, J. B., and Gershman, S. J. Building machines that learn and think like people. Behavioral and brain sciences, 40, 2017.
[^38]: Liao, H., McDermott, E., and Senior, A. Large scale deep neural network acoustic modeling with semi-supervised training data for youtube video transcription. In 2013 IEEE Workshop on Automatic Speech Recognition and Understanding, pp. 368–373. IEEE, 2013.
[^39]: Likhomanenko, T., Xu, Q., Pratap, V., Tomasello, P., Kahn, J., Avidov, G., Collobert, R., and Synnaeve, G. Rethinking evaluation in asr: Are our models robust enough? arXiv preprint arXiv:2010.11745, 2020.
[^40]: Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
[^41]: Luong, M.-T., Le, Q. V., Sutskever, I., Vinyals, O., and Kaiser, L. Multi-task sequence to sequence learning. arXiv preprint arXiv:1511.06114, 2015.
[^42]: Mahajan, D., Girshick, R., Ramanathan, V., He, K., Paluri, M., Li, Y., Bharambe, A., and Van Der Maaten, L. Exploring the limits of weakly supervised pretraining. In Proceedings of the European conference on computer vision (ECCV), pp. 181–196, 2018.
[^43]: Mauch, M.andEwert, S. Theaudiodegradation toolbox and its application to robustness evaluation. In Proceedings of the 14th International Society for Music Information Retrieval Conference (ISMIR 2013), Curitiba, Brazil, 2013. accepted.
[^44]: McCann, B., Keskar, N. S., Xiong, C., and Socher, R. The natural language decathlon: Multitask learning as question answering. arXiv preprint arXiv:1806.08730, 2018.
[^45]: Meyer, J., Rauchenstein, L., Eisenberg, J. D., and Howell, N. Artie bias corpus: An open dataset for detecting demographic bias in speech applications. In Proceedings of the 12th Language Resources and Evaluation Conference, pp. 6462–6468, Marseille, France, May 2020. European Language Resources Association. ISBN 979-10-9554634-4. URL https://aclanthology.org/2020. lrec-1.796.
[^46]: Miller, J., Krauth, K., Recht, B., and Schmidt, L. The effect of natural distribution shift on question answering models. In ICML, 2020.
[^47]: Mohamed, A.-r., Dahl, G., Hinton, G., et al. Deep belief networks for phone recognition. In Nips workshop on deep learning for speech recognition and related applications, volume 1, pp. 39, 2009.
[^48]: Narayanan, A., Misra, A., Sim, K. C., Pundak, G., Tripathi, A., Elfeky, M., Haghani, P., Strohman, T., and Bacchiani, M. Toward domain-invariant speech recognition via large scale training. In 2018 IEEE Spoken Language Technology Workshop (SLT), pp. 441–447. IEEE, 2018.
[^49]: Panayotov, V., Chen, G., Povey, D., and Khudanpur, S. Librispeech: an asr corpus based on public domain audio books. In 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP), pp. 5206–5210. IEEE, 2015.
[^50]: pandas development team, T. pandas-dev/pandas: Pandas, February 2020. URL https://doi.org/10. 5281/zenodo.3509134. 
[^51]: Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D., and Le, Q. V. SpecAugment: A simple data augmentation method for automatic speech recognition. arXiv preprint arXiv:1904.08779, 2019.
[^52]: Pascanu, R., Mikolov, T., and Bengio, Y. On the difficulty of training recurrent neural networks. In International conference on machine learning, pp. 1310–1318. PMLR, 2013.
[^53]: Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems 32, pp. 80248035, 2019.
[^54]: Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830, 2011.
[^55]: Polyak, B. T. and Juditsky, A. B. Acceleration of stochastic approximation by averaging. SIAM journal on control and optimization, 30(4):838–855, 1992.
[^56]: Pratap, V., Sriram, A., Tomasello, P., Hannun, A. Y., Liptchinsky, V., Synnaeve, G., and Collobert, R. Massively multilingual asr: 50 languages, 1 model, 1 billion parameters. ArXiv, abs/2007.03001, 2020a.
[^57]: Pratap, V., Xu, Q., Sriram, A., Synnaeve, G., and Collobert, R. Mls: A large-scale multilingual dataset for speech research. arXiv preprint arXiv:2012.03411, 2020b.
[^58]: Press, O. and Wolf, L. Using the output embedding to improve language models. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers, pp. 157–163, Valencia, Spain, April 2017. Association for Computational Linguistics. URL https: //aclanthology.org/E17-2025.
[^59]: Provilkov, I., Emelianenko, D., and Voita, E. Bpe-dropout: Simple and effective subword regularization. arXiv preprint arXiv:1910.13267, 2019.
[^60]: Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. Language models are unsupervised multitask learners. 2019.
[^61]: Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020, 2021.
[^62]: Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P. J., et al. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21(140):1–67, 2020.
[^63]: Ravanelli, M., Parcollet, T., Plantinga, P., Rouhe, A., Cornell, S., Lugosch, L., Subakan, C., Dawalatabad, N., Heba, A., Zhong, J., Chou, J.-C., Yeh, S.-L., Fu, S.-W., Liao, C.-F., Rastorgueva, E., Grondin, F., Aris, W., Na, H., Gao, Y., Mori, R. D., and Bengio, Y. SpeechBrain: A general-purpose speech toolkit, 2021. arXiv:2106.04624.
[^64]: Recht, B., Roelofs, R., Schmidt, L., and Shankar, V. Do ImageNet classifiers generalize to ImageNet? In Chaudhuri, K. and Salakhutdinov, R. (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 5389–5400. PMLR, 09–15 Jun 2019. URLhttps://proceedings.mlr.press/v97/recht19a.html.
[^65]: Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115(3): 211–252, 2015.
[^66]: Schultz, T. and Kirchhoff, K. Multilingual speech processing. Elsevier, 2006.
[^67]: Seide, F., Li, G., Chen, X., and Yu, D. Feature engineering in context-dependent deep neural networks for conversational speech transcription. In 2011 IEEE Workshop on Automatic Speech Recognition & Understanding, pp. 24–29. IEEE, 2011.
[^68]: Sennrich, R., Haddow, B., and Birch, A. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.
[^69]: Speer, R. ftfy. Zenodo, 2019. URL https://doi.org/10.5281/zenodo.2591652. Version 5.5.
[^70]: Sutskever, I., Vinyals, O., and Le, Q. V. Sequence to sequence learning with neural networks. Advances in neural information processing systems, 27, 2014.
[^71]: Taori, R., Dave, A., Shankar, V., Carlini, N., Recht, B., and Schmidt, L. Measuring robustness to natural distribution shifts in image classification. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 18583–18599. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/d8330f857a17c53d217014ee776bfd50-Paper.pdf.
[^72]: Torralba, A. and Efros, A. A. Unbiased look at dataset bias. CVPR 2011, pp. 1521–1528, 2011.
[^73]: Toshniwal, S., Sainath, T. N., Weiss, R. J., Li, B., Moreno, P. J., Weinstein, E., and Rao, K. Multilingual speech recognition with a single end-to-end model. 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 4904–4908, 2018.
[^74]: Valk, J. and Alum¨ae, T. Voxlingua107: a dataset for spoken language recognition. In 2021 IEEE Spoken Language Technology Workshop (SLT), pp. 652–658. IEEE, 2021.
[^75]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. In Advances in neural information processing systems, pp. 5998–6008, 2017.
[^76]: Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., Carey, C. J., Polat, ˙ I., Feng, Y., Moore, E. W., VanderPlas, J., Laxalde, D., Perktold, J., Cimrman, R., Henriksen, I., Quintero, E. A., Harris, C. R., Archibald, A. M., Ribeiro, A. H., Pedregosa, F., van Mulbregt, P., and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17:261–272, 2020. doi: 10.1038/s41592-019-0686-2.
[^77]: Wang, C., Tang, Y., Ma, X., Wu, A., Okhonko, D., and Pino, J. fairseq s2t: Fast speech-to-text modeling with fairseq. arXiv preprint arXiv:2010.05171, 2020a.
[^78]: Wang, C., Wu, A., and Pino, J. Covost 2 and massively multilingual speech-to-text translation. arXiv preprint arXiv:2007.10310, 2020b.
[^79]: Wang, C., Riviere, M., Lee, A., Wu, A., Talnikar, C., Haziza, D., Williamson, M., Pino, J., and Dupoux, E. Voxpopuli: Alarge-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation. arXiv preprint arXiv:2101.00390, 2021.
[^80]: Wang, P., Sainath, T. N., and Weiss, R. J. Multitask training with text data for end-to-end speech recognition. arXiv preprint arXiv:2010.14318, 2020c.
[^81]: Watanabe, S., Mandel, M., Barker, J., Vincent, E., Arora, A., Chang, X., Khudanpur, S., Manohar, V., Povey, D., Raj, D., et al. Chime-6 challenge: Tackling multispeaker speech recognition for unsegmented recordings. arXiv preprint arXiv:2004.09249, 2020. 
[^82]: Xu, Q., Baevski, A., Likhomanenko, T., Tomasello, P., Conneau, A., Collobert, R., Synnaeve, G., and Auli, M. Selftraining and pre-training are complementary for speech recognition. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 3030–3034. IEEE, 2021.
[^83]: Zhang, Y., Qin, J., Park, D. S., Han, W., Chiu, C.-C., Pang, R., Le, Q. V., and Wu, Y. Pushing the limits of semisupervised learning for automatic speech recognition. arXiv preprint arXiv:2010.10504, 2020.
[^84]: Zhang, Y., Park, D. S., Han, W., Qin, J., Gulati, A., Shor, J., Jansen, A., Xu, Y., Huang, Y., Wang, S., et al. BigSSL: Exploring the frontier of large-scale semi-supervised learning for automatic speech recognition. arXiv preprint arXiv:2109.13226, 2021.
