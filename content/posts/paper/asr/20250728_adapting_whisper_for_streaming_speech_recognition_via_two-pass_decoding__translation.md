---
title: Adapting Whisper for Streaming Speech Recognition via Two-Pass Decoding 번역
subtitle: ""
draft: false
date: 2025-07-28 10:49:49 +0900
categories: [Paper, Translation]
tags: [Paper, Machine Learning, ASR, Whisper]
math: true
mermaid: true
image:
  path: /images/paper/asr/adaption_whisper_for_streaming_speech_recognition_via_two_pass_decoding_translation/Subject.png
---


{{< figure src="/images/paper/asr/adaption_whisper_for_streaming_speech_recognition_via_two_pass_decoding_translation/Subject.png" alt="subject" class="center" width="80%" >}}

## Abstract

OpenAI Whisper는 680,000 시간 분량의 오디오로 학습된 강건한 ASR 모델이다.
하지만, sequence-to-sequence 목적함수로 학습된 encoder-decoder 아키텍처는 스트리밍 ASR에 대한 기본 지원이 부족하다.
본 논문에서는 Whisper를 WeNet 툴킷을 활용해 Unified Two-pass(U2) 구조를 도입하여 스트리밍 ASR에 맞게 파인튜닝하였다.
추가로 인과적 어텐션 마스크(causal attention mask)로 훈련된 Connectionist Temporal Classification (CTC) 디코더를 스트리밍 부분 전사를 생성하기 위해 도입하였으며, 기존의 Whisper 디코더는 이 부분 전사를 재정렬(rerank)한다.
LibriSpeech 와 earning call 데이터셋에서 실험한 결과, 충분한 파인 튜닝 데이터를 제공하면, Whisper는 가용가능한 스트리밍 ASR 모델에 적응될 수 있음을 보여준다.
또한, 하이브리드 토크나이저 접근법을 도입했다.
이는 CTC 디코더에서 작은 토큰 공간을 사용하는 반면에 어텐션 디코더에는 Whisper의 원래 토공간을 사용한다.
결과적으로 데이터 효율성과 일반화 성능을 향상시켰다.

**Index Terms:**: speech recognition, machine learning, speech signal processing

## 1. Introduction

대규모 학습은 음성 인식 모델에서 정확성과 견고성을 크게 향상시켰다.
OpenAI에서 공개된 Whisper는 대표적인 예시이다.
Whisper[^1]는 680,000 시간의 오디오로 훈련됐으며, 다양한 공개 벤치마크 전체에서 높은 성능을 달성했다.
하지만, 비인과적(non-causal) 설계로, 본질적으로 스트리밍 음성 인식에 제약이 있다.
Whisper을 스트리밍 ASR에 적용하기위한 많은 노력이 있었다.
UFAL streaming Whisper의 구현 [^2]은 들어오는 오디오를 정해진 길이까지 모델이 동작하기 전까지 버퍼링한다. 연속된 비 스트리밍 예측 결과를 비교하여 전사를 확정하는 스트리밍 정책을 사용한다.

{{< admonition type=info >}}

UFAL (Ústav formální a aplikované lingvistiky): 체코 Chales University의 연구 그룹이다.

{{</ admonition >}}

따라서, UFAL streaming whisper는 whisper streaming 이다.
다른 접근법인 SimulWhisper[^3]는 더 나아가 단어 경계 절단을 피하고 유사 스트리밍 추론을 향상시키기 위해 입력 절단을 사용했다.
이러한 방법들은 Whisper의 파인튜닝을 요구하지 않는다.
하지만, 둘다 WHIsper을 완전히 스트리밍 모델로 변환한 것은 아니다.
여전히 Whisper의 오리지널 비스트리밍 인터페이스에 의존하고있기 때문이다.
오로지 sequence-to-sequence 목표로 훈련된 모델로 불완전한 오디오 세그먼트에서 추론하는 것은 훈련과 추론 사이의 불일치를 이끌며, 낮은 레이턴시 시나리오에서의 정확도를 떨어뜨린다.
더 나아가, 추론 효율성이 떨어진다.
이러한 접근법은 반복적으로 같은 오디오를 연산하기 때문이다.
추가적으로로, Whisper 모델은 30초의 길이로 입력이 패딩된 상태에서 훈련되었다.
이로 인해 추론 단계에서는 실제 입력 길이와 관계없이 항상 30초로 입력을 패딩해야 하며, 이는 추가적인 계산 부담을 초래한다.

이에 반해, 대부분 스트리밍 ASR 모델들은 정렬 없는 훈련을 통해 부분 입력에 대해 높은 퀄리티의 부분 전사를 강요한다.
예를들어, CTC[^4] 와 RNN-T[^5] 아키텍처들은 근본적으로 실시간 스트리밍에 잘 맞는다.
왜냐하면 훈련에서 암묵적인 정렬을 통해 인과성(causality) 달성하기 때문이다.
WeNet[^7][^8]로 구현된 U2 모델[^6]은 Whisper을 스트리밍 음성 인식에 적용시키기 위한 자연스러운 해법을 제공한다.
이는 encoder-decoder 아키텍처에 스트리밍 기능을 결합하고 있기 때문이다.

우리는 U2 Whisper을 제안한다.
이는 U2 모델 아키텍처에서 Whisper을 스트리밍 ASR로 적용한 것이다.
구체적으로, CTC 디코더를 Whisper 인코더 위에 도입했고, 하이브리드 CTC-attention 로스를 활용하여 파인튜닝한다.
CTC 브랜치(branch)는 스트리밍 인식을 위해 훈련되었고, 원본 Whisper 디코더 브랜치(branch)는 원래의 sequence-to-sequence 셋업을 유지한다.
추론하는 동안, CTC 디코더는 스트리밍 부분 가설(hypotheses)을 생성한다.
종단점이 감지되면, Whisper 디코더는 최종 결과를 결정하기 위해 리스코어링으로 사용된다.
이러한 방식은 Whisper을 인과적 스트리밍 모델로 변환시킬 뿐만 아니라 실시간 상황에서 CPU를 활용한 모델 추론 과정에서도 효율성을 향상시킨다.
그럼에도 불구하고, ablation study U2 경로 단독으로는 파인튜닝에 사용되는 데이터가 매우 제한적인 도전적인 경우에서는 테스트 셋에서 일반화가 잘되지 않을 수 있음이 나타났다이
이에따라, 우리는 CTC 디코더의 토큰 공간을 축소하고, 특히, 자원이 적은 조건에서 희귀 단어를 잘 다루기 위해 더 세분화된 subword 토큰 모델링을 활용했다.[^9]
CTC decoder 을 위해 Whisper의 오리지널 토크나이저의 첫 8,000 개의 토큰을 선택한다.
반면에 Whisper decoder에서는 원본 tokenizer로 유지한다.
실험결과 하이브리드 토크나이저 기법은 일반화를 향상시켰음을 보였으며, 특히 훈련 데이터가 부족할 때 효과적이었다.

## 2. Methods

### 2.1. Adapting Whisper to a U2 model

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

[^1]: A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proc. of the 40th International Conference on Machine Learning, 2023, pp. 28492–28518.
[^2]: D.Mach´aˇcek, R. Dabre, and O. Bojar, “Turning whisper into real time transcription system,” in Proc. of the 13th International Joint Conference on Natural Language Processing and the 3rd Confer ence of the Asia-Pacific Chapter of the Association for Computational Linguistics: System Demonstrations, 2023, pp. 17–24.
[^3]: H. Wang, G. Hu, G. Lin, W.-Q. Zhang, and J. Li, “Simul-whisper: Attention-guided streaming whisper with truncation detection,” in Proc. INTERSPEECH 2024– 25th Annual Conference of the International Speech Communication Association, 2024, pp. 44834487.
[^4]: A. Graves, S. Fern´ andez, F. Gomez, and J. Schmidhuber, “Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks,” in Proc. of the 23rd International Conference on Machine Learning, 2006, pp. 369376.
[^5]: A. Graves, “Sequence transduction with recurrent neural networks,” in Proc. ICML Representation Learning Workshop, 2012.
[^6]: B.Zhang, D.Wu,Z.Yao,X.Wang,F.Yu,C.Yang, L.Guo, Y.Hu, L. Xie, and X. Lei, “Unified streaming and non-streaming twopass end-to-end model for speech recognition,” arXiv preprint arXiv:2012.05481, 2020.
[^7]: Z. Yao, X. W.DiWu,B.Zhang,F.Yu, C.Yang, Z.Peng, X. Chen, L. Xie, and X. Lei, “Wenet: Production oriented streaming and non-streaming end-to-end speech recognition toolkit,” in Proc. INTERSPEECH 2021– 22nd Annual Conference of the International Speech Communication Association, Brno, Czech Republic, Aug. 2021, pp. 4454–4458.
[^8]: B. Zhang, D. Wu, Z. Peng, X. Song, Z. Yao, H. Lv, L. Xie, C. Yang, F. Pan, and J. Niu, “Wenet 2.0: More productive endto-end speech recognition toolkit,” in Proc. INTERSPEECH 2022– 23rd Annual Conference of the International Speech Communication Association, Incheon, Korea, Sep. 2022, pp. 1661–1665.
[^9]: R. Sennrich, B. Haddow, and A. Birch, “Neural machine translation of rare words with subword units,” in Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), K. Erk and N. A. Smith, Eds. Berlin, Germany: Association for Computational Linguistics, Aug. 2016, pp. 1715–1725. [Online]. Available: https://aclanthology.org/P16-1162/
[^10]: S. Watanabe, T. Hori, S. Kim, J. R. Hershey, and T. Hayashi, “Hybrid ctc/attention architecture for end-to-end speech recognition,” IEEE Journal of Selected Topics in Signal Processing, vol. 11, no. 8, pp. 1240–1253, 2017.
[^11]: P. Gage, “A new algorithm for data compression,” The C Users Journal, vol. 12, no. 2, pp. 23–38, 1994.
[^12]: A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, “Language models are unsupervised multitask learners,” OpenAI blog, vol. 1, no. 9, 2019.
[^13]: T. Kudo and J. Richardson, “Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing,” 2018. [Online]. Available: https://arxiv.org/abs/1808.06226
[^14]: J. Ansel, E. Yang, H. He, and et al., “PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation,” in 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS ’24). ACM, Apr. 2024. [Online]. Available: https://pytorch.org/assets/pytorch2-2.pdf
[^15]: V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Librispeech: an asr corpus based on public domain audio books,” in 2015 IEEE International Conference on Acoustics, Speech and Signal processing (ICASSP). IEEE, 2015, pp. 5206–5210.
[^16]: M. Del Rio, P. Ha, Q. McNamara, C. Miller, and S. Chandra, “Earnings-22: A practical benchmark for accents in the wild,”
 arXiv preprint arXiv:2203.15591, 2022
