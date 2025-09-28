---
title: "Streaming ASR 관련 연구 및 실험 설계"
subtitle: ""
draft: false
date: 2025-09-25 15:26:00 +0900
categories: [Project, asr, RT-Whisper]
tags: [Paper, Machine Learning, ASR, Whisper]
math: true
mermaid: true
---


## 개요

트랜스포머 모델인 Whisper를 기반으로 하여 Streaming ASR을 시도한 몇 연구들을 소개합니다.
본 논문의 실험 부분 작성을 위해 실험 계획을 수립합니다.

## 동기

트랜스포머 기반 ASR은 대규모 데이터 학습으로 얻은 강력한 인식 성능과 언어 도메인 일반화 능력을 갖추고 있다.
그 예시가 whisper 이다.
그러나 트랜스포머라는 아키텍처의 한계로 실시간 활용이 어렵다.
이를 가능하게 하기 위한 연구들이 이후 소개되는 연구들이다.

## Streaming ASR 연구

### Whisper Streaming (IJCNLP B등급, 상위 28.1%, 2023년)

+ Local Agreement를 사용하여 문맥 유지하여 스트리밍 구현.
+ 소개 되는 논문에서 가장 기초적인 논문이며, 본 기법에서 영감을 받음.

{{< figure src="/images/paper/asr/turning_whisper_into_real_time_transcription_system/Figure1.png" alt="Figure 1" title="Figure 1" class="center" width="80%" caption="Streaming Whisper">}}

### Simul-whisper (Interspeech A등급, 상위 14.9%, 2024년)

+ 청크의 단어 잘림이 오류의 주요 문제로 지적
+ 청크 내의 단어가 잘리지 않게 적당한 위치에 멈추는 기법 적용
+ AR 과정에서 디코더가 좀 더 주의집중하는 인코더의 키벨류를 중점으로 단어의 위치를 예측함
+ 인코더의 출력을 바탕으로 간단한 선형 계층과 시그모이드 출력층으로 이루어진 모델을 사용한다. 해당 모델을 바탕으로 오디오의 발화 횟수를 계산하고, 해당 발화 횟수만큼 AR 하도록 한다.
+ 프롬프트에 대한 언급은 따로 없으며(아직 Result를 완전히 다 읽지 않음), 청크 오버랩을 하지 않아 연산 시간을 절약할 수 있음

{{< figure src="/images/paper/asr/simul_whisper_attention_guided_streaming_whisper_with_truncation_detection/Figure1.png" alt="Figure1" title="Figure1" class="center" width="50%" caption="Simul-whisper">}}

### U2-whisper  (Interspeech A등급, 상위 14.9%, 2025년)

+ CPU 환경에서의 전사를 목표로 한다.
+ 오디오 프레임과 출력 토큰의 길이는 다르다. 따라서 1:1 학습이 힘들다. 그래서 CTC 로스 적용
+ CTC 로스를 적용한 CTC 디코더와 Whisper decoder의 로스를 함께 사용하여 축소된 공간에서 학습한다.
+ CTC에서 추론된 결과로 여러 후보를 예측하고, 이 후보들을 디코더를 이용해 rescore하여 최종전사를 만든다.
+ 프롬프트를 이용하여 일종의 문맥을 유지하지만, 청크를 오버랩하는 등 프롬프트외 문맥 유지는 없다.

{{< figure src="/images/paper/asr/adaption_whisper_for_streaming_speech_recognition_via_two_pass_decoding_translation/Figure1.png" alt="Figure1" title="Figure1" class="center" width="80%" caption="U2-whisper">}}

## 실험 계획

### 데이터셋

+ ESIC (European Simultaneous Interpretation Corpus): 유럽 의회 발표 데이터셋 (streaming whisper 만든 UFAL 연구소에서 수집 및 제작). 10시간 분량의 데이터셋으로 현장 녹음되어 노이즈가 포함되어 있음.
+ LibriSpeech: LibriVox 프로젝트의 오디오북에서 추출된 1000시간 분량의 영어 음성 데이터셋. 노이즈가 적고 발화 품질이 좋다.
+ VoxPopuli: Meta AI에서 공개한 다국어 음성 데이터셋, 유럽 의회 연설 녹음을 기반으로 구축되었다.
+ Tedlium: TED 강연에서 추출된 452시간 분량의 영어 음성 데이터셋으로, 현장 녹음과 청중의 반응이 포함되어 있어 실시간 환경 평가에 유용하다.
+ KSponSpeech: 2000명의 한국인이 깨끗한 환경에서 녹음한 969시간 분량의 데이터셋이다.
+ Zeroth-Korean: 100명의 한국인이 깨끗한 환경에서 녹음한 53시간 분량의 한국어 음성 데이터셋이다.

한국인이라 한국어 데이터셋도 포함하였다.
4개의 영어 데이터셋과 2개의 한국어 데이터셋이다.
기존 데이터셋에 영어 외의 다른 언어도 포함하고 있으나, 다른 언어는 평가하지 않는다.
평가 점수가 유의미하게 나오고, 시간이 된다면 넣어볼 수 있다.

### 계획

+ ESIC 데이터셋 기준으로 하이퍼파라미터와 모델 학습 및 평가
+ 청크 길이 1초, 2초, 3초 3개의 모델을 학습 및 평가
+ ESIC 기반으로 학습한 모델로 다른 데이터셋을 평가
+ 1초, 2초, 3초 에서 학습한 3개의 모델을 서로 다른 환경에서 평가
+ ESIC가 아닌 다른 데이터셋에서 1초, 2초, 3초로 모델을 학습하여 평가하고, ESIC 데이터셋과 파라미터를 비교하여 데이터셋별 파라미터 변화 평가

### 현재 진행 상황

| 데이터셋 | Faster-whisper | Streaming Whisper 1s | Streaming Whisper 2s | Streaming Whisper 3s | w/o 1s (ESIC) | w/o 2s (ESIC) | w/o 3s (ESIC) | 1s (ESIC) | 2s (ESIC) | 3s (ESIC) | w/o 1s | w/o 2s | w/o 3s | 1s | 2s | 3s |
|---------|---------|-------|-------|-------|---------------|---------------|---------------|-----------|-----------|----------|--------|--------|--------|----|----|----|
| ESIC | $7.2%$ | $5.6%$ | $7.6%$ | $5.1%$ | $5.4%$ | $5.4%$ | $5.3%$ | | | | | | | | | |
| LibriSpeech (clean) | $6.4%$ | $8.0%$ | $8.0%$ | $7.8%$ | | | $2.9%$ | | | | | | | | | |
| LibriSpeech (other) | | | | | | | | | | | | | | | | |
| VoxPopuli | | | | | | | | | | | | | | | | |
| Tedlium | $57.3%$ | | | | | | | | | | | | | | | |
| KSponSpeech | $35.5%$ | $35.8%$ | $36.2%$ | $36.1%$ | | | | | | | | | | | | |
| Zeroth-Korean | $26.9%$ | $25.2%$ | $24.2%$ | $24.2%$ | | | | | | | | | | | | |

| 데이터셋 | Faster-whisper | Streaming Whisper 1s | Streaming Whisper 2s | Streaming Whisper 3s | w/o 1s (ESIC) | w/o 2s (ESIC) | w/o 3s (ESIC) | 1s (ESIC) | 2s (ESIC) | 3s (ESIC) | w/o 1s | w/o 2s | w/o 3s | 1s | 2s | 3s |
|---------|---------|-------|-------|-------|---------------|---------------|---------------|-----------|-----------|----------|--------|--------|--------|----|----|----|
| ESIC | X | | | | | | | | | | | | | | | |
| LibriSpeech (clean) | X | | | | | | | | | | | | | | | |
| LibriSpeech (other) | X | | | | | | | | | | | | | | | |
| VoxPopuli | X | | | | | | | | | | | | | | | |
| Tedlium | X | | | | | | | | | | | | | | | |
| KSponSpeech | X | | | | | | | | | | | | | | | |
| Zeroth-Korean | X | | | | | | | | | | | | | | | |

### 예상 기여

1. prompt-free 모델로 확장 가능. (prompt 없이 wer이 높게 나온다.)
2. 고정된 청크 길이로 예측 가능한 latency 및 streaming whisper 보다 빠른 latency
3. 일부 데이터셋에서 향상된 WER
4. 단어 잘림으로 나온 오류는 모델이 아닌 알고리즘으로도 일부 극복 가능

## 향후 연구

+ 레이턴시 측정 방법 변경. 기존 단순한 시간 측정에서 DAL 기법 사용 (입력 및 출력 비율로 입력당 출력률 계산.)
+ 진짜 향후 연구로, 단어 기반이 아닌 토큰 기반으로 확장할 수 있음
