---
title: Revisiting Temporal Modeling for CLIP-based Image-to-Video Knowledge Transferring
date: 2024-09-04 12:26:00 +0900
categories: [paper, abstract]
tags: [AI, Paper, CLIP]
pin: true
img_path: /img/paper/abstract
---

<style>
    .highlight{
        font-weight:bold;
        color: black;
    }
</style>

![Desktop View](Revisiting Temporal Modeling for CLIP-based Image-to-Video Knowledge Transferring.png)
_[**논문 바로가기**](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Revisiting_Temporal_Modeling_for_CLIP-Based_Image-to-Video_Knowledge_Transferring_CVPR_2023_paper.html) / [**pdf 바로가기**](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Revisiting_Temporal_Modeling_for_CLIP-Based_Image-to-Video_Knowledge_Transferring_CVPR_2023_paper.pdf)_


<br/>


![Desktop View](Ruyang Liu.png){: width="1577" height="1333" .w-50 .left}
[**Ruyang Liu**](https://scholar.google.com/citations?user=pZ3sWH0AAAAJ&h) <br/>
2023 <br/>
[**CVPR 2023 Accepted Papers**](https://cvpr.thecvf.com/Conferences/2024/AcceptedPapers) <br/>
<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>

## Abstract

&nbsp; Image-text pretrained models, e.g., <span class="highlight">CLIP,</span> have shown impressive general multi-modal knowledge learned from large-scale image-text data pairs, thus attracting increasing attention for their potential to improve visual representation learning in the video domain. <span class="highlight">In this paper, based on the CLIP model, we revisit temporal modeling in the context of image-to-video knowledge transferring, which is the key point for extending image-text pretrained models to the video domain. We find that current temporal modeling mechanisms are tailored to either high-level semantic-dominant tasks (e.g., retrieval) or low-level visual pattern-dominant tasks (e.g., recognition), and fail to work on the two cases simultaneously. The key difficulty lies in modeling temporal dependency while taking advantage of both high-level and low-level knowledge in CLIP model. To tackle this problem, we present Spatial-Temporal Auxiliary Network (STAN) -- a simple and effective temporal modeling mechanism extending CLIP model to diverse video tasks.</span> Specifically, to realize both low-level and high-level knowledge transferring, STAN adopts a branch structure with decomposed spatial-temporal modules that enable multi-level CLIP features to be spatial-temporally contextualized. We evaluate our method on two representative video tasks: Video-Text Retrieval and Video Recognition. Extensive experiments demonstrate the superiority of our model over the state-of-the-art methods on various datasets, including MSR-VTT, DiDeMo, LSMDC, MSVD, Kinetics-400, and Something-Something-V2. Codes will be available at [**https://github.com/farewellthree/STAN**](https://github.com/farewellthree/STAN)
<br/>

> Translate of ChatGPT
{: .prompt-tip }


> &nbsp;이미지-텍스트 사전 학습 모델, 예를 들어 <span class="highlight">CLIP</span>,은 대규모 이미지-텍스트 데이터 쌍에서 학습된 인상적인 범용 멀티모달 지식을 보여주며, 이를 통해 비디오 도메인에서의 시각적 표현 학습을 개선할 잠재력에 대한 관심이 높아지고 있습니다. <span class="highlight">본 논문에서는 CLIP 모델을 기반으로 이미지에서 비디오로의 지식 전이에 있어 핵심 요소인 시간적 모델링을 재검토합니다. 현재의 시간적 모델링 메커니즘이 고수준의 의미적 패턴이 지배적인 작업(예: 검색) 또는 저수준의 시각적 패턴이 지배적인 작업(예: 인식)에 맞춰져 있어, 두 경우를 동시에 처리하지 못하는 문제점을 발견했습니다. 주요 어려움은 CLIP 모델의 고수준 및 저수준 지식을 모두 활용하면서 시간적 의존성을 모델링하는 데 있습니다. 이러한 문제를 해결하기 위해, 우리는 CLIP 모델을 다양한 비디오 작업으로 확장하는 간단하면서도 효과적인 시간적 모델링 메커니즘인 Spatial-Temporal Auxiliary Network (STAN)를 제안합니다.</span> 구체적으로, 저수준 및 고수준 지식 전이를 모두 실현하기 위해 STAN은 분해된 공간-시간 모듈을 통해 다중 수준의 CLIP 특징들이 공간-시간적으로 맥락화될 수 있는 브랜치 구조를 채택합니다. 우리는 Video-Text Retrieval 및 Video Recognition이라는 두 가지 대표적인 비디오 작업에서 우리의 방법을 평가하였으며, MSR-VTT, DiDeMo, LSMDC, MSVD, Kinetics-400, Something-Something-V2 등 다양한 데이터셋에서 최신 방법들보다 우수한 성능을 보임을 입증했습니다. 코드와 관련 자료는 [**https://github.com/farewellthree/STAN**](https://github.com/farewellthree/STAN)에서 제공될 예정입니다.
<br/>




## Understanding

#### 목표
CLIP과 같은 image-text 모델의 비디오 도메인에 대한 확장
#### 문제
현재 사용되는 시간적 모델링 기법들은 검색과 같은 고수준의 의미적 패턴이 지배적인 작업 또는 인식같은 저수준의 시각적 패턴이 지배적인 작업에 맞춰져 있어, 두 경우를 동시에 처리하지 못한다는 문제가 있다.
#### 제안
STAN( Spatial-Temporal Auxiliary Network)이라는 새로운 시간적 모델링 메커니즘을 제안한다.
#### 성과
다양한 데이터 셋에서 최신 방법들보다 우수한 성능을 보임을 입증했다.






