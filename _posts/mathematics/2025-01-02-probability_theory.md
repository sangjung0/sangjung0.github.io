---
title: 확률론
date: 2025-01-02 15:49:00 +0900
categories: [Mathematics]
tags: [Mathematics, Probability, Theory]
pin: false
math: true
mermaid: true
img_path: /img/mathematics/probability_theory
---

## Probability Theory (확률론)
  [**확률론**](https://ko.wikipedia.org/wiki/%ED%99%95%EB%A5%A0%EB%A1%A0)이란 확률에 대해 연구하는 수학의 한 분야이다.  

## 학파

### 빈도주의 학파
  실제로 시행 가능하며, 그것을 무한히 할 수 있는 것만을 정의한다.  
  확률은 무한대로 시행한 것을 의미한다.  
  인공지능에서는 안쓰인다.  

### 베이지안 학파
  통계적 해석을 바탕으로 하며, "주관적인 믿음"으로 정의한다.  
  인공지능 논리의 기반이 된다.  


## Conditional Probability (조건부 확률)
  식: $ P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad \text{단 } P(B) > 0 $  
  사건이 독립이면  
  $$ \begin{align*}
    P(A \cap B) &= 0 \\
    P(A \cup B) &= P(A) + P(B) \\
    P(A|B) &= 0 
  \end{align*} $$  
  독립이 아니라면  
  $$ \begin{align*}
    P(A \cup B) &= P(A) + P(B) - P(A \cap B) \\
  \end{align*} $$  
  베이즈 정리: $ \frac{P(A|B)P(B)}{P(A)} = P(B|A) $