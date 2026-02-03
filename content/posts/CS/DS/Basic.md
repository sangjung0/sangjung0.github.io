---
title: Basic
subtitle: 기초
draft: false
date: 2025-11-24 16:03:00 +0900
categories: [CS, Data Structure]
tags: [CS, Data Structure]
math: true
mermaid: true
---

## Abstract Data Type: 추상적 데이터 타입 (인터페이스 명세, 클래스의 설계도)

데이터 객체 및 연산의 명세와 데이터 객체의 내부 표현양식과 연산의 구현 내용을 분리한 것을 의미한다.  
예시로 Ada package와 C++ class가 있다.

+ ADT에서 연산의 명세
  + 구성 요소: 함수 이름, 인자들의 타입, 결과들의 타입
  + 함수의 호출 방법 및 결과물이 무엇인지 설명
  + 함수의 내부 동작과정 및 구현 방법은 은폐

+ 연산 명세에서 내부 함수들의 종류
  + 생성자 (Creator/Constructor): 새로운 인스턴스 생성
  + Transformer: 기존 인스턴스를 이용해 새로운 인스턴스 생성
  + 관찰자(Observer/Reporter): 인스턴스에 대한 정보를 출력

## Complexity 복잡도

시간복잡도와 공간복잡도가 있으며, 성능 분석과 측정에 사용되는 지표이다.

+ Space Complexity (공간 복잡도): 프로그램에 요구되는 공간으로, 고정적인 공간과 가변적인 공간 요구 사항이 있다. $ S(P) = c + S_p(I) $ 로 나타내며, $ c $ 는 고정적으로 요구되는 공간, $ S_p(I) $ 는 가변적으로 요구되는 공간이다.
+ Time Complexity (시간 복잡도): 프로그램의 실행시간이다. $ T_p = C + E $ 로 나타낼 수 있으며, $ C $는 컴파일 시간, $ E $는 실행 시간이다.
  + 근사 표현: $ O, \Omega, \Theta $ 로 표현가능 하며, 각각 상한, 하한 그리고 상한이자 하한을 의미한다. $ \Theta $ 는 $ O $ 와 $ \Omega $ 가 같은 상황이다.
  