---
title: 선형 대수 기초
date: 2024-12-29 17:13:00 +0900
categories: [Mathematics]
tags: [Mathematics, Linear Algebra, Basics]
pin: false
math: true
mermaid: true
---

## Linear Algebra (선형 대수학)
  [**선형대수학**](https://ko.wikipedia.org/wiki/%EC%84%A0%ED%98%95%EB%8C%80%EC%88%98%ED%95%99)은 벡터 공간, 벡터, 선형 변환, 행렬, 연립 선형 방정식 등을 연구하는 대수학의 한 분야이다.  

## 용어

### 선형
  $ f(ax + by) = af(x) + bf(y) $  
  선형이란 위 식을 만족하는 것을 말한다.  
  
### Vector
  Vector는 Scalar + Direction이다.  

### Matrix
  Matrix는 Concat of Vector이다.

### Gradient
  각 축의 기울기이다.  
  $$ \nabla f(x, y, z) = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z} \right) $$  

### Laplacian
  그라디언트의 총 변화율이다.  
  $$ \nabla f(x, y) = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right) $$  


## 행렬 연산
  교환법칙은 안되며, 분배법칙만 된다. (모노이드)  
  - $A^{-1}$ 역행렬: $A^{-1} = \frac{1}{\text{det}(A)}\cdot \text{adj}(A)$
  - $A^T$ 전치행렬: $A=\begin{bmatrix} a & b \\ c & d\end{bmatrix}, A^T=\begin{bmatrix} a & c \\ b & d \end{bmatrix}$  
  - $(AB)^{-1} = B^{-1}A^{-1}$  
  - $(A^T)^{-1} = (A^{-1})^T$  
  - $(kA)^{-1} = \frac{1}{k}A^{-1} \rightarrow kA(kA)^{-1} = I \therefore kA\frac{1}{k}A^{-1} = I$

## Square Matrix (정사각 행렬)
  - I (항등 행렬)  
    $ AI = A $  
    행렬곱에 대한 항등원이다. 위 식에서 $I$를 의미한다.  
    $I$는 모든 값이 1인 Diagonal Matrix (대각 행렬)이다.  
    $$
        I = \begin{bmatrix}
                1 & 0 \\
                0 & 1
            \end{bmatrix}
    $$  
  - $ A^{-1} $ (역행렬)  
    $ A \cdot A^{-1} = I $  
    행렬곱에 대한 역원이며, 위 식에서 $A^{-1}$을 의미한다.  
    역행렬은 행렬에 따라 존재할 수도 있으며, 없을 수도 있다.  
    역행렬을 구하는 방법은 아래와 같다.  
    $$ \begin{align*}
        &A = \begin{bmatrix}
            a & b \\
            c & d
        \end{bmatrix} \, , \quad
        A^{-1} = \begin{bmatrix}
            e & f \\
            g & h 
        \end{bmatrix} \\
        &A^{-1} = \frac{1}{ad-bc} 
            \begin{bmatrix} 
                d & -b \\
                -c & a
            \end{bmatrix}
    \end{align*} $$  
    위 식에서 $ad-bc$는 determinant(행렬식)라고 한다.  
    위 식에서 행렬식과 곱해지는 행렬을 adjoint matrix(수반 행렬)이라고 한다.  
    
## Vector
  Vector는 Scalar + Direction이다.  

### 표기법  
  - 일반 표기법: $\vec{v}, \dot{v}, \dot{0}$, 각각 벡터 표기법이며, 마지막은 0벡터이다.  
  - 열벡터: $\begin{bmatrix} a \\ b \end{bmatrix}$  
  - 행벡터: $\begin{bmatrix} a & b & c \end{bmatrix}$   
  
### 연산
  - Scalar Multiplication (스칼라곱): $ t\cdot\vec{V} = \begin{bmatrix} t \cdot a & t \cdot b \end{bmatrix} $  
  - Addition: $ \vec{V1} + \vec{v2} = \begin{bmatrix} a+c & b+d \end{bmatrix} $, 벡터 합은 서로 같은 차원을 가져야한다.  
  - Cross Product (벡터곱, 외적): $ \vec{A} \times \vec{B} = \begin{Vmatrix}\vec{A}\end{Vmatrix}\begin{Vmatrix}\vec{B}\end{Vmatrix}\sin\theta\cdot n $, 두 벡터의 법선벡터 즉 한차원 높은 법선벡터의 단위벡터$n$을 곱하고, 벡터가 이루는 편행사변형의 넓이의 스칼라를 가지는 벡터를 의미한다(아직 이게 무슨 의미를 가지는지는 이해를 못함). 또한 전통적인 외적에서는 2차원에서의 두 벡터에 대한 외적은 차원 확장 없이 스칼라값만 나온다. 3차원에서는 외적이 나오지만, 3개 이상의 벡터에 대한 외적은 정의하지 않는다.  
  - Inner Product (내적): $ \vec{A} \cdot \vec{B} = \begin{Vmatrix}\vec{A}\end{Vmatrix}\begin{Vmatrix}\vec{B}\end{Vmatrix}\cos\theta = a_1b_1 + \dots + a_nb_n $, 두 벡터 중 하나의 벡터가 다른 하나의 벡터 방향으로 프로젝션한 후 스칼라 곱이다.  

### 용어
  - Eigen Value (고유값): $ Ax = bx $ 일 때, x가 벡터 A가 행렬 그리고 b가 상수일 때, 벡터 x가 0이 아닌 벡터일 때, 이 식을 만족하는 b는 행렬 A의 고유값이다. 이 경우 $ \text{det}(A -bI) = 0 $이 성립한다.  
  - Eigen vector (고유 벡터): 위에서 고유값이 존재한다면, x는 해당 고유값에 대응되는 고유벡터이다.   
  - Vector Space (벡터 공간): 벡터로 표현되는 공간이다. 특정 벡터 $\vec{V}$가 특정 벡터 집합으로 표현 가능할 때, 이 벡터 집합을 기저라고 한다. 기저는 가능한 최소 집합으로 한다. 벡터공간은 $ S(V, \times, +) $로 표기된다. 여기서 $\times$는 스칼라곱을 말한다.  
