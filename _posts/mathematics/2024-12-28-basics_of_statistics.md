---
title: 통계학 기초
date: 2024-12-28 19:12:00 +0900
categories: [Mathematics]
tags: [Mathematics, Statistics, Basics]
pin: false
math: true
mermaid: true
img_path: /img/mathematics/basics_of_statistics
---

## [**Statistics**](https://ko.wikipedia.org/wiki/%ED%86%B5%EA%B3%84%ED%95%99) (통계학)
통계학은 다량의 데이터를 관찰하고 정리 및 분석하는 방법을 연구하는 수학의 한 분야입니다.  
귀납적 추론에 가까운 학문으로, 특정 상황에서는 연역법도 보조적으로 사용됩니다.


## 용어

### 모집단과 표본
- **Population (모집단)**  
  분석 대상 전체 집합을 의미합니다.  
  모집단은 전체 데이터를 포함하며, 연구나 분석의 기준점이 됩니다.  

- **Sample (표본)**  
  모집단에서 추출한 일부 데이터 집합입니다.  
  표본은 모집단의 특성을 추정하기 위해 사용되며, 적절한 표본은 모집단을 잘 대표해야 합니다.  



### 평균과 분산
- **Population Mean (모평균)**  
  모집단의 평균값으로, 모집단에 속한 모든 데이터의 합을 데이터 개수로 나눕니다.  
  $$\mu = \frac{1}{N} \sum_{i=1}^{N} X_i$$

- **Sample Mean (표본평균)**  
  표본 데이터의 평균값으로, 모집단이 아닌 표본 데이터에서 계산한 평균입니다.  
  $$\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i$$

- **Population Variance (모분산)**  
  모집단의 분산으로, 모집단 데이터가 평균으로부터 얼마나 퍼져 있는지를 나타냅니다.  
  $$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (X_i - \mu)^2$$

- **Sample Variance (표본분산)**  
  표본의 분산으로, 표본 데이터가 평균으로부터 얼마나 퍼져 있는지를 나타냅니다. 분산 계산 시 \(n-1\)을 사용해 편향을 보정합니다.  
  $$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})^2$$

- **Standard Deviation (표준편차)**  
  분산의 제곱근으로, 데이터가 평균으로부터 얼마나 떨어져 있는지를 나타냅니다. 표준편차는 분산보다 더 직관적인 해석이 가능합니다.  
  $$s = \sqrt{s^2}$$



### 대푯값 (Central Tendency)
- **Mean (평균)**  
  데이터의 중심값으로, 일반적으로 산술평균을 의미합니다.  
  평균을 나타내는 기법에는 산술평균, 기하평균, 조화평균이 있으며 기법들마다 이상치의 영향이 다릅니다.  
  이상치의 영향은 산술평균, 기하평균, 조화평균 순으로 강하다. 따라서, 산술평균을 하였을 때, 이상치의 영향이 강하면 기하평균 또는 조화평균을 고려할 수 있다.  

- **Arithmetic Mean (산술평균)**  
  데이터 값을 단순히 더한 후 데이터 개수로 나눈 값입니다.  
  $$\text{AM} = \frac{1}{n} \sum_{i=1}^{n} X_i$$

- **Geometric Mean (기하평균)**  
  데이터 값을 모두 곱한 후 데이터 개수의 제곱근을 계산합니다.  
  $$\text{GM} = \left( \prod_{i=1}^{n} X_i \right)^{\frac{1}{n}}$$

- **Harmonic Mean (조화평균)**  
  데이터 값의 역수의 평균을 계산한 후 역수를 취합니다.  
  $$\text{HM} = \frac{n}{\sum_{i=1}^{n} \frac{1}{X_i}}$$

- **Median (중앙값)**  
  데이터를 크기 순으로 정렬했을 때 정중앙에 위치하는 값입니다. 
  중앙값은 이상치에 영향을 받지 않기 때문에 대푯값으로 자주 사용됩니다.



### 확률
- **Probability (확률)**  
  특정 사건이 발생할 가능성을 나타내는 수치입니다.

- **Random Variable (확률변수)**  
  확률 실험의 결과를 수치로 표현한 변수입니다.  

  - **Discrete Random Variable (이산 확률변수)**: 값이 특정한 이산적인 형태로 나타남 (예: 주사위 눈의 값).  
  - **Continuous Random Variable (연속 확률변수)**: 값이 연속적인 형태로 나타남 (예: 키, 몸무게).

- **Probability Density (확률 밀도)**  
  확률 변수가 나타날 가능성의 상대적 크기입니다.

- **Probability Density Function (확률 밀도 함수)**  
  Standard Probability Density Function이란, Continuous Random Variable를 x축으로, Probability Density를 y축으로 하며, 넓이가 1인 Standard Normal Distribution (표준 정규 분포, mean 0, standard deviation 1) 그래프를 말합니다.  
  $$f(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$$  
  위의 식은 Standard Probability Density Function 그래프 식입니다.  
  $$P(a \leq X \leq b) = \int_{a}^{b} \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}} \, dx$$  
  Standard Probabiliity Density Function 에서 특정 범위에 대한 넓이는 해당 범위에서 Random Variable 범위가 발생할 Probability입니다.  
  따라서 위와같이 특정 범위에 대한 Probability로 나타낼 수 있습니다.  
  Standard Probability Density Function은 mean 0과 standard deviataion 1인 data set에 대한 probability density를 나타냅니다.  
  따라서, 다른 data set에 적용하려면 mean과 standard deviation를 그에 맞게 적용해줘어야 합니다.  
  $$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$  
  위 식은 Normal Distribution (일반 정규 분포) 식으로, 이를 통해 Probability Density Function를 만들 수 있습니다.  
  위 식을 활용하여 넓이가 1이며, mean과 standard deviation이 원하는 data set과 동일한 Probability Density Function이 만들어집니다.  

- **Probability Mass Function (확률 질량 함수)**  
  Probability Mass Function은 x축은 Discrete Random Varable이 될 수 있는 특정 값들이며, y축은 해당 Discrete Random Varable가 나올 확률입니다.  
  $$P(X = x) = f(x)$$  
  위 식처럼 확률이 y축 값이 되는 것입니다.  
  $$\sum_{x \in X} P(X = x) = 1$$  
  따라서 위의 식이 성립하는 그래프가 됩니다.  

- **Expected Value (기댓값)**  
  확률분포에서 가질 수 있는 값들의 가중평균으로, 관찰 가능한 값에 대한 확률로 가중된 평균값입니다.  
  $$E(X) = \sum_{i} X_i \cdot P(X_i)$$  
  위 식은 Discrete Random Varable에 대한 Expoected Value이며,  
  $$E(X) = \int_{-\infty}^\infty x \cdot f(x) \, dx$$  
  위 식은 Continuous Random Varable에 대한 Expected Value입니다.  

- **Confidence Interval (신뢰구간)**  
  모집단의 특정 값이 특정 확률로 포함될 것으로 예상되는 범위입니다.  
  $$\text{Confidence Interval} = \bar{X} \pm Z \cdot \frac{\sigma}{\sqrt{n}}$$  
  - $$\bar{X}$$: 표본 평균
  - $$Z$$: 신뢰도에 따른 Z-값 (표준 정규분포 기준)
  - $$\sigma$$: 모집단 표준편차
  - $$n$$: 표본 크기

- **Confidence Level (신뢰도)**  
신뢰구간이 참 값을 포함할 확률로 표현합니다.  
  - **95% 신뢰도**:
    $$P(-1.96 \leq Z \leq 1.96) \approx 0.95$$
  - **99% 신뢰도**:
    $$P(-2.576 \leq Z \leq 2.576) \approx 0.99$$







## n-moment
  확률변수의 분포를 설명하는 데 사용된다.
  $$ \mu_n' = E(X^n) $$  
  * $$\mu_n'$$: n-moment 원점 기준 n-차 모멘트
  * $$X$$: 확률변수
  * $$E$$: 기댓값
  * $$n$$: 모멘트 차수


### 1-moment (평균)  
  데이터의 대푯값 즉 평균이다.  
  원점 기준: $$ \mu_1' = E(X) $$  
  중심 기준: $$ \mu_1 = E(X - \mu) = 0 $$  
  $$ \therefore \quad \frac{1}{n}E(X) = E(\frac{1}{n})$$  

### 2-moment (분산)
  데이터가 평균을 중심으로 얼마나 퍼져있는지 알 수 있는 분산을 의미한다.  
  원점 기준: $$ \mu_2' = E(X^2) $$  
  중심 기준: $$ \mu_2 = E((X - \mu)^2) = \sigma^2 $$  
  $$ \therefore \quad \frac{1}{n^2}V(X) = V(\frac{X}{n}) $$  
  중심 기준에서는 분산이 된다.  

### 3-moment (왜도)
  데이터 분포의 대칭성 또는 비대칭성을 나타낸다.  
  $$ \mu_3 = E((X - \mu)^3) $$  
  $$ \text{Skewness} = \frac{\mu_3}{\sigma^3} $$  
  $$ \therefore \quad \frac{1}{n^3}S(X) = S(\frac{X}{n}) $$  

  * $$u_3>0$$: 오른쪽 꼬리가 더 큼  
  * $$u_3<0$$: 왼쪽 꼬리가 더 큼  
  * $$u_3=0$$: 대칭적 분포  

### 4-moment (첨도)
  데이터 분포의 뾰족함 또는 꼬리의 두께를 나타낸다.  
  $$ \mu_4 = E((X - \mu)^4) $$  
  $$ \text{Kurtosis} = \frac{\mu_4}{\sigma^4} $$  
  $$ \therefore \quad \frac{1}{n^4}K(X) = K(\frac{X}{n}) $$  

  * $$u_4<3$$: 평평한 분포  
  * $$u_4=3$$: 정규분포와 비슷한 분포  
  * $$u_4>3$$: 뾰족하고 꼬리가 두꺼운 분포  

## Lemma 1
  정리: $$ E(X^2) = \sigma^2 + m^2 $$  
  $$ X \subseteq \{x_1, x_2, \dots, x_n\} $$  
  $$ E(X) = \sum_{i} x_i \cdot P(X = x_i) $$  
  $$ V(X) = \sum_{i} (x_i - E(X))^2 \cdot P(X = x_i) $$  
  $$ \sigma^2 = V(X) = \frac{(x_1-m)^2 + \, \dots \, + (x_n-m)^2}{n} = \frac{1}{n}(x_1^2 + \dots + x_n^2 - 2m(x_1+ \dots + x_n) + nm^2) $$  
  $$ E(X^2) = x_1^2 \cdot P(x_1^2) + \, \dots \, + x_n^2 \cdot P(x_n^2) = \frac{1}{n} \sum_{i=1}^n x_i $$  
  $$ \therefore \sigma^2 = E(X^2) - m^2 \therefore E(X^2) = \sigma^2 + m^2 $$  

## Lemma 2
  정리: $$ \frac{1}{n^2}V(X) = V(\frac{X}{n}) \quad \rightarrow \quad V(\bar{X})=\frac{1}{n}V(X) $$  
  표본 평균: 모집단에서 나올 수 있는 특정 표본의 평균이다. $$ \bar{X} = \frac{1}{n} \sum_{i=1}^n X_i $$  
  표본이 독립 동일 분포(표본이 같은 분산을 가진다면)라면, 아래식이 성립한다.  
  $$ V(\bar{X}) = V(\frac{x_i + \dots + x_n}{n}) = \frac{1}{n^2}V(x_i + \dots + x_n) = \frac{1}{n}V(X) $$  
  따라서,  
  표본 분산이 $$ s^2 $$이며, 모분산이 $$ \sigma^2 $$일 때,  
  $$
    \begin{align*}
      s^2 &= \frac{1}{n}\sum_{i=1}^n(X_i-\bar{X})^2 \\
      E(s^2) &= \frac{1}{n}E(X_1^2 + \dots + X_n^2 -2X_1\bar{X} + \dots + -2X_n\bar{X} + n\bar{X}^2) \\
             &= \frac{1}{n}\{nE(X^2) - 2nE(\bar{X})E(\bar{X}) + nE(\bar{X}^2)\} \\
             &= E(X^2) - 2E(\bar{X})^2 + E(\bar{X}^2) \\
             &= \sigma^2 + m^2 -2m^2 + \frac{\sigma^2}{n} + m^2\\
             &= \frac{n+1}{n}\sigma^2 \\
      \therefore \sigma^2 &= \frac{1}{n-1}\sum_{i=1}^n(X_i-\bar{X})^2 \\
    \end{align*} $$  
  위 식을 만족하게 된다.  
  따라서, 표본에서 편향되지 않은 분산을 구하기 위해서는 n 대신 n-1을 나누어줘야 한다.  


## Distribution (분포)  

### Discrete Distribution (이산 분포)
  확률 변수가 특정한 이산적인 값을 가질 때의 분포이다.  

  - **Binomial Distribution (이항 분포)**  
    고정된 횟수의 독립적인 시행에서 성공할 확률을 모델링한 분포이다.  
    성공과 실패로 이루어진 실험에서 사용된다.  
    + 표기법: $$ X \sim B(n, k) $$  
      * $$ n $$: 개수/횟수  
      * $$ k $$: 확률  
      $$ \therefore E(B(n,k)) = nk $$  
    + 식: $$ P(X = k) = \binom{n}{k} p^k (1-p)^{n-k} $$  
      * $$ n $$: 시행 횟수  
      * $$ k $$: 성공 횟수  
      * $$ p $$: 성공 확률  

  - **Poisson Distribution (포아송 분포)**  
    일정한 시간이나 공간에서 발생하는 사건의 수를 모델링하는 분포이다.  
    드물게 발생하는 사건에 사용된다.  
    + 표기법: $$ X \sim \text{Poisson}(\lambda)  
      * $$ \lambda $$: 단위 시간/공간당 평균 발생률  
    + 식: $$ P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!} $$  
      * $$ \lambda $$: 평균 발생률  
      * $$ k $$: 사건의 수  

  - **Geometric Distribution (기하 분포)**  
    첫 성공이 나타날 때까지 실패 횟수를 모델링한 분포이다.  
    + 표기법: $$ X \sim \text{Geom}(p)  
      * $$ p $$: 성공확률  
    + 식: $$ P(X = k) = (1-p)^{k-1} p $$  
      * $$ p $$: 성공 확률  
      * $$ k $$: 첫 성공이 나타나는 시점

### Continuous Distribution (연속 분포)
  확률변수가 연속적인 값을 가질 때의 분포이다.  
  모든 값이 특정 구간 내에 있을 확률을 계산한다.  

  - **Normal Distribution (정규 분포)**  
    데이터가 평균을 중심으로 대칭적인 종 모양의 분포를 따를 때 사용한다.  
    + 표기법: $$ X \sim N(\mu, \sigma^2) $$  
      * $$ \mu $$: 평균  
      * $$ \sigma^2 $$: 분산  
    + 식: $$ f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} $$  
      * $$ \mu $$: 평균  
      * $$ \sigma^2 $$: 분산  
    + Z-Score Normalization (Z-Score 정규화)        
      모든 데이터가 평균이 0이며 표준편차 1을 갖도록 조정하는 정규화 방법이다.  
      식: $$ z = \frac{x-\mu}{\sigma} $$  
  
  - **t-Distribution (t-분포, t-검정)**  
    표본 크기가 작은 경우나 모분산을 모를 때 평균 비교를 위해 사용한다.  
    + 표기법: $$ X \sim t(\nu) $$  
      * $$ \nu $$: 자유도  
    + 식: $$ f(t) = \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\sqrt{\nu\pi} \, \Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{t^2}{\nu}\right)^{-\frac{\nu+1}{2}} $$  
      * $$ \nu $$: 자유도  
    + Mean Testing (평균 검증)  
      데이터의 집단 평균이 특정 값 또는 다른 집단의 평균과 유의미하게 다른지를 검증하기 위한 통계 방법이다.  
      모집단 평균 추론 또는 집단 간 비교를 위해 사용된다.  
      1. 가설 세우기  
         - 귀무가설 ($$ H_0 $$): 평균에 차이가 없다를 말하려 하며, 이미 널리 알려진 사실을 말한다.  
         - 대립가설 ($$ H_1 $$): 평균에 차이가 있음을 말하려 하며, 주장하려는 가설이다.  
      2. 검정 통계량 계산  
         - One-Sample t-Test (단일 표본 t-검정, 대립 표본 t-검정)    
          모집단 평균이 특정 값과 유의미하게 다른지 검정한다.  
          식: $$ t = \frac{\bar{x}-\mu}{s/\sqrt{n}} $$  
          * $$ \bar{x} $$: 표본 평균  
          * $$ \mu $$: 모집단 평균  
          * $$ s $$: 표본 표준편차  
          * $$ n $$: 표본 크기  
         - Independent t-Test (독립 표본 t-검정)  
          두 독립된 집단의 평균이 같은지 검정한다.  
         - Paired t-Test (대응 표본 t-검정)  
          동일한 집단에서 처리 전후의 평균 차이를 검정한다.  
         - z-test (z-검정)  
          모집단의 분산이 알려져 있거나, 표본 크기가 큰 경우(정규성을 띄면) 사용한다.  
          식: $$ z = \frac{\bar{x} - \mu}{\sigma / \sqrt{n}} $$  
          * $$ \bar{x} $$: 표본 평균  
          * $$ \mu $$: 모집단 평균  
          * $$ \sigma $$: 모집단 표준편차  
          * $$ n $$: 표본 크기  
         - ANOVA (분산분석)  
          세 집단 이상의 평균을 비교할 때 사용한다.  
      3. 유의 수준 설정  
        귀무가설을 기각할 기준을 설정하는 값으로, "오류를 허용할 확률"을 의미한다.  
        일반적으로 0.05를 사용한다.  
      4. p-value  
        검정통계량을 기반으로 p-value를 구한다.  
        검정통계량의 값이다.  
      5. 결론 도출  
        - $$ \text{p-value} < \sigma $$: 귀무 가설 기각 (평균 차이가 있다)  
        - $$ \text{p-value} < \sigma $$: 귀무 가설 채택 (평균 차이가 없다)  

  - **F-Distribution (F-분포, F-검정)**  
    두 집단의 분산 비율을 비교할 때 사용한다.  
    + 표기법: $$ X \sim F(d_1, d_2) $$  
      * $$ d_1 $$: 분자 자유도  
      * $$ d_2 $$: 분모 자유도  
    + 식: $$ f(x) = \frac{\sqrt{\left(\frac{d_1 x}{d_1 + d_2}\right)^{d_1} \left(\frac{d_2}{d_1 + d_2 x}\right)^{d_2}}}{x B\left(\frac{d_1}{2}, \frac{d_2}{2}\right)} $$  
      * $$ d1, d2 $$: 각각의 자유도  

  - **Chi-Square Distribution (카이 분포, 카이 검정)**  
    독립적이고 정규분포를 따르는 확률변수들의 제곱합의 분포를 확인하기 위해 사용한다.  
    범주형 데이터의 적합성 검정이나 독립성 검정을 위해서도 사용된다.
    + 표기법: $$ X \sim \chi^2(k) $$  
      * $$ k $$: 자유도  
    + 식: $$ f(x) = \frac{1}{2^{k/2} \Gamma(k/2)} x^{k/2-1} e^{-x/2} $$  
      * $$ k $$: 자유도   


