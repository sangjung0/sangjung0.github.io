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

## Statistics (통계학)
[통계학](https://ko.wikipedia.org/wiki/%ED%86%B5%EA%B3%84%ED%95%99)은 다량의 데이터를 관찰하고 정리 및 분석하는 방법을 연구하는 수학의 한 분야이다.  
귀납적 추론에 가까운 학문이며, 특정 상황에서는 연역법도 보조적으로 사용한다.


## 용어

### 모집단과 표본
  - **Population (모집단)**  
    분석 대상 전체 집합을 의미한다.  
    모집단은 전체 데이터를 포함하며, 연구나 분석의 기준점이 된다.  

  - **Sample (표본)**  
    모집단에서 추출한 일부 데이터 집합이다.  
    표본은 모집단의 특성을 추정하기 위해 사용하며, 적절한 표본은 모집단을 잘 대표해야 한다.  

### 평균과 분산
  - **Population Mean (모평균)**  
    모집단의 평균값이다.  
    모집단 전체 데이터의 합을 데이터 개수로 나누어 계산한다.  
    식: $ \mu = \frac{1}{N} \sum_{i=1}^{N} X_i $  

  - **Sample Mean (표본평균)**  
    표본 데이터의 평균값이다.  
    모집단이 아닌 표본에서 계산한 평균이다.  
    식: $ \bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i $  

  - **Population Variance (모분산)**  
    모집단의 분산이다.  
    모집단 데이터가 평균으로부터 얼마나 퍼져 있는지를 나타낸다.  
    식: $ \sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (X_i - \mu)^2 $  

  - **Sample Variance (표본분산)**  
    표본의 분산이다.  
    표본 데이터가 평균으로부터 얼마나 퍼져 있는지를 나타내며, 분산 계산 시 $n-1$로 나누어 편향을 보정한다.  
    식: $ s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})^2 $  

  - **Standard Deviation (표준편차)**  
    분산의 제곱근으로, 데이터가 평균으로부터 얼마나 떨어져 있는지를 나타낸다.  
    분산보다 더 직관적이며 해석이 용이하다.  
    식: $ s = \sqrt{s^2} $  

### 대푯값 (Central Tendency)
  - **Mean (평균)**  
    데이터의 중심값으로, 일반적으로 산술평균을 의미한다.  
    평균을 구하는 기법에는 산술평균, 기하평균, 조화평균이 있으며 각각 이상치에 대한 민감도가 다르다.  
    이상치의 영향은 산술평균 > 기하평균 > 조화평균 순으로 강하므로, 산술평균에서 이상치 영향이 크다면 기하평균이나 조화평균을 고려할 수 있다.  

  - **Arithmetic Mean (산술평균)**  
    데이터 값을 단순히 더한 뒤 데이터 개수로 나눈 값이다.  
    식: $ \text{AM} = \frac{1}{n} \sum_{i=1}^{n} X_i $  

  - **Geometric Mean (기하평균)**  
    모든 데이터 값을 곱한 뒤 데이터 개수만큼 거듭제곱(제곱근)을 취한다.  
    식: $ \text{GM} = \left(\prod_{i=1}^{n} X_i\right)^{\frac{1}{n}} $  

  - **Harmonic Mean (조화평균)**  
    데이터 값들의 역수의 평균을 구한 뒤, 그 역수를 다시 취한다.  
    식: $ \text{HM} = \frac{n}{\sum_{i=1}^{n} \frac{1}{X_i}} $  

  - **Median (중앙값)**  
    데이터를 크기 순으로 나열했을 때 정중앙에 위치하는 값이다.  
    이상치 영향이 적어 대푯값으로 자주 사용된다.  

### 확률
  - **Probability (확률)**  
    특정 사건이 일어날 가능성을 나타내는 수치이다.  

  - **Random Variable (확률변수)**  
    확률 실험의 결과를 수치로 표현한 변수이다.  
    * **Discrete Random Variable (이산 확률변수)**: 값이 특정한 이산적 형태로 나타난다 (예: 주사위 눈).  
    * **Continuous Random Variable (연속 확률변수)**: 값이 연속적으로 나타난다 (예: 키, 몸무게).  

  - **Probability Density (확률 밀도)**  
    확률 변수가 나타날 가능성의 상대적 크기를 의미한다.

  - **Probability Density Function (확률 밀도 함수)**  
    Standard Probability Density Function이란, Continuous Random Variable를 x축으로, Probability Density를 y축으로 하며, 넓이가 1인 Standard Normal Distribution (표준 정규 분포, mean 0, standard deviation 1) 그래프를 말한다.  
    연속 확률변수의 분포를 나타내는 함수이며, 넓이가 1이 되는 곡선 아래의 영역으로 확률을 해석한다.
    * **Standard Probability Density Function (표준 확률 밀도 함수)**  
      평균 0, 표준편차 1을 갖는 확률 밀도 함수이며, 표준 정규 분포 그래프와 동일하다.  
      식: $ f(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}} $  
      특정 구간 $[a, b]$에서의 확률은 아래 적분값으로 정의한다.  
      적분 값: $ P(a \leq X \leq b) = \int_{a}^{b} \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}} \, dx $  

    * **Normal Probability Density Function (일반 확률 밀도 함수)**  
      평균 $\mu$, 표준편차 $\sigma$를 갖는 분포이다. 확률 밀도 함수는 다음과 같다.  
      식: $ f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} $  
      특정 구간에서의 확률은 앞서 언급한 적분하는 방법과 동일하다.  

- **Probability Mass Function (확률 질량 함수)**  
  이산 확률변수에 대한 분포를 나타내는 함수이다.  
  x축은 확률변수가 가질 수 있는 특정 값이고, y축은 그 값이 나타날 확률이다.  
  확률 식: $ P(X = x) = f(x) $  
  그래프: $ \sum_{x \in X} P(X = x) = 1 $  

- **Expected Value (기댓값)**  
  확률분포에서 관찰 가능한 값들이 가중평균된 결과이다.  
  $ E(X) = \sum_{i} X_i \cdot P(X_i) \quad (\text{이산 확률변수}) $  
  $ E(X) = \int_{-\infty}^\infty x \cdot f(x) \, dx \quad (\text{연속 확률변수}) $

- **Confidence Interval (신뢰구간)**  
  모집단의 특정 값(예: 모평균)이 일정 확률로 포함될 것으로 예상되는 구간이다.  
  $ \text{Confidence Interval} = \bar{X} \pm Z \cdot \frac{\sigma}{\sqrt{n}} $  
  - $\bar{X}$: 표본 평균  
  - $Z$: 신뢰도에 따른 Z-값 (표준 정규분포 기준)  
  - $\sigma$: 모집단 표준편차  
  - $n$: 표본 크기  

- **Confidence Level (신뢰도)**  
  신뢰구간이 참값을 포함할 확률을 의미한다.  
  - **95% 신뢰도**: $ P(-1.96 \leq Z \leq 1.96) \approx 0.95 $  
  - **99% 신뢰도**: $ P(-2.576 \leq Z \leq 2.576) \approx 0.99 $  


## n-moment
  확률변수의 분포를 설명하는 데 사용되는 개념으로, $n$차 모멘트라고 한다.
  표기법: $ \mu_n' = E(X^n) $  
  - $\mu_n'$: 원점 기준 $n$차 모멘트  
  - $X$: 확률변수  
  - $E$: 기댓값  
  - $n$: 모멘트 차수  

### 1-moment (평균)  
  데이터의 대푯값, 즉 평균을 의미한다.  
  - 원점 기준: $ \mu_1' = E(X) $  
  - 중심 기준: $ \mu_1 = E(X - \mu) = 0 $  
  - $ \therefore \quad \frac{1}{n}E(X) = E(\frac{1}{n}) $  

### 2-moment (분산)
  데이터가 평균을 중심으로 얼마나 퍼져 있는지 나타내는 분산을 의미한다.  
  - 원점 기준: $ \mu_2' = E(X^2) $  
  - 중심 기준: $ \mu_2 = E((X - \mu)^2) = \sigma^2 $  
  - $ \therefore \quad \frac{1}{n}E(X) = E(\frac{1}{n}) $  

### 3-moment (왜도)
  데이터 분포가 대칭적인지 비대칭적인지를 나타낸다.  
  - $ \mu_3 = E((X - \mu)^3) $, $ \text{Skewness} = \frac{\mu_3}{\sigma^3} $  
    * $\mu_3 > 0$: 오른쪽 꼬리가 더 긴 분포  
    * $\mu_3 < 0$: 왼쪽 꼬리가 더 긴 분포  
    * $\mu_3 = 0$: 대칭적인 분포  

### 4-moment (첨도)
  데이터 분포의 뾰족함 또는 꼬리의 두께를 나타낸다.  
  - $ \mu_4 = E((X - \mu)^4) $, $ \text{Kurtosis} = \frac{\mu_4}{\sigma^4} $  
    * $\mu_4 < 3$: 정규분포보다 꼬리가 얕고 평평한 분포  
    * $\mu_4 = 3$: 정규분포와 유사한 분포  
    * $\mu_4 > 3$: 더 뾰족하고 꼬리가 두꺼운 분포  

## 표본 분산의 편향
  표본 분산 식: $ s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})^2 $  
    - $ s^2 $: 표본 분산  
    - $ n $: 표본의 크기  
    - $ X_i $: 개별 데이터 값  
    - $ \bar{X} $: 표본 평균  
  
  위의 표본분산 식에 따르면 $n$ 대신 $n-1$ 로 나누는 것을 알 수 있다.  
  이는 편향에 대한 조정으로 아래는 그에 대한 증명이다.  

### Lemma 1
  정리: $ E(X^2) = \sigma^2 + m^2 $  
  증명  
  $$ \begin{align*}
    X &\subseteq \{x_1, x_2, \dots, x_n\} \\
    E(X) &= \sum_{i} x_i \cdot P(X = x_i) \\
    V(X) &= \sum_{i} (x_i - E(X))^2 \cdot P(X = x_i) \\
    \sigma^2 = V(X) &= \frac{(x_1-m)^2 + \, \dots \, + (x_n-m)^2}{n} \\
    &= \frac{1}{n}(x_1^2 + \dots + x_n^2 - 2m(x_1+ \dots + x_n) + nm^2) \\
    E(X^2) &= x_1^2 \cdot P(x_1^2) + \, \dots \, + x_n^2 \cdot P(x_n^2) = \frac{1}{n} \sum_{i=1}^n x_i^2 \\
    \therefore \quad \sigma^2 &= E(X^2) - m^2 \rightarrow E(X^2) = \sigma^2 + m^2
  \end{align*} $$  

### Lemma 2
  정리: $ \frac{1}{n^2}V(X) = V\Bigl(\frac{X}{n}\Bigr) \quad \Longrightarrow \quad V(\bar{X}) = \frac{1}{n} V(X) $  
  표본평균 $\bar{X}$는 특정 표본의 평균이며, 식은 다음과 같다. $ \bar{X} = \frac{1}{n}\sum_{i=1}^n X_i $  
  표본이 독립·동일 분포(i.i.d.)라면, 다음 식을 만족한다.  
  $$ \begin{align*}
    V(\bar{X}) &= V\Bigl(\frac{X_1 + \dots + X_n}{n}\Bigr) \\
                &= \frac{1}{n^2}V(X_1 + \dots + X_n) \\
                &= \frac{1}{n}V(X)
  \end{align*} $$  

### 증명
  $$ \begin{align*}
    s^2 &= \frac{1}{n}\sum_{i=1}^n(X_i-\bar{X})^2 \\
    E(s^2) &= \frac{1}{n}E(X_1^2 + \dots + X_n^2 -2X_1\bar{X} + \dots + -2X_n\bar{X} + n\bar{X}^2) \\
            &= \frac{1}{n}\{nE(X^2) - 2nE(\bar{X})E(\bar{X}) + nE(\bar{X}^2)\} \\
            &= E(X^2) - 2E(\bar{X})^2 + E(\bar{X}^2) \\
            &= \sigma^2 + m^2 -2m^2 + \frac{\sigma^2}{n} + m^2\\
            &= \frac{n+1}{n}\sigma^2 \\
    \therefore \quad \sigma^2 &= \frac{1}{n-1}\sum_{i=1}^n(X_i-\bar{X})^2 \\
  \end{align*} $$  
    - $ s^2 $: 표본 분산  
    - $ \sigma^2 $: 모분산  

  따라서, 편향되지 않은(불편) 분산을 구하기 위해서는 $\frac{1}{n-1}$을 사용한다.  


## Distribution (분포)

### Discrete Distribution (이산 분포)
  확률변수가 특정한 이산적인 값을 가질 때의 분포이다.  

  - **Binomial Distribution (이항 분포)**  
    고정된 횟수의 독립적 시행에서 성공할 확률을 모델링한 분포이다. 성공·실패만 존재하는 실험에서 자주 사용한다.
    + 표기법: $ X \sim B(n, p) $  
        - $n$: 시행 횟수  
        - $p$: 성공 확률  
        - $ \therefore E(X) = np $  
      + 식: $ P(X = k) = \binom{n}{k} \, p^k (1-p)^{n-k} $  
        - $k$: 성공 횟수  

  - **Poisson Distribution (포아송 분포)**  
    일정한 시간이나 공간에서 발생하는 사건(드물게 발생하는 사건 포함)의 수를 모델링하는 분포이다.
    + 표기법: $ X \sim \text{Poisson}(\lambda) $  
      - $\lambda$: 단위 시간(공간)당 평균 발생률  
    + 식: $ P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!} $  
      - $\lambda$: 평균 발생률  
      - $k$: 사건의 수  

  - **Geometric Distribution (기하 분포)**  
    첫 성공이 나타날 때까지의 실패 횟수(또는 첫 성공까지 걸리는 횟수 등)를 모델링한 분포이다.  
    + 표기법: $ X \sim \text{Geom}(p) $  
      - $p$: 성공 확률  
    + 식: $ P(X = k) = (1-p)^{k-1} \, p $  
      - $p$: 성공 확률  
      - $k$: 첫 성공이 나타나는 시행 횟수  

### Continuous Distribution (연속 분포)
  확률변수가 연속적인 값을 가질 때의 분포이다.  
  특정 구간 내 모든 값을 가질 수 있으며, 구간 적분을 통해 확률을 구한다.  

  - **Normal Distribution (정규 분포)**  
    데이터가 평균을 중심으로 대칭적인 종 모양의 분포를 따를 때 사용한다.  
    + 표기법: $ X \sim N(\mu, \sigma^2) $  
      - $\mu$: 평균  
      - $\sigma^2$: 분산  
    + 식: $ f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\Bigl(-\frac{(x-\mu)^2}{2\sigma^2}\Bigr) $  
    + **Z-Score Normalization (Z-Score 정규화)**  
      모든 데이터가 평균 0, 표준편차 1을 갖도록 조정하는 방법이다.  
      식: $ z = \frac{x - \mu}{\sigma} $  

  - **t-Distribution (t-분포, t-검정)**  
    표본 크기가 작거나 모분산을 모를 때 평균 비교를 위해 사용한다.  
    + 표기법: $ X \sim t(\nu) $  
      - $\nu$: 자유도  
    + 식: $ f(t) = \frac{\Gamma\Bigl(\frac{\nu + 1}{2}\Bigr)}{\sqrt{\nu\pi}\,\Gamma\Bigl(\frac{\nu}{2}\Bigr)}\Bigl(1 + \frac{t^2}{\nu}\Bigr)^{-\frac{\nu+1}{2}}$
      - $\nu$: 자유도  
    + **Mean Testing (평균 검증)**  
      모집단의 평균이 특정 값 혹은 다른 집단의 평균과 유의미하게 다른지 검증하기 위한 통계 방법이다.  
      1. **가설 세우기**  
         - 귀무가설($H_0$): 평균에 차이가 없다를 말하려 하며, 이미 널리 알려진 사실을 말한다.  
         - 대립가설($H_1$): 평균에 차이가 있음을 말하려 하며, 주장하려는 가설이다.  
      2. **검정 통계량 계산**  
         - **One-Sample t-Test (단일 표본 t-검정, 대립 표본 t-검정)**  
          모집단 평균이 특정 값과 유의미하게 다른지 검정한다.  
          식: $ t = \frac{\bar{x} - \mu}{s/\sqrt{n}} $  
             * $\bar{x}$: 표본 평균  
             * $\mu$: 가정된 모집단 평균  
             * $s$: 표본 표준편차  
             * $n$: 표본 크기  
         - **Independent t-Test (독립 표본 t-검정)**  
          두 독립된 집단의 평균이 같은지 검정한다.  
         - **Paired t-Test (대응 표본 t-검정)**  
          동일한 집단에서 처리 전·후의 평균 차이를 검정한다.  
         - **z-Test (z-검정)**  
          모집단의 분산이 알려져 있거나, 표본 크기가 충분히 커서 정규성 가정이 가능할 때 사용한다.  
          식: $ z = \frac{\bar{x} - \mu}{\sigma / \sqrt{n}} $   
            * $ \bar{x} $: 표본 평균  
            * $ \mu $: 모집단 평균  
            * $ \sigma $: 모집단 표준편차  
            * $ n $: 표본 크기  
         - **ANOVA (분산분석)**  
          세 집단 이상의 평균을 동시에 비교할 때 사용한다.  
      3. **유의 수준($\alpha$) 설정**  
         귀무가설을 기각할 기준(오류 허용 확률)을 의미한다. 보통 0.05 사용한다.  
      4. **p-value**  
         검정통계량으로부터 계산되는 확률값이다.  
      5. **결론 도출**  
         - $p\text{-value} < \alpha$: 귀무가설 기각 (평균에 차이가 있다고 본다)  
         - $p\text{-value} \geq \alpha$: 귀무가설 채택 (평균에 차이가 없다고 본다)  

  - **F-Distribution (F-분포, F-검정)**  
    두 집단의 분산 비율을 비교할 때 사용한다.  
    + 표기법: $ X \sim F(d_1, d_2) $  
      * $d_1$: 분자 자유도  
      * $d_2$: 분모 자유도  
    + 식: $ f(x) = \frac{\sqrt{\Bigl(\frac{d_1 x}{d_1 + d_2}\Bigr)^{d_1}\Bigl(\frac{d_2}{d_1 + d_2 x}\Bigr)^{d_2}}}{\,x \,B\Bigl(\frac{d_1}{2}, \frac{d_2}{2}\Bigr)} $  
      * $ d1, d2 $: 각각의 자유도  

  - **Chi-Square Distribution (카이제곱 분포, 카이 검정)**  
    독립적이고 정규분포를 따르는 확률변수들의 제곱합의 분포를 확인하거나 범주형 데이터의 적합성·독립성 검정에 사용된다.  
    + 표기법: $ X \sim \chi^2(k) $  
      * $k$: 자유도  
    + 식: $ f(x) = \frac{1}{2^{k/2}\,\Gamma\Bigl(\frac{k}{2}\Bigr)}\,x^{k/2 - 1} \exp\Bigl(-\frac{x}{2}\Bigr) $  
      * $k$: 자유도  
