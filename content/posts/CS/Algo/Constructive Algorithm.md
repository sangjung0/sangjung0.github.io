---
title: Constructive Algorithm
subtitle: 구성적 알고리즘
draft: false
date: 2025-11-24 16:23:00 +0900
categories: [CS, Algorithm]
tags: [CS, Algorithm, Search]
math: true
mermaid: true
---

## Permutations: 순열

```text
function permutation(list, start, n){
    if (start == n) {
        list // permutation의 한 원소
    } else {
        for ( j=start; j <= n; j++ ) {
            // 순차적으로 자리 교체하며, 모든 순열 순회
            swap(list[start], list[j])
            perm(list, start + 1, n)
            swap(list[start], list[j])
        }
    }
}
```

+ 배열의 모든 순열을 출력하는 재귀함수
+ 성능: $ O(n^2n!) $

## Magic Square, Siamese method

+ 중앙 맨 위를 1로하여, n*n 까지 왼쪽 위로 대각선으로 채우는 방법. 사방이 동일한 매직스퀘어 있다고 가정.
+ 성능: $ O(n^2) $

## Equivalence Relations: 동치관계

```text
function equivalence() {
    while ( there are more pairs ) {
        read the next pair <i, j>;
        put j on the seq[i] list;
        put i on the seq[j] list;
    }
}
```

+ 동치 관계를 담고 있는 배열을 통해 집합을 동치 클래스로 분할하는 알고리즘.
+ 출력할 때는, 리스트를 순회하며 출력하면 되며, 각 값에 해당하는 동치도 출력해야한다. 이미 출력했으면 출력하지 않는다.
