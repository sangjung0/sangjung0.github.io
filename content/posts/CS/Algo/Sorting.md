---
title: Sorting
subtitle: 정렬
draft: false
date: 2025-11-24 15:28:00 +0900
categories: [CS, Algorithm]
tags: [CS, Algorithm, Sorting]
math: true
mermaid: true
---

## Selection Sorting: 선택 정렬

```text
for(i=0; i < n-1; i++){
    min = i; // 기준값 위치
    for (j = i+1; j<n; j++){
        if (list[j] < list[min]) { // 만약, 기준값 보다 작다면
            min = j;  // 새로운 기준값 인덱스 할당
        }
    }
    SWAP(list[i], list[min]) // 최솟값 앞으로
}
```

+ 순차적으로 최솟값을 골라 정렬하는 방법
+ 성능: $ O(n^2) $
