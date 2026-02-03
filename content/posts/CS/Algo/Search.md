---
title: Search
subtitle: 검색
draft: false
date: 2025-11-24 15:43:00 +0900
categories: [CS, Algorithm]
tags: [CS, Algorithm, Search]
math: true
mermaid: true
---

## Binary Search: 이진 검색

```text
while (left <= right) {
    middle = (left + right)/2;
    if( key < list[middle] )
        right = middle - 1;
    else if ( key == list[middle] )
        return middle;
    else
        left = middle + 1;
}
```

+ 정렬된 배열에서 원하는 값의 위치를 반환하는 알고리즘
+ 성능: $ O(\log_{2}{N}) $
