---
title: ubuntu에 tensorflow 설치!
date: 2025-03-04 21:45:00 +0900
categories: [Problem Solved, Docker]
tags: [Problem Solved, Docker]
pin: false
math: true
mermaid: true
---


# 문제
도커 컨테이너에 tensorflow을 설치하였으나 동작하지 않는다.  
GPU는 잘 잡으나 DNN Library가 없다는 오류가 뜬다.  
or 컴파일된 cudnn과 설치된 cudnn이 다르다는 오류가 뜬다.

# 해결 과정

## Dockerfile 점검
수십번한것같다.. 도커파일을 수십번 수정하였다.  
Cuda, cudnn 전부 nvidia에서 알려주는 공식 설치 과정을 따랐음에도, 고쳐지지 않았다.
이유는 정말 단순했다..  
인터넷에 널리 알려진 cuda 환경 변수와 nvidia에서 dpkg를 통해서 설치해주는 경로가 다르다!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
아.. 처음부터 순정 ubuntu에서 설치과정을 시도했다면 훨씬 빠르게 해결했을텐데..  
오늘도 후회한다. 도커 파일 만들때는 순정상태에서 해보고 필요없는 라이브러리를 지우는 순으로 가자..  
```bash
cp /usr/include/cudnn*.h /usr/local/cuda/include
cp /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64
```

이렇게하면 해결된다.. (cudnn 9,3, cuda 12.5, ubuntu 22.04 기준)

끝
