---
title: 도커 원격 컨텍스트 연결
date: 2025-03-04 21:45:00 +0900
categories: [Problem Solved, Docker]
tags: [Problem Solved, Docker]
pin: false
math: true
mermaid: true
---


# 문제
호스트 pc에서 원격 pc로 도커 컨테이너를 열어 사용하고싶다.  
조건: 코드는 호스트 pc에 그대로 있으며, docker-compose의 volume 기능을 통해 호스트 데이터를 원격과 동기화를 하며 원격 pc의 자원으로 코딩하는 것

# 해결 과정

## Docker Context에 원격 PC의 도커 Context 추가
도커 Context를 현재 호스트 PC의 Docker Context에 연결하는 방법이 있다.  
Docker Context와 직접적으로 연결할 수 있게 포트를 개방하는 방법도 있지만, 이는 보안상 좋지 않으므로 공식 문서에서 언급되는 Docker over SSH를 사용한다.  

```powershell

docker context create remote-server --docker "host=ssh:server-name"

```

> server-name은 ~/.ssh/config에 정의해놓은 ssh 주소이다. 비밀번호는 공개키를 이용한다.
{: .prompt=info}

이러면 끝이다. context 변경을 통해 연결하면 된다.  
그러나, 오류가 발생한다.  
오류를 수정해야하지만, 생각해보니 volume은 docker 데몬이 실행하는 pc의 volume을 설정하는 것이다.   
즉, 이건 쓸모없는 짓이란 것을 깨달았다...  

## VS Code ssh 연결
vs code의 ssh 연결을 사용하자.. 그게 가장 편하다..

끝.
