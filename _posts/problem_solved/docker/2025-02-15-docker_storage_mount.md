---
title: 도커 로컬디스크 마운트
date: 2025-02-15 21:00:00 +0900
categories: [Problem Solved, Docker]
tags: [Problem Solved, Docker]
pin: false
math: true
mermaid: true
---


# 문제
windows 11의 wsl2 환경이다. Docker Desktop을 사용하며, Devcontainer 환경이다. docker-compose에서 외부 디스크의 접근 설정을 해두었으나, 동작하지 않는다.

```yml
# 당시 코드

name: python-tensorflow-pytorch
services:
  app:
    build:
      context: ..
      dockerfile: .devcontainer/Dockerfile
      target: runtime
    volumes:
      - ../:/workspaces/dev
      - /mnt/f/Datasets:/Datasets:rw
    environment:
      DISPLAY: ${DISPLAY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: ["tail", "-f", "/dev/null"]  

```

# 해결 과정

## mount --bind

wsl2에서 /mnt/f/Datasets은 파일 시스템을 그대로 보여주는 것이지, 리눅스 네이티브 경로는 아니다. 따라서 bind 명령어로 docker가 인식할 수 있게 해줘야 한다.

```bash
# wsl2에서 디스크 마운트

sudo mkdir -p /mnt/datasets
sudo mount --bind /mnt/f/Datasets /mnt/Datasets 

```

```yml
# docker-compose 내의 volumes 수정

/mnt/Datasets:/Datasets:rw

```

**실패**


## rsync

도커가 wsl2 내의 리눅스 파일에 직접 접근하기에 동기화를 해주어 시도할 수 있다.

```bash
# windows 파일 시스템과 wsl2 경로와 동기화

sudo rsync -av /mnt/f/Datasets ~/Datasets

```

```yml
# docker-compose 내의 volumes 수정

~/Datasets:/Datasets:rw

```

**실패**


## windows 경로 직접 입력

docker-compose 에서 windows 경로를 직접 입력한다.

```yml
# docker-compose의 volumes 수정

F:/Datasets:/Datasets:rw

```

**성공**
그러나 성능이 좋지 않다.


## cp

wsl2 내의 리눅스 파일로 원래 경로에 있는 데이터를 완전히 옮겨서 엑세스한다.

```bash
# cp

cp -r /mnt/f/Datasets ~/Datasets

```

```yml
# docker-compose의 volumes 수정

~Datasets:/Datasets:rw

```

**실패**


# 결론
일단 되는걸로 쓰자.. 성능이 감소되긴 하지니니

-> 근데 다시 찾아보니, Windows에서 실행된 Dev Container의 경로는 Windows 경로로 인식하기 떄문에, 실패된 방식들을 사용하려면 WSL2에서 실행해야 한다.
