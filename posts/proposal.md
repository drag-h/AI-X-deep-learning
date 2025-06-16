layout: post
title:  "I. Proposal"
date:   2025-06-16 09:00:00 +0900
categories: proposal

Motivation

현재 대부분의 물류창고 로봇은 사전에 지정된 경로를 따르도록 프로그래밍됩니다.창고 구조나 작업 패턴이 변경될 때마다 경로를 일일이 재설계해야 운영 효율이 떨어집니다.Q-learning을 적용하면 테이블 형태로 상태-행동 가치를 학습하지만, 상태 공간이 커지면 수렴 속도가 느리고 일반화가 어려운 한계가 있습니다.

Objectives

Q-learning의 한계 분석

작은 Grid 환경에서 학습 곡선과 보상 수렴 특성 평가

상태 공간 확장 시 학습 지연 및 성능 저하 문제 도출

DQN (Deep Q-Network) 도입 및 성능 향상

신경망 기반 Q-function 근사로 일반화 능력 확보

Experience Replay와 Target Network 적용

하이퍼파라미터 튜닝으로 학습 안정성 개선

멀티모달 환경 및 장애물(벽) 추가

다양한 목표 위치와 멀티태스크 수행 시나리오 확장

장애물(벽) 추가로 경로 탐색 난이도 증가

코드 확장성을 고려한 환경 구성

비교 평가 및 시각화

5×5, 10×10 Grid 환경에서 Q-learning vs. DQN 성능 비교

이동 궤적, 학습 곡선(loss, reward) 그래프 시각화

멀티모달 및 장애물 환경에서 정책 일반화 능력 분석
