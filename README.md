# AI-X-deep-learning
---
"AI+X Deep Learning 프로젝트"
---
데이터사이언스학부 데이터사이언스전공 2023062760 정병윤
데이터사이언스학부 심리뇌과학전공     2023066735 전용현

---
강화학습 기반 물류창고 로봇 경로 최적화
---
categories: proposal
--------------------

## Motivation

현재 대부분의 물류창고 로봇은 사전에 지정된 경로를 따르도록 프로그래밍됩니다.
창고 구조나 작업 패턴이 변경될 때마다 경로를 일일이 재설계해야 운영 효율이 떨어집니다.
**Q-learning**을 적용하면 테이블 형태로 상태-행동 가치를 학습하지만, 상태 공간이 커지면 수렴 속도가 느리고 일반화가 어려운 한계가 있습니다.

## Objectives

1. **Q-learning**의 한계 분석

   * 작은 Grid 환경에서 학습 곡선과 보상 수렴 특성 평가
   * 상태 공간 확장 시 학습 지연 및 성능 저하 문제 도출
2. **DQN (Deep Q-Network)** 도입 및 성능 향상

   * 신경망 기반 Q-function 근사로 일반화 능력 확보
   * Experience Replay와 Target Network 적용
   * 하이퍼파라미터 튜닝으로 학습 안정성 개선
3. **멀티모달 환경 및 장애물(벽) 추가**

   * 다양한 목표 위치와 멀티태스크 수행 시나리오 확장
   * 장애물(벽) 추가로 경로 탐색 난이도 증가
   * 코드 확장성을 고려한 환경 구성
4. **비교 평가 및 시각화**

   * 5×5, 10×10 Grid 환경에서 Q-learning vs. DQN 성능 비교
   * 이동 궤적, 학습 곡선(loss, reward) 그래프 시각화
   * 멀티모달 및 장애물 환경에서 정책 일반화 능력 분석
  
   * categories: datasets
--------------------

## Environment Setup

* Python 시뮬레이션 환경 (NumPy, PyTorch, Pygame)
* **Grid Sizes**: 5×5 (훈련용, Q-learning/DQN/PPO), 10×10 (테스트용, DQN/PPO)

## State Representation

* 에이전트 위치: (x, y) 정수 좌표
* 아이템 보유 여부: 0 or 1 (불리언 → 정수)
* 상태 벡터: `[x/GRID_SIZE, y/GRID_SIZE, has_item]` 형태로 정규화 (RL 코드는 좌표를 0–1 사이로 스케일)

## Data Collection & Training Parameters

| 알고리즘       | Grid | Episodes | Max Steps | 주요 파라미터                                                       |
| ---------- | ---- | -------- | --------- | ------------------------------------------------------------- |
| Q-learning | 5×5  | 1,000    | 50        | α=0.1, γ=0.9, ε=0.1                                           |
| DQN        | 5×5  | 500      | 50        | γ=0.99, ε₀=1.0→εₘᵢₙ=0.1, decay=0.995, batch=32, buffer=10,000 |
| PPO        | 5×5  | 1,000    | 100       | γ=0.99, λ=0.95, lr=1e-3                                       |

## Testing Scenarios

* **DQN/PPO(테스트)**: 10×10 Grid에서 최종 모델 로드 후 최대 100 스텝 수행
* **Q-learning(테스트)**: 5×5 Grid에서 학습된 Q-table 기반으로 최대 3,000 스텝 시각화

## Reward & Logging

* 기본 보상: 이동 –1, 벽 충돌 –5 (DQN/PPO)
* 목표 도달 보상: 픽업 +10, 드롭 +20, 리턴 +5
* 보조(Shaping) 보상: 맨해튼 거리 기반 –0.1×거리
* 저장 항목: episode, step, 누적 보상, loss(Q-network/PPO critic), Q-values

## Preprocessing

* 상태 정규화: 좌표값을 GRID\_SIZE로 나눠 0\~1 범위로 스케일링
* 장애물 맵: sparse 형태(벽 위치 리스트) → step 함수에서 invalid move로 처리

## Synthetic Scenario Generation

* **멀티모달 조합**: 픽업1→드롭1→픽업2→… 형태의 순차적 목표
* **장애물 배치**: 고정 벽 패턴 및 동적 장애물(추가 구현 예정)

<!-- TODO: 실제 장애물 맵 예시 이미지, 목표 위치 조합 표 삽입 -->

