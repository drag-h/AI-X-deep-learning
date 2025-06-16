categories: datasets
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

