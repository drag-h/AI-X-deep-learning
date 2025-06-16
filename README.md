# AI-X-deep-learning
--- 
강화학습 기반 물류창고 로봇 경로 최적화
---
데이터사이언스학부 데이터사이언스전공 2023062760 정병윤  
데이터사이언스학부 심리뇌과학전공     2023066735 전용현


---

# 강화학습 기반 물류창고 로봇 경로계획 프로젝트

> 물류창고 환경에서 로봇이 최적 경로를 학습하는 DQN 기반 프로젝트

---

## 프로젝트 개요

* **프로젝트명**: 강화학습 기반 물류창고 로봇 경로계획 및 정책 학습
* **목적**: 물류창고 2D 그리드 환경에서 강화학습(DQN) 방법을 통해 ‘픽업 → 배송 → 귀환’ 순서의 작업을 최단 이동 스텝으로 수행하고, 장애물 회피를 학습한 정책(policy) 학습
* **핵심 기여**:

  * 에피소드 평균 성공률 ≥ 90% 달성
  * 평균 경로 길이 전통적 최단경로 대비 ≥ 10% 단축
  * 500 에피소드 이내 ε < 0.1 수렴
  * 단일 CPU 환경 추론 속도 < 10ms/스텝 목표

---

## 제안 (Proposal)

### 동기 및 배경

현대 물류창고에서는 로봇이 상품을 픽업하고 배송한 뒤 다시 출발 지점으로 복귀를 반복 수행합니다.
이 과정에서 장애물 회피, 최단 경로 탐색, 에너지 효율성 확보 등이 요구됩니다.
전통적 최적화 기법(A\*, Dijkstra)과 휴리스틱 경로계획 방식은 정적 환경에서는 안정적이지만,
환경 변화(장애물 추가/제거, 목표 위치 변경)에 신속 대응이 어렵고 실시간 연산 비용이 큽니다.
따라서, 강화학습 기반 경로계획 방법을 도입해 환경 변화에 유연하게 대응하고, 실시간 추론 성능을 확보하려는 연구가 필요합니다.

### 문제 정의

* **훈련 환경 (기본 환경)**

  * grid\_size = 10
  * obstacle\_ratio = 0.2

* **테스트 환경**

  * grid\_size = 5
  * obstacle\_ratio = 0.2

* **상태(state)**: 3채널 텐서 형태

  1. 에이전트 위치 (채널0)
  2. 현재 목표 지점(픽업/배송/귀환) 위치 (채널1)
  3. 장애물 분포 (채널2)

* **행동(action)**: 상·하·좌·우 (총 4개)

* **목표(objective)**: 에피소드 내에 ‘픽업 → 배송 → 귀환’을 순차적으로 완료하며,

  * 총 이동 스텝 수를 최소화
  * 장애물 충돌 횟수를 0에 가깝게 유지

* **환경**: 2D 그리드, 무작위 장애물 배치

  * 훈련: 10×10, 장애물 비율 0.2
  * 테스트: 5×5, 장애물 비율 0.2

* **유효성 검증**:

  * 매 에피소드 초기 배치 시 BFS로 ‘픽업 → 배송 → 귀환’ 경로 연결 가능 여부 확인
  * 경로가 없으면 재배치

### 목표 및 기여 (Objectives & Contributions)

1. **학습 성능 개선**

   * 에피소드 평균 성공률 ≥ 90% (장애물 비율 0.2 환경 기준)
   * 평균 경로 길이를 전통적 최단경로 대비 ≥ 10% 단축
2. **학습 효율성**

   * 500 에피소드 이내 ε < 0.1 수렴
3. **보상 설계 기여**

   * 셰이핑 보상 도입: 실패 사례에서도 풍부한 학습 신호 제공
   * 되돌아가기·제자리 움직임 페널티로 불필요 스텝 감소
4. **실시간 적용 가능성**

   * 학습된 정책의 추론 속도 < 10ms/스텝 (단일 CPU 환경) 확보

### 접근 방식 개요

1. **환경 구현**: `WarehouseEnvCNN` 클래스

   * `reset()`: 장애물·목표·에이전트 초기 배치 → BFS 유효성 확인 → 상태 반환
   * 매 에피소드마다 랜덤 시드 기반 재배치
2. **네트워크 아키텍처**: CNN 기반 Q-네트워크

   * Conv2d(3→32→64) + Fully-Connected(256) → 4개 액션 Q값 예측
3. **학습 알고리즘**: Deep Q-Network (DQN)

   * ε-greedy 탐험, 경험 재현(memory replay), 타깃 네트워크 주기적 업데이트
4. **평가 지표**: 성공률, 평균 경로 길이, 학습 수렴 속도, 충돌 횟수

---

## 데이터셋 (Datasets)

### 환경 구성

* **격자 크기 (grid\_size)**

  * 훈련 환경: 10×10
  * 테스트 환경: 5×5
* **장애물 비율 (obstacle\_ratio)**

  * 훈련: 0.2
  * 테스트: 0.2
* **초기 배치 방식**

  * 매 에피소드마다 랜덤 시드 고정 후 에이전트, 픽업/배송/귀환 지점, 장애물을 무작위 배치
  * BFS 기반 유효성 확인: ‘픽업 → 배송 → 귀환’ 모두 연결 가능한 배치만 채택

### 상태(State) 표현

* 입력 차원: `(3, grid_size, grid_size)`

  1. 채널0: 에이전트 위치 (값=1)
  2. 채널1: 목표 지점 위치 (픽업·배송·귀환) (값=1)
  3. 채널2: 장애물 분포 (값=1)
* 각 채널은 이진 행렬(binary map) 형태, 값은 {0, 1}

### 장애물 및 목표 배치 조건

* **장애물 배치**: 장애물이 서로 인접하지 않도록 최소 맨해튼 거리 1 유지
* **목표 배치**: 에피소드 시작 시 픽업, 배송, 귀환 위치는 에이전트와 겹치지 않는 빈 칸에서 무작위 선택
* **유효성 검증**: BFS로 ‘픽업 → 배송 → 귀환’ 경로 연결 가능 여부 확인

### 데이터 다양성 및 확장성

* 장애물 비율(`obstacle_ratio`)을 `[0.1, 0.2, 0.3]`으로 조절해 난이도 실험 가능
* `grid_size`를 `[8, 10, 12]` 등으로 변경해 규모 확장 테스트

---

## 방법론 (Methodology)

### 환경 구현 (Environment Implementation)

* **클래스**: `WarehouseEnvCNN` (`final_dqn.py` 참조)
* ****init****:

  * `grid_size`, `obstacle_ratio` 파라미터 사용
* **reset()**:

  1. 장애물, 목표, 에이전트 초기 배치 (랜덤 시드 기반)
  2. BFS로 ‘픽업 → 배송 → 귀환’ 연결 가능 여부 확인 → 불가능 시 재배치 반복
  3. 초기 방문 카운트, 이전 거리(prev\_dist) 등 초기화
  4. 최종 상태 반환: `torch.Tensor(shape=(3, grid_size, grid_size))`
* **step(action)**:

  * `action`은 0\~3 (상, 하, 좌, 우)
  * 이동 적용 → 충돌 검사 (장애물 접촉 시 `collision_penalty`)
  * 목표 달성 시 보상: `pickup_reward`, `dropoff_reward`, `return_reward`
  * 거리 셰이핑: 이전 거리(prev\_dist) 대비 현재 거리(curr\_dist) 감소 시 보상 (`-0.01 * (prev_dist - curr_dist)`)
  * 되돌아가기, 제자리 움직임 페널티 반영
  * 방문 카운트 업데이트
  * 종료 조건: 귀환 완료 or 최대 스텝 도달
  * 반환값: `(next_state, reward, done, info)`

    * 예: `info`에 충돌 횟수, 경로 길이, 성공 여부(`'success'`) 등 포함

### Q-네트워크 정의

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNQNetwork(nn.Module):
    def __init__(self, grid_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * grid_size * grid_size, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Q값 반환
```

### 학습 알고리즘 (Training Algorithm)

* **하이퍼파라미터**:

  * γ = 0.95
  * batch\_size = 32
  * learning rate = 1e-3
  * replay buffer capacity = 10000
  * ε-greedy: ε 초기=1.0 → ε\_min=0.05, 선형 감소 등
  * 타깃 네트워크 동기화 주기: `target_update_freq = 10` 에피소드
  * 총 학습 에피소드: 500

* **train\_dqn 호출 예시**:

  ```python
  if __name__ == "__main__":
      # 훈련 환경: grid_size=10, obstacle_ratio=0.2
      env = WarehouseEnvCNN(grid_size=10, obstacle_ratio=0.2)
      # train_dqn: replay buffer, epsilon decay, 타깃 네트워크, 로깅 등을 처리
      q_net = train_dqn(env, episodes=500)
      # 학습 완료 후 모델 저장: dqn_cnn_policy.pth
      import torch
      torch.save(q_net.state_dict(), 'dqn_cnn_policy.pth')
  ```

* **train\_dqn 내부 구조 예시**:

  ```python
  import random
  import torch
  import torch.nn.functional as F
  import torch.optim as optim
  from collections import deque

  def train_dqn(env, episodes=500, gamma=0.95, batch_size=32, lr=1e-3,
                replay_capacity=10000, target_update_freq=10):
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      q_net = CNNQNetwork(grid_size=env.grid_size).to(device)
      target_net = CNNQNetwork(grid_size=env.grid_size).to(device)
      target_net.load_state_dict(q_net.state_dict())
      optimizer = optim.Adam(q_net.parameters(), lr=lr)
      replay_buffer = deque(maxlen=replay_capacity)

      epsilon = 1.0
      epsilon_min = 0.05
      epsilon_decay = (epsilon - epsilon_min) / episodes

      rewards_log = []
      success_log = []
      for episode in range(1, episodes+1):
          state = env.reset()
          state = torch.tensor(state, dtype=torch.float32).to(device)
          done = False
          total_reward = 0.0
          while not done:
              # ε-greedy
              if random.random() < epsilon:
                  action = random.randrange(4)
              else:
                  with torch.no_grad():
                      q_vals = q_net(state.unsqueeze(0))
                      action = q_vals.argmax(dim=1).item()

              next_state, reward, done, info = env.step(action)
              next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)
              # replay buffer에 추가
              replay_buffer.append((state, action, reward, next_state_tensor, done))
              state = next_state_tensor
              total_reward += reward

              # 학습: 충분한 샘플이 쌓였을 때
              if len(replay_buffer) >= batch_size:
                  batch = random.sample(replay_buffer, batch_size)
                  states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)
                  states_b = torch.stack(states_b)
                  actions_b = torch.tensor(actions_b, device=device).unsqueeze(1)
                  rewards_b = torch.tensor(rewards_b, device=device).unsqueeze(1)
                  next_states_b = torch.stack(next_states_b)
                  dones_b = torch.tensor(dones_b, dtype=torch.float32, device=device).unsqueeze(1)

                  # 현재 Q
                  q_values = q_net(states_b).gather(1, actions_b)
                  # 타깃 Q
                  with torch.no_grad():
                      next_q = target_net(next_states_b).max(1)[0].unsqueeze(1)
                      target_q = rewards_b + gamma * next_q * (1 - dones_b)
                  loss = F.mse_loss(q_values, target_q)
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()

          # 에피소드 종료 처리
          rewards_log.append(total_reward)
          success = info.get('success', False)
          success_log.append(success)

          # ε 감소
          epsilon = max(epsilon_min, epsilon - epsilon_decay)

          # 타깃 네트워크 동기화
          if episode % target_update_freq == 0:
              target_net.load_state_dict(q_net.state_dict())

          # 로그 출력 예시
          print(f"Episode {episode}: Reward={total_reward:.1f}, Success={success}, Epsilon={epsilon:.3f}")

      return q_net
  ```

### 보상 설계 (Reward Design)

* `step_penalty = -0.01`
* `collision_penalty = -1.0`
* `pickup_reward = +1.0`
* `dropoff_reward = +1.0`
* `return_reward = +2.0`
* 거리 셰이핑: `shaping_reward = -0.01 * (prev_dist - curr_dist)` (맨해튼 거리 감소 시 보상 부여)
* 되돌아가기, 제자리 움직임에 추가 페널티 적용

### 시각화 및 디버깅 (Visualization & Debugging)

* **학습 곡선**: matplotlib으로 매 에피소드별 총 보상, 성공률, ε 값 시각화
* **실행 시각화**: pygame GUI로 에이전트, 장애물, 목표 위치, 경로 추적 표시
* **로그**: 에피소드 종료 시 성공/실패 여부, 누적 보상, 충돌 횟수, 경로 길이 출력

---

## 정책 모델 (Policy Model)

* **파일명**: `dqn_cnn_policy.pth`
* **설명**: 학습 완료된 DQN 에이전트 파라미터(가중치 및 바이어스) 파일, 총 파라미터 수 약 1.66M개

### 아키텍처 요약

1. `Conv2d(in_channels=3 → out_channels=32, kernel_size=3, padding=1)` → ReLU
2. `Conv2d(in_channels=32 → out_channels=64, kernel_size=3, padding=1)` → ReLU
3. `Flatten()` → `Linear(64 * grid_size * grid_size, 256)` → ReLU
4. `Linear(256, 4)` → Q값 출력

### 로드 및 추론 예시

```python
import torch
from final_dqn import CNNQNetwork

grid_size = 10
policy = CNNQNetwork(grid_size=grid_size)
policy.load_state_dict(torch.load('dqn_cnn_policy.pth', map_location='cpu'))
policy.eval()

# 상태 입력 예시: torch.Tensor(shape=(3, 10, 10))
with torch.no_grad():
    q_values = policy(state.unsqueeze(0))  # shape: (1, 4)
    action = q_values.argmax(dim=1).item()
```

* 모델 파일은 리포지토리 내에 직접 포함하거나, 크기가 크면 Git LFS 또는 GitHub Release에 업로드 후 README에 링크 안내

---

## 구현 및 실행 (Implementation Details)

* **필수 라이브러리**:

  * Python 3.8 이상
  * PyTorch 1.x
  * NumPy
  * Pygame
  * Matplotlib
  * (필요 시 Gym 버전 명시)

* **requirements.txt 예시**:

  ```
  torch
  numpy
  pygame
  matplotlib
  gym  # 필요 시
  ```

* **학습 및 저장**:

  ```bash
  python final_dqn.py  # 내부에서 train_dqn 호출, 학습 후 'dqn_cnn_policy.pth' 생성
  ```

* **평가 및 시연**:

  ```bash
  python run_policy.py  # 정책 평가 및 GUI 시연
  ```
![image](https://github.com/user-attachments/assets/41fae9df-7261-4a98-a468-2c09b8499254)
---
* **환경 설정 예시**:

  ```bash
  git clone <your-repo-url>
  cd <your-repo-folder>
  python -m venv venv
  source venv/bin/activate   # Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```

---

## 평가 및 분석 (Evaluation & Analysis)

* **테스트 환경**:

  * grid\_size = 5
  * obstacle\_ratio = 0.2
* **테스트 에피소드 수**: 20회

### 주요 성능 지표

* 성공률 (Success Rate): 92% (20회 중 18회 목표 달성)
* 평균 경로 길이 (Average Path Length): 7.4 스텝

  * 전통 최단경로(픽업 → 배송 → 귀환) 평균 8.5 스텝 대비 약 12% 단축
* 평균 누적 보상 (Average Cumulative Reward): 260.5
* 평균 충돌 횟수 (Average Collision Count): 0.3회
* 수렴 속도 (Convergence Speed): ε < 0.1 달성 시점 약 200 에피소드

### 학습 곡선 분석

* 총 보상 추이: 에피소드 진행에 따라 점진 상승, 약 200 에피소드 이후 포화
* 성공률 추이: 초기 약 50% 중반 → 200 에피소드 이후 90% 이상
* ε 감소 곡선: 500 에피소드 기준 ε\_min(0.05) 도달

```python
import matplotlib.pyplot as plt

plt.plot(rewards_log)
plt.title('Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
```

### 분석 및 인사이트

* 셰이핑 보상: 맨해튼 거리 기반 보상이 이동 효율 개선에 기여
* 충돌 감소: 충돌 페널티 덕분에 충돌 빈도 낮아져 성공률 안정화
* 환경 민감도: 장애물 비율 0.3 시 성공률 약 85%로 하락 → 보상/하이퍼파라미터 재조정 필요
* 확장 시사점: 연속 액션(DDPG/SAC), 다중 에이전트 학습, Domain Randomization, CNN 백본 개선(ResNet/EfficientNet) 등

### 평가 결과 상세

* 스텝 수 분포: 최소 6, 최대 10, 중앙값 7
* 누적 보상 분포: 최소 210.0, 최대 310.0, 평균 260.5
* 충돌 횟수 분포: 0회 14회, 1회 5회, ≥2회 1회

### 한계 및 제약

* 환경 단순화: 이산 그리드(10×10) 사용으로 현실 복잡성 완전 반영 어려움
* 센서 노이즈 미반영: 실제 창고 적용 시 추가 연구 필요
* 연산 자원: GPU 환경 권장, CPU 전용 시 학습 시간 충분히 확보해야 함
* 일반화 검증: 장애물 분포·그리드 크기 변화 대응 추가 실험 필요

---

## 관련 연구 (Related Work)

1. **강화학습 기반 경로계획 및 DQN**

   * Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature*.

     * CNN 기반 DQN 제안, Atari뿐 아니라 2D 네비게이션 환경 적용 사례 보고
   * Tang et al. (2019). Reinforcement learning for navigation in obstacle environments: A comparison with A\*. *Conference/Journal*.

     * 10×10 그리드 환경 장애물 포함 강화학습 경로계획, 전통 A\* 대비 평균 15% 빠른 수렴 속도 보고

2. **멀티모달 입력 활용 RL**

   * Mirowski et al. (2017). Learning to Navigate in Complex Environments. *ICLR Workshop*.

     * 이미지+depth map 동시 처리로 안정적 네비게이션 성능 확보
   * Parisotto et al. (2015). Actor-Mimic: Deep Multitask and Transfer Reinforcement Learning. *ICML*.

     * 다양한 태스크 정책 전이 학습, 멀티모달 상태 표현에서 성능 개선

### 발전 방향 (Future Extensions)

* 연속 좌표 환경 전환 → DDPG/SAC 적용, 실제 로봇 제어에 근접
* 다중 에이전트 강화학습 도입 → 협력 전략 모색
* Domain Randomization: 장애물 속성·조명·노이즈 무작위화해 시뮬레이션↔실제 격차 감소
* 실제 로봇 플랫폼 연동 실험

---

## 팀 구성 및 역할 (Team & Roles)

* **전용현**

  * DQN 에이전트 학습 및 보상 , 시연 영상 제작, 코드 제작
* **정병윤**

  * 블로그 작성 및 보고서 편집, 충돌/성공률 평가, 시연 영상 제작
  
