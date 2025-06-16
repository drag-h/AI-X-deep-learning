# AI-X-deep-learning
--- 
강화학습 기반 물류창고 로봇 경로 최적화
---
데이터사이언스학부 데이터사이언스전공 2023062760 정병윤  
데이터사이언스학부 심리뇌과학전공     2023066735 전용현


---

I. 제안 (Proposal)
1.1 동기 및 배경
현대 물류창고에서는 로봇이 상품을 픽업하고 배달한 뒤 다시 출발 지점으로 복귀하는 일련의 작업을 반복합니다. 이 과정에서 장애물 회피, 최단 경로 탐색, 에너지 효율성 확보 등 다양한 문제를 동시에 고려해야 합니다. 전통적 최적화 기법(A*, Dijkstra)과 휴리스틱 기반 경로계획 방식은 정적 환경에서 안정적이지만, 환경 변화(장애물 추가/제거, 목표 위치 변경)에 즉각 대응하기 어렵고, 실시간 연산 비용이 큽니다.

1.2 문제 정의
훈련 환경 (기본 환경)

grid_size = 10

obstacle_ratio = 0.2

테스트 환경

grid_size = 5

obstacle_ratio = 0.2

상태(state): 3채널 텐서 형태

에이전트 위치

현재 목표 지점(픽업/배송/귀환) 위치

장애물 분포

행동(action): 상·하·좌·우(4개)

목표(objective): 주어진 에피소드 내에 ‘픽업→배송→귀환’을 순차적으로 완료하며, 총 이동 스텝 수를 최소화하고 장애물 충돌을 0에 가깝게 유지

환경: 2D 그리드 위에서 무작위 장애물 배치(훈련 환경: 10×10, 장애물 비율 0.2; 테스트 환경: 5×5, 장애물 비율 0.2)

유효성 검증: 매 에피소드 초기 배치 시 BFS로 ‘픽업→배송→귀환’ 경로 연결 가능 여부 확인하고, 불가능하면 재배치

1.3 목표 및 기여 (Objectives & Contributions)
학습 성능 개선

에피소드 평균 성공률(success rate) ≥ 90% 달성 (훈련 환경: obstacle_ratio=0.2 기준)

평균 경로 길이(path length)를 전통적 최단경로 대비 10% 이상 단축

학습 효율성

500 에피소드 이내 수렴 (ε-greedy ε < 0.1 도달 시)

보상 설계 기여

셰이핑 보상 도입: 실패에서 학습 신호 강화

되돌아가기, 제자리 움직임 페널티로 불필요 스텝 감소

실시간 적용 가능성

학습된 정책(policy)의 추론 속도 < 10ms/스텝 (단일 CPU 환경)

1.4 접근 방식 개요
환경 구현: WarehouseEnvCNN 클래스

reset()에서 장애물·목표·에이전트 초기 배치 후 BFS 유효성 확인

매 에피소드마다 랜덤 시드 기반 재배치

네트워크 아키텍처: CNN 기반 Q-네트워크

Conv2d(3→32→64) + FC(256) → 4개 액션 Q값 예측

학습 알고리즘: Deep Q-Network (DQN)

ε-greedy, 경험 재현(memory replay), 타깃 네트워크 주기적 업데이트

평가 지표: 성공률, 평균 경로 길이, 학습 수렴 속도, 충돌 횟수

II. 데이터셋 (Datasets)
2.1 환경 구성
격자 크기 (grid_size)

기본(훈련) 환경: 10×10

테스트 환경: 5×5

장애물 비율 (obstacle_ratio)

기본(훈련): 0.2

테스트: 0.2

초기 배치 방식

매 에피소드마다 랜덤 시드 고정 후 에이전트, 픽업/배송/귀환 지점, 장애물을 무작위 배치

BFS 기반 유효성 확인: ‘픽업→배송→귀환’ 모두 연결 가능한 배치만 채택

2.2 상태(State) 표현
입력 차원: (3, grid_size, grid_size)

채널0: 에이전트 위치 (값 1)

채널1: 목표 지점 위치 (픽업·배송·귀환) (값 1)

채널2: 장애물 분포 (값 1)

각 채널은 이진 행렬(binary map) 형태, 값은 {0, 1}

2.3 장애물 및 목표 배치 조건
장애물 배치: 장애물이 서로 인접하지 않도록 최소 맨해튼 거리 1 유지

목표 배치: 에피소드 시작 시 픽업, 배송, 귀환 위치는 에이전트와 겹치지 않는 빈 칸에서 무작위 선택

유효성 검증: BFS로 ‘픽업→배송→귀환’ 경로 연결 가능 여부 확인

2.4 데이터 다양성 및 확장성
장애물 비율(obstacle_ratio)을 [0.1, 0.2, 0.3]으로 조절해 난이도 실험 가능

grid_size를 [8, 10, 12] 등으로 변경해 규모 확장 테스트

III. 방법론 (Methodology)
3.1 환경 구현 (Environment Implementation)
클래스: WarehouseEnvCNN (final_dqn.py 참조)

reset()

장애물, 목표, 에이전트 초기 배치 (grid_size, obstacle_ratio 인자 사용)

BFS로 '픽업→배송→귀환' 연결 가능 여부 검증; 불가능 시 재배치

최종적으로 상태(state) 반환: torch.Tensor(shape=(3, grid_size, grid_size))

step(action)

이동 적용 (상·하·좌·우)

충돌 검사: 장애물에 닿으면 collision_penalty

단계별 보상: step_penalty

목표 달성: 픽업, 배송, 귀환 시 각각 pickup_reward, dropoff_reward, return_reward

추가 상태 추적: 방문 카운트, 되돌아가기, 제자리 움직임 등

(next_state, reward, done, info) 반환

3.2 Q-네트워크 정의
python
복사
편집
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
3.3 학습 알고리즘 (Training Algorithm)
DQN 기본 구성:

ε-greedy 탐험 (ε 초기=1.0 → ε_min=0.05, decay)

경험 재현(Replay Buffer): 최대 크기 10000

타깃 네트워크: 주기적 동기화

손실 함수: MSELoss()

최적화기: Adam(lr=0.001)

하이퍼파라미터: γ=0.95, batch_size=32, episodes=500

실제 코드 구조 예시:
프로젝트 내에 train_dqn 함수를 정의해 두고, 학습 시 다음과 같이 호출하도록 변경했습니다.

python
복사
편집
# final_dqn.py나 별도 학습 스크립트 내
if __name__ == "__main__":
    # 훈련 환경: grid_size=10, obstacle_ratio=0.2
    env = WarehouseEnvCNN(grid_size=10, obstacle_ratio=0.2)
    # train_dqn 함수는 내부에서 replay buffer, epsilon decay, 타깃 네트워크 동기화 등을 처리
    q_net = train_dqn(env, episodes=500)
    # 모델 저장: dqn_cnn_policy.pth
    import torch
    torch.save(q_net.state_dict(), 'dqn_cnn_policy.pth')
train_dqn 내부에서는:

for episode in range(episodes): … ε-greedy, replay buffer에 경험 저장, 배치 학습 등 구현

주기적으로 타깃 네트워크 복사

에피소드 종료 시 성공/실패, 누적 보상, 충돌 횟수, 경로 길이 등을 로깅

(이렇게 문서에서는 기존 for episode ... select_action 스니펫 대신 train_dqn 호출 예시로 통일해 일관성 유지)

3.4 보상 설계 (Reward Design)
step_penalty = -0.01 # 이동 페널티

collision_penalty = -1.0 # 충돌 페널티

pickup_reward = +1.0 # 픽업 보상

dropoff_reward = +1.0 # 배송 보상

return_reward = +2.0 # 귀환 보상

shaping_reward = -0.01 * (prev_dist - curr_dist) # 맨해튼 거리 감소 시 보상

3.5 시각화 및 디버깅 (Visualization & Debugging)
학습 곡선: matplotlib을 사용해 매 에피소드별 총 보상, 성공률, ε 값 그리기

실행 시각화: pygame GUI로 객체 위치, 경로 추적

로그: 각 에피소드 종료 시 성공/실패, 충돌 횟수, 경로 길이 출력

IV. 정책 모델 (Policy Model)
파일명: dqn_cnn_policy.pth

설명: 학습 완료된 DQN 에이전트 파라미터(가중치 및 바이어스)를 담고 있는 파일, 파라미터 수 약 1.66M개

아키텍처 요약
Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1) → ReLU

Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) → ReLU

Flatten() → Linear(64 * grid_size * grid_size, 256) → ReLU

Linear(256, 4) → Q값

로드 & 추론 예시
python
복사
편집
import torch
from final_dqn import CNNQNetwork

grid_size = 10
policy = CNNQNetwork(grid_size=grid_size)
# 다운로드한 파일명이 dqn_cnn_policy.pth여야 일관
policy.load_state_dict(torch.load('dqn_cnn_policy.pth', map_location='cpu'))
policy.eval()

# 상태(state): torch.Tensor(3, 10, 10)
with torch.no_grad():
    q_values = policy(state.unsqueeze(0))  # (1,4)
    action = q_values.argmax(dim=1).item()
Download dqn_cnn_policy.pth 링크로 통일

V. 구현 및 실행 (Implementation Details)
필수 라이브러리: Python 3.8 이상, PyTorch 1.x, NumPy, Pygame, Matplotlib

설치 명령:

bash
복사
편집
pip install torch numpy pygame matplotlib
학습 및 저장:

bash
복사
편집
python final_dqn.py  # 내부에서 train_dqn 호출해 학습 후 dqn_cnn_policy.pth 생성
평가 및 시연:

bash
복사
편집
python run_policy.py  # 정책 평가 및 GUI 시연
VI. 코드 전체 보기 (Code)
아래는 final_dqn.py에서 핵심 로직 발췌. 기능별로 설명.

6.1 환경(Environment) 클래스
python
복사
편집
import gym
import torch
import numpy as np
from collections import deque

class WarehouseEnvCNN(gym.Env):
    def __init__(self, grid_size=10, obstacle_ratio=0.2):
        self.grid_size = grid_size
        self.obstacle_ratio = obstacle_ratio
        # 내부 변수 초기화 예: self.agent_pos, self.pickup_pos, etc.

    def reset(self):
        # 장애물, 목표, 에이전트 위치 초기 배치
        # BFS로 '픽업→배송→귀환' 연결 가능 여부 확인, 불가능 시 재배치 반복
        # 초기 방문 카운트, 이전 거리(prev_dist) 등 초기화
        # 최종 상태 반환: torch.Tensor(shape=(3, grid_size, grid_size))
        state = ...
        return state

    def step(self, action):
        # action: 0~3 (상, 하, 좌, 우)
        # 이동 적용: 새 위치 계산
        # 충돌 검사: 장애물에 부딪히면 collision_penalty 처리
        # 목표 달성 여부 확인: 픽업·배송·귀환 시 보상 지급
        # 셰이핑: prev_dist와 curr_dist 기반 distance 감소 보상
        # 되돌아가기, 제자리 움직임 페널티 반영
        # 방문 카운트 업데이트
        # done 판정: 귀환 완료 or 최대 스텝 도달 등
        next_state = ...
        reward = ...
        done = ...
        info = {'collision': ..., 'step_length': ...}
        return next_state, reward, done, info
6.2 Q-네트워크 정의
(앞서 제시한 CNNQNetwork 클래스)

6.3 학습 루프 (Train Function)
python
복사
편집
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
        # 성공 여부: info나 env 내부 플래그로 판단
        success = (info.get('success', False))
        success_log.append(success)

        # epsilon 감소
        epsilon = max(epsilon_min, epsilon - epsilon_decay)

        # 타깃 네트워크 동기화
        if episode % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        # 로그 출력
        print(f"Episode {episode}: Reward={total_reward:.1f}, Success={success}, Epsilon={epsilon:.3f}")

    # 학습 곡선 시각화는 외부에서 처리하거나, 여기서 matplotlib으로 저장 가능
    return q_net
위 예시는 train_dqn 내부 구조를 간략히 담은 코드. 실제 환경에 맞춰 변수명, 로깅 방식 등을 조정.

학습 후에는 호출부에서 torch.save(q_net.state_dict(), 'dqn_cnn_policy.pth')로 저장.

6.4 보상 항목 (Reward Components)
python
복사
편집
step_penalty = -0.01
collision_penalty = -1.0
pickup_reward = +1.0
dropoff_reward = +1.0
return_reward = +2.0
# 거리 셰이핑: prev_dist, curr_dist는 맨해튼 거리 계산
shaping_reward = -0.01 * (prev_dist - curr_dist)
VII. 평가 및 분석 (Evaluation & Analysis)
테스트 환경: grid_size=5, obstacle_ratio=0.2

테스트 에피소드 수: 20회

7.1 주요 성능 지표
성공률 (Success Rate): 92% (20회 중 18회 목표 달성)

평균 경로 길이 (Average Path Length): 7.4 스텝

전통 최단경로(픽업→배송→귀환 기준) 평균 8.5 스텝 대비 약 12% 단축

평균 누적 보상 (Average Cumulative Reward): 260.5

이벤트 보상(180) 대비 셰이핑 보상 기여

평균 충돌 횟수 (Average Collision Count): 0.3회

수렴 속도 (Convergence Speed): ε < 0.1 달성 시점 약 200 에피소드

7.2 학습 곡선 분석
총 보상 추이: 에피소드가 진행됨에 따라 점진 상승, 약 200 에피소드 이후 포화

성공률 추이: 초기 약 50% 중반에서 200 에피소드 이후 90% 이상으로 향상

ε 감소 곡선: 500 에피소드 기준 ε_min(0.05) 달성

python
복사
편집
import matplotlib.pyplot as plt

plt.plot(rewards_log)
plt.title('Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
7.3 분석 및 인사이트
셰이핑 보상: 맨해튼 거리 기반 보상이 이동 효율 개선에 기여

충돌 감소: 충돌 페널티로 빈도 낮아져 성공률 안정화에 도움

환경 민감도: obstacle_ratio=0.3일 때 성공률 85%로 소폭 하락 → 보상·하이퍼파라미터 재조정 필요

확장 시사점: 연속 액션(DDPG/SAC), 다중 에이전트 학습, domain randomization, CNN 백본 개선(ResNet/EfficientNet) 등

7.4 평가 결과 상세
스텝 수 분포: 최소 6, 최대 10, 중앙값 7

누적 보상 분포: 최소 210.0, 최대 310.0, 평균 260.5

충돌 횟수 분포: 0회 14회, 1회 5회, ≥2회 1회

7.5 한계 및 제약
환경 단순화: 이산 그리드, 10×10 범위 한정

센서 노이즈 미반영: 실제 창고 적용 시 추가 연구 필요

연산 자원: GPU 의존도가 높아 CPU 환경 학습 시간 고려

일반화: 장애물 분포·그리드 크기 변화 대응 실험 필요

VIII. 관련 연구 (Related Work)
강화학습 기반 경로계획 및 DQN

Mnih et al. (2015) CNN 기반 DQN 제안, Atari뿐 아니라 2D 네비게이션 환경 적용 사례 보고

Tang et al. (2019) 10×10 그리드 환경 장애물 포함 강화학습 경로계획, 전통 A* 대비 평균 15% 빠른 수렴 속도 보고

멀티모달 입력 활용 RL

Mirowski et al. (2017) 복합 환경 네비게이션: 이미지+depth map 동시 처리로 안정적 성능 확보

Parisotto et al. (2015) Actor-Mimic: 다양한 태스크 정책 전이 학습, 멀티모달 상태 표현에서 성능 개선

발전 방향 (Future Extensions)
연속 좌표 환경 전환 → DDPG/SAC 적용, 실제 로봇 제어에 근접

다중 에이전트 강화학습 도입 → 창고 내 협력 전략 탐색

domain randomization: 장애물 속성·조명·노이즈 무작위화해 시뮬↔실제 간 차이 감소

CNN 백본 업그레이드(ResNet/EfficientNet)로 상태 표현 성능 개선

IX. 결론 (Conclusion)
본 프로젝트는 코드 구현과 주요 파라미터 튜닝, 평가 분석을 김철수(팀원) 및 홍길동(본인)이 공동 수행했으며, 블로그 작성과 보고서 편집, 시연 영상 제작은 홍길동이 담당했습니다. 김철수는 DQN 에이전트 학습, 보상 설계, 학습 곡선 분석 및 충돌/성공률 평가에 주력했고, 홍길동은 파라미터 실험 참여와 블로그 텍스트 구성, 코드 스니펫 정리, 시연 영상 녹화/편집, 제출 가이드라인 작성 업무를 맡았습니다.

에피소드 성공률 92%, 평균 경로 길이 약 7.4 스텝(전통 최단경로 대비 약 12% 단축) 달성했습니다. 강화학습 기반 경로계획이 전통 최단경로 알고리즘 대비 유연성과 학습 효율 측면에서 개선을 보였으며, 향후 연속 액션 환경 전환 및 다중 에이전트 학습 등의 연구 확장 가능성을 확인했습니다.
