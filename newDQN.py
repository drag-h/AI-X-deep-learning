import numpy as np
import random
import matplotlib.pyplot as plt
import pygame
import time
from collections import defaultdict

class WarehouseEnvCNN:
    def __init__(self, grid_size=10, obstacle_ratio=0.1):
        self.grid_size = grid_size
        self.obstacle_ratio = obstacle_ratio
        self.actions = ['up', 'down', 'left', 'right']
        self.action_dict = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        self.reset()

    from collections import deque

    def reset(self):
        while True:
            self.obstacles = set()
            self.visit_count = defaultdict(int)
            self.grid = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
            self.agent_pos = self._random_empty_cell()
            self.pickup_pos = self._random_empty_cell()
            self.drop_pos = self._random_empty_cell()
            self.return_pos = self._random_empty_cell()
            self.has_item = False
            self.delivered = False
            self.done = False
            self.prev_pos = None
            self.prev_prev_pos = None

            total_cells = self.grid_size ** 2
            num_obstacles = int(total_cells * self.obstacle_ratio)
            while len(self.obstacles) < num_obstacles:
                ox, oy = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
                if (ox, oy) not in [self.agent_pos, self.pickup_pos, self.drop_pos, self.return_pos]:
                    self.obstacles.add((ox, oy))
            for (ox, oy) in self.obstacles:
                self.grid[2, ox, oy] = 1.0

        # 🧠 유효성 검사: agent → pickup → drop → return 가능해야 함
            if self._is_reachable(self.agent_pos, self.pickup_pos) and \
            self._is_reachable(self.pickup_pos, self.drop_pos) and \
            self._is_reachable(self.drop_pos, self.return_pos):
                break  # 도달 가능하면 맵 확정

        return self._get_state()
    def _is_reachable(self, start, goal):
        visited = set()
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True
            for dx, dy in self.action_dict.values():
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if (nx, ny) not in self.obstacles and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return False

    def _random_empty_cell(self):
        while True:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos not in self.obstacles:
                return pos

    def _get_state(self):
        self.grid[0].fill(0)  # agent
        self.grid[1].fill(0)  # goal
        ax, ay = self.agent_pos
        self.grid[0, ax, ay] = 1.0
        if not self.has_item:
            gx, gy = self.pickup_pos
        elif not self.delivered:
            gx, gy = self.drop_pos
        else:
            gx, gy = self.return_pos
        self.grid[1, gx, gy] = 1.0
        return np.copy(self.grid)

    def step(self, action):
        dx, dy = self.action_dict[action]
        x, y = self.agent_pos
        nx, ny = x + dx, y + dy
        reward = -1  # 기본 이동 페널티

        # ----- 이동 처리 -----
        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
            if (nx, ny) not in self.obstacles:
                self.agent_pos = (nx, ny)
            else:
                reward = -10  # 장애물 충돌
        else:
            reward = -10 # 벽 충돌

        # ----- 중복 방문 패널티 -----
        self.visit_count[self.agent_pos] += 1
        if self.visit_count[self.agent_pos] > 1:
            reward -= 1 * (self.visit_count[self.agent_pos] - 1)  # 누적 감점

        # ----- 목표 지점까지의 shaping reward -----
        gx, gy = (
            self.pickup_pos if not self.has_item
            else self.drop_pos if not self.delivered
            else self.return_pos
        )
        ax, ay = self.agent_pos
        curr_dist = abs(ax - gx) + abs(ay - gy)

        # 🔸 min_dist가 없다면 초기화
        if not hasattr(self, 'min_dist') or self.min_dist is None:
            self.min_dist = curr_dist

        # ✅ 1. 새로 더 가까워졌을 때만 보상 (중복 없이)
        if curr_dist < self.min_dist:
            delta = self.min_dist - curr_dist
            reward += 2 ** delta         # 🎁 지수적 보상
            self.min_dist = curr_dist    # 최소 거리 갱신

        # ✅ 2. 이전 위치보다 가까워졌을 때만 소량 shaping 보상
        if self.prev_pos:
            prev_dist = abs(self.prev_pos[0] - gx) + abs(self.prev_pos[1] - gy)
            if curr_dist < prev_dist:
                reward += 2.0  # 소량 shaping reward

        # ✅ 3. 목표 지점 도달 시 거리 기록 초기화
        if self.agent_pos == (gx, gy):
            self.min_dist = None       

        # ----- 이벤트 도달 보상 -----
        if self.agent_pos == self.pickup_pos and not self.has_item:
            self.has_item = True
            reward += 100
        elif self.agent_pos == self.drop_pos and self.has_item and not self.delivered:
            self.delivered = True
            self.has_item = False
            reward += 60
        elif self.agent_pos == self.return_pos and self.delivered:
            self.done = True
            reward += 20
        if self.visit_count[self.agent_pos] > 1:
            reward -= 5
        # ----- 제자리 및 왔다 갔다 패널티 -----
        if self.prev_pos == self.agent_pos:
            reward -= 20  # 제자리 행동 감점
        if hasattr(self, 'prev_prev_pos') and self.prev_prev_pos == self.agent_pos:
            reward -= 15  # 두 칸 전 위치로 되돌아온 경우 감점

        # ----- 위치 기록 -----
        self.prev_prev_pos = self.prev_pos
        self.prev_pos = self.agent_pos

        # reward 하한선 적용
        reward = max(reward, -50)  # 예: reward는 최소 -50까지만

        return self._get_state(), reward, self.done
    
import torch
import torch.nn as nn

class CNNQNetwork(nn.Module):
    def __init__(self, input_channels=3, action_size=4, grid_size=3):
        super(CNNQNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * grid_size * grid_size, 256),  # 동적으로 처리
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

import torch.optim as optim
from collections import deque
import random

def train_dqn(env, episodes=1000, batch_size=32, gamma=0.95, epsilon=1.0,
              epsilon_min=0.1, epsilon_decay=0.997, lr=0.001, buffer_size=10000,max_steps=50):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = CNNQNetwork(grid_size=env.grid_size).to(device)
    target_net = CNNQNetwork(grid_size=env.grid_size).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    success_list = []

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    memory = deque(maxlen=buffer_size)
    action_list = ['up', 'down', 'left', 'right']

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        for _ in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # (1, 3, 10, 10)

            if random.random() < epsilon:
                action_idx = random.randint(0, 3)
            else:
                with torch.no_grad():
                    q_vals = q_net(state_tensor)  # shape: (1, 4)
                    q_vals += torch.rand_like(q_vals) * 1e-2  # tie-breaker (optional)
                    action_idx = torch.argmax(q_vals[0]).item()  # ✅ 안전하고 확실함


            next_state, reward, done = env.step(action_list[action_idx])
            memory.append((state, action_idx, reward, next_state, done))
            total_reward += reward
            state = next_state

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                dones = torch.BoolTensor(dones).unsqueeze(1).to(device)

                q_vals = q_net(states).gather(1, actions)
                with torch.no_grad():
                    next_q = target_net(next_states).max(1, keepdim=True)[0]
                    target = rewards + gamma * next_q * (~dones)

                loss = nn.MSELoss()(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        if ep % 10 == 0:
            target_net.load_state_dict(q_net.state_dict())
            # ----- 에피소드 종료 후 -----
            if env.has_item:
                success_list.append(1)
                success = True
            else:
                success_list.append(0)
                success = False

            cumulative_success_rate = sum(success_list) / len(success_list)

            # ----- 로그 출력 -----
            print(f"📘 Episode {ep}, Total Reward: {total_reward:.2f}, "
                f"Epsilon: {epsilon:.3f}, "
                f"✅ Success: {success}, "
                f"🎯 Success Rate: {cumulative_success_rate:.3f}")

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    print("✅ 학습 완료")
    torch.save(q_net.state_dict(), "/Users/jeon-yonghyeon/Desktop/hanyang/3-1/aix/과제/enw/dqn_model.pth")
    print("💾 정책 저장됨: dqn_cnn_policy.pth")
    return q_net

def test_policy(env, q_net, episodes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net.eval()
    action_list = ['up', 'down', 'left', 'right']

    for ep in range(episodes):
        state = env.reset()
        done = False
        step = 0
        print(f"\n🧪 테스트 에피소드 {ep + 1}")
        while not done and step < 100:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals = q_net(state_tensor)
                action_idx = torch.argmax(q_vals).item()
            state, reward, done = env.step(action_list[action_idx])
            print(f"Step {step:2d}, Action: {action_list[action_idx]}, Reward: {reward}")
            step += 1



CELL_SIZE = 100
WINDOW_SIZE = 500  # 5x5 맵이면 500x500, 10x10 맵이면 1000x1000 등으로 조절

def draw_env(state, grid_size):
    pygame.init()
    screen = pygame.display.set_mode((CELL_SIZE * grid_size, CELL_SIZE * grid_size))
    pygame.display.set_caption("DQN Agent View")
    screen.fill((255, 255, 255))

    for i in range(grid_size):
        for j in range(grid_size):
            x = j * CELL_SIZE
            y = i * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)

            agent = state[0, i, j]
            goal = state[1, i, j]
            obstacle = state[2, i, j]

            if obstacle == 1:
                pygame.draw.rect(screen, (50, 50, 50), rect)
            elif goal == 1:
                pygame.draw.rect(screen, (255, 0, 0), rect)
            elif agent == 1:
                pygame.draw.rect(screen, (0, 0, 255), rect)

    pygame.display.flip()

def run_with_gui(env, q_net):
    pygame.init() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net.eval()
    state = env.reset()
    done = False
    step = 0
    clock = pygame.time.Clock()

    while not done and step < 100:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        draw_env(state, env.grid_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals = q_net(state_tensor)
            action_idx = torch.argmax(q_vals).item()

        state, reward, done = env.step(env.actions[action_idx])
        print(f"Step {step}, Action: {env.actions[action_idx]}, Reward: {reward}")
        time.sleep(0.5)
        step += 1

    time.sleep(2)
    pygame.quit()


import os
if __name__ == "__main__":
    env = WarehouseEnvCNN(grid_size=5, obstacle_ratio=0.1)
    print("🚀 학습 시작")
    model = train_dqn(env, episodes=1000, epsilon_min=0.05)
    
    print("🎯 테스트 시작")
    run_with_gui(env, model)  # 테스트를 pygame GUI로 진행 