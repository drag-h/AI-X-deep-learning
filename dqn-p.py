import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import time
from collections import deque

# ----- 환경 설정 -----
GRID_SIZE = 5
CELL_SIZE = 100
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

PICKUP_LOC = (0, 0)
DROP_LOC = (4, 4)
RETURN_LOC = (2, 2)

actions = ['up', 'down', 'left', 'right']
action_dict = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

# ----- 환경 클래스 -----
class WarehouseEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.agent_pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        self.has_item = False
        self.delivered = False
        self.done = False
        return self.get_state()

    def get_state(self):
        return (*self.agent_pos, int(self.has_item))

    def step(self, action):  # ← 반드시 같은 수준 들여쓰기
        dx, dy = action_dict[action]
        x, y = self.agent_pos
        nx, ny = x + dx, y + dy

        valid_move = 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE
        if valid_move:
            self.agent_pos = (nx, ny)
            reward = -1
        else:
            reward = -5

        if not self.has_item:
            goal = PICKUP_LOC
        elif self.has_item and not self.delivered:
            goal = DROP_LOC
        else:
            goal = RETURN_LOC

        distance = abs(self.agent_pos[0] - goal[0]) + abs(self.agent_pos[1] - goal[1])
        shaping_reward = -0.1 * distance
        reward += shaping_reward if valid_move else 0

        if self.agent_pos == PICKUP_LOC and not self.has_item:
            self.has_item = True
            reward = 10
        elif self.agent_pos == DROP_LOC and self.has_item:
            self.has_item = False
            self.delivered = True
            reward = 20
        elif self.agent_pos == RETURN_LOC and self.delivered:
            self.done = True
            reward = 5

        return self.get_state(), reward, self.done


# ----- DQN 구성 -----
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

state_size = 3
action_size = 4

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
batch_size = 32
buffer_size = 10000
target_update_freq = 10
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = QNetwork(state_size, action_size).to(device)
target_net = QNetwork(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
memory = deque(maxlen=buffer_size)

env = WarehouseEnv()

# ----- 학습 -----
for episode in range(500):
    state = np.array(env.reset(), dtype=np.float32)
    for t in range(50):
        if random.random() < epsilon:
            action_idx = random.randint(0, action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action_idx = torch.argmax(q_values).item()

        next_state, reward, done = env.step(actions[action_idx])
        next_state = np.array(next_state, dtype=np.float32)
        memory.append((state, action_idx, reward, next_state, done))
        state = next_state

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions_, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states).to(device)
            actions_ = torch.LongTensor(actions_).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.BoolTensor(dones).unsqueeze(1).to(device)

            q_values = policy_net(states).gather(1, actions_)
            with torch.no_grad():
                next_q = target_net(next_states).max(1, keepdim=True)[0]
                target_q = rewards + gamma * next_q * (~dones)

            loss = nn.MSELoss()(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("\n✅ DQN 학습 완료")
# --- 정책 저장 ---
torch.save(policy_net.state_dict(), "dqn_policy.pth")
print("💾 정책이 dqn_policy.pth 파일로 저장되었습니다.")

# ----- Pygame GUI -----
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Warehouse Robot DQN")
clock = pygame.time.Clock()


def draw_grid(agent_pos, has_item):
    screen.fill((255, 255, 255))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(y*CELL_SIZE, x*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)
            if (x, y) == PICKUP_LOC:
                pygame.draw.rect(screen, (255, 255, 0), rect)  # 노랑
            elif (x, y) == DROP_LOC:
                pygame.draw.rect(screen, (255, 165, 0), rect)  # 주황
            elif (x, y) == RETURN_LOC:
                pygame.draw.rect(screen, (0, 255, 0), rect)    # 초록

    ax, ay = agent_pos
    agent_rect = pygame.Rect(ay*CELL_SIZE+20, ax*CELL_SIZE+20, 60, 60)
    pygame.draw.ellipse(screen, (255, 0, 0), agent_rect)  # 빨강 에이전트
    if has_item:
        pygame.draw.circle(screen, (0, 0, 255), agent_rect.center, 10)  # 파랑 박스
    pygame.display.flip()

# ----- 실행 시각화 -----
state = np.array(env.reset(), dtype=np.float32)
running = True
step_count = 0

while running and step_count < 50:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = policy_net(state_tensor)
        action_idx = torch.argmax(q_values).item()

    next_state, reward, done = env.step(actions[action_idx])
    draw_grid(env.agent_pos, env.has_item)
    print(f"Step {step_count} 위치: {env.agent_pos}, 액션: {actions[action_idx]}, 보상: {reward}")
    state = np.array(next_state, dtype=np.float32)
    step_count += 1

    if done:
        time.sleep(1)
        running = False

    clock.tick(2)

pygame.quit()