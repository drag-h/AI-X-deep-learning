import numpy as np
import torch
import torch.nn as nn
import pygame
import time
import random

# ----- 환경 설정 -----
GRID_SIZE = 10
CELL_SIZE = 60
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

PICKUP_LOC = (0, 0)
DROP_LOC = (9, 9)
RETURN_LOC = (5, 5)

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

    def step(self, action):
        dx, dy = action_dict[action]
        x, y = self.agent_pos
        nx, ny = x + dx, y + dy

        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            self.agent_pos = (nx, ny)

        reward = -1
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

# ----- DQN 네트워크 정의 (저장된 것과 동일해야 함) -----
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

# ----- 모델 로드 -----
state_size = 3
action_size = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = QNetwork(state_size, action_size).to(device)
policy_net.load_state_dict(torch.load("dqn_policy.pth", map_location=device))
policy_net.eval()

# ----- 시각화 -----
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Warehouse Robot DQN (10x10 Test)")
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
    agent_rect = pygame.Rect(ay*CELL_SIZE+10, ax*CELL_SIZE+10, 40, 40)
    pygame.draw.ellipse(screen, (255, 0, 0), agent_rect)  # 빨강 에이전트
    if has_item:
        pygame.draw.circle(screen, (0, 0, 255), agent_rect.center, 8)  # 파랑 박스
    pygame.display.flip()

# ----- 시뮬레이션 -----
env = WarehouseEnv()
state = np.array(env.reset(), dtype=np.float32)
running = True
step_count = 0

while running and step_count < 100:
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

    clock.tick(4)

pygame.quit()