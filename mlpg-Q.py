import pygame
import numpy as np
import random
import time

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
            reward = 20
        ##elif self.agent_pos == RETURN_LOC and not self.has_item:
        ##    self.done = True
        ##    reward = 5

        return self.get_state(), reward, self.done

# ----- Q-learning -----
Q_table = {}
alpha = 0.1
gamma = 0.9
epsilon = 0.1

env = WarehouseEnv()
for episode in range(1000):
    state = env.reset()
    for _ in range(50):
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            q_values = [Q_table.get((state, a), 0) for a in actions]
            action = actions[np.argmax(q_values)]

        next_state, reward, done = env.step(action)
        best_next_q = max([Q_table.get((next_state, a), 0) for a in actions])
        Q_table[(state, action)] = Q_table.get((state, action), 0) + alpha * (
            reward + gamma * best_next_q - Q_table.get((state, action), 0)
        )
        state = next_state
        if done:
            break

print("✅ 학습 완료")

# ----- Pygame GUI -----
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Warehouse Robot RL")
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

# ----- GUI 루프 -----
state = env.reset()
running = True
step_count = 0

while running and step_count < 3000:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    q_values = [Q_table.get((state, a), 0) for a in actions]
    action = actions[np.argmax(q_values)]
    next_state, reward, done = env.step(action)

    draw_grid(env.agent_pos, env.has_item)
    print(f"{step_count} 위치: {env.agent_pos}, 액션: {action}, 보상: {reward}")
    state = next_state
    step_count += 1

    if done:
        time.sleep(1)
        running = False

    clock.tick(2)

pygame.quit()