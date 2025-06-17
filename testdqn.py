import torch
import pygame
import time
import numpy as np
from newDQN import WarehouseEnvCNN, CNNQNetwork

CELL_SIZE = 100

def draw_env(state, grid_size):
    pygame.init()
    screen = pygame.display.set_mode((CELL_SIZE * grid_size, CELL_SIZE * grid_size))
    pygame.display.set_caption("DQN Agent (Policy Test)")
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

def run_policy():
    pygame.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 환경 생성 (랜덤 장애물 포함)
    env = WarehouseEnvCNN(grid_size=5, obstacle_ratio=0.1)

    # 모델 불러오기
    q_net = CNNQNetwork(grid_size=env.grid_size)
    q_net.load_state_dict(torch.load("dqn_cnn_policy.pth", map_location=device))
    q_net.to(device)
    q_net.eval()

    state = env.reset()
    done = False
    step = 0
    total_reward = 0 

    while True:
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


        time.sleep(0.4)
        step += 1

    pygame.quit()

if __name__ == "__main__":
    run_policy()