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
        return np.array([
            self.agent_pos[0] / GRID_SIZE,
            self.agent_pos[1] / GRID_SIZE,
            float(self.has_item)
        ], dtype=np.float32)

    def step(self, action):
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

# ----- PPO 에이전트 -----
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.fc(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)

# ----- 학습 파라미터 -----
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = WarehouseEnv()

state_dim = 3
action_dim = 4
policy_net = PolicyNetwork(state_dim, action_dim).to(dev)
value_net = ValueNetwork(state_dim).to(dev)

policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)

# ----- PPO 학습 -----
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    values = values + [0]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] - values[step]
        gae = delta + gamma * lam * gae
        returns.insert(0, gae + values[step])
    return returns

epochs = 1000
for epoch in range(epochs):
    log_probs = []
    values = []
    rewards = []
    states = []
    actions_taken = []

    state = env.reset()
    for t in range(100):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(dev)
        logits = policy_net(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        next_state, reward, done = env.step(actions[action.item()])

        log_probs.append(dist.log_prob(action))
        values.append(value_net(state_tensor).item())
        rewards.append(reward)
        states.append(state_tensor.squeeze(0))
        actions_taken.append(action)

        state = next_state
        if done:
            break

    # GAE & Advantage
    returns = compute_gae(rewards, values)
    returns = torch.FloatTensor(returns).to(dev)
    values = torch.FloatTensor(values).to(dev)
    states = torch.stack(states).to(dev)
    actions_taken = torch.stack(actions_taken).to(dev)
    log_probs = torch.stack(log_probs).to(dev)

    advantage = returns - values.detach()

    # Policy update
    new_logits = policy_net(states)
    new_probs = torch.softmax(new_logits, dim=-1)
    new_dist = torch.distributions.Categorical(new_probs)
    new_log_probs = new_dist.log_prob(actions_taken)

    ratio = (new_log_probs - log_probs).exp()
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
    policy_loss = -torch.min(surr1, surr2).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # Value update
    value_pred = value_net(states).squeeze()
    value_loss = nn.MSELoss()(value_pred, returns)

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: Policy Loss = {policy_loss.item():.3f}, Value Loss = {value_loss.item():.3f}")

print("\n✅ PPO 학습 완료")
# ----- 정책 저장 -----
torch.save(policy_net.state_dict(), "ppo_policy.pth")
print("📁 정책이 'ppo_policy.pth' 파일로 저장되었습니다.")