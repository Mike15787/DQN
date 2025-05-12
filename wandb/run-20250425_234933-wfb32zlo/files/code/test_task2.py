# Spring 2025, 535507 Deep Learning
# Lab5 Task-2 – Vanilla DQN for Pong-v5  (high-dim visual input)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DQN(nn.Module):
    """
        Pong-style CNN ⇨ FC 網路
        Input shape: (batch, 4, 84, 84)
    """
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        ########## YOUR CODE HERE (CNN 5~10 行) ##########
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),                               # (batch, 7*7*64 = 3136)
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        # x 已經是 float32, range [0,1]
        return self.network(x)


class AtariPreprocessor:
    """
        灰階 ➜ 84×84 ➜ 堆 4 帧
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized  # (84,84) uint8

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)],
                            maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class IdentityPreprocessor:
    """在低維環境下什麼都不做（Task-2 不會用到）"""
    def reset(self, obs): return obs
    def step(self, obs):  return obs


class PrioritizedReplayBuffer:
    """Task-2 還沒用 PER，因此僅保留空殼。"""
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
    def add(self, transition, error):               pass
    def sample(self, batch_size):                   pass
    def update_priorities(self, indices, errors):   pass


class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n

        # 依 observation 維度選擇 Preprocessor
        if len(self.env.observation_space.shape) == 3:      # Pong
            self.preprocessor = AtariPreprocessor()
        else:                                               # 其他 toy env
            self.preprocessor = IdentityPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        print("Using device:", self.device)

        self.q_net = DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        ########## YOUR CODE HERE (memory) ##########
        self.memory = deque(maxlen=args.memory_size)
        ########## END OF YOUR CODE ##########

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21    # Pong 最低得分
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # ---------- interaction ----------
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            q = self.q_net(s)
        return q.argmax().item()

    def run(self, episodes=2000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            done, total_r, step = False, 0, 0

            while not done and step < self.max_episode_steps:
                a = self.select_action(state)
                next_obs, r, term, trunc, _ = self.env.step(a)
                done = term or trunc
                next_state = self.preprocessor.step(next_obs)

                ########## YOUR CODE HERE (push replay) ##########
                self.memory.append((state, a, r, next_state, float(done)))
                ########## END OF YOUR CODE ##########

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_r += r
                step += 1
                self.env_count += 1

            print(f"[Ep {ep}] R={total_r:4.1f}  Steps={self.env_count}"
                  f"  ε={self.epsilon:.4f}")
            wandb.log({"Episode": ep, "Reward": total_r,
                       "Env Steps": self.env_count, "Epsilon": self.epsilon})

            if ep % 20 == 0:
                eval_r = self.evaluate()
                print(f"  > Eval Avg R = {eval_r:.2f}")
                wandb.log({"Eval Reward": eval_r})

            if ep % 200 == 0:
                p = os.path.join(self.save_dir, f"pong_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), p)

    def evaluate(self, episodes: int = 5):
        rs = []
        for _ in range(episodes):
            obs, _ = self.test_env.reset()
            state = self.preprocessor.reset(obs)
            done, tr = False, 0
            while not done:
                s = torch.from_numpy(state).float().unsqueeze(0)\
                                            .to(self.device) / 255.0
                a = self.q_net(s).argmax().item()
                next_obs, r, term, trunc, _ = self.test_env.step(a)
                done = term or trunc
                tr += r
                state = self.preprocessor.step(next_obs)
            rs.append(tr)
        return np.mean(rs)

    # ---------- learning ----------
    def train(self):
        if len(self.memory) < max(self.replay_start_size, self.batch_size):
            return
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        ########## YOUR CODE HERE (<5 lines) ##########
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        ########## END OF YOUR CODE ##########

        states = torch.from_numpy(np.stack(states)).float().to(self.device) \
                 / 255.0
        next_states = torch.from_numpy(np.stack(next_states))\
                     .float().to(self.device) / 255.0
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        ########## YOUR CODE HERE (~10 lines) ##########
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            tgt = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.mse_loss(q, tgt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        ########## END OF YOUR CODE ##########

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 10000 == 0:
            print(f"[Train {self.train_count}] loss={loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5")
    parser.add_argument("--save-dir", type=str, default="./results_pong")
    parser.add_argument("--wandb-run-name", type=str, default="task2_pong")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=10_000)
    parser.add_argument("--replay-start-size", type=int, default=50_000)
    parser.add_argument("--max-episode-steps", type=int, default=100_000)
    parser.add_argument("--train-per-step", type=int, default=1)
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-Pong",
               name=args.wandb_run_name, save_code=True)

    agent = DQNAgent(env_name=args.env_name, args=args)
    agent.run()
