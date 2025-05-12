# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL – Task 1 (CartPole-v1 Vanilla DQN)

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
        Simple 2-hidden-layer MLP for CartPole (state dim = 4).
        For Task 2 / 3，請在另一份檔案改為 CNN。
    """
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        input_dim = 4          # CartPole state = (x, v, θ, ω)
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        return self.network(x)


class AtariPreprocessor:
    """
        原用於 Atari；CartPole 不用到。
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

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
    """對低維觀測做『甚麼都不做』的前處理。"""
    def reset(self, obs):
        return obs

    def step(self, obs):
        return obs


class PrioritizedReplayBuffer:
    """Task 1 不用 PER；僅做佔位。"""
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity

    # 以下留空即可
    def add(self, transition, error):  pass
    def sample(self, batch_size):      pass
    def update_priorities(self, indices, errors):  pass


class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n

        # ★ 依觀測維度選擇前處理器
        if len(self.env.observation_space.shape) == 1:
            self.preprocessor = IdentityPreprocessor()
            self.state_dim = self.env.observation_space.shape[0]
        else:
            self.preprocessor = AtariPreprocessor()
            self.state_dim = 84  # Dummy, not used in Task 1

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
        self.best_reward = 0
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state))\
                           .float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = \
                    self.env.step(action)
                done = terminated or truncated

                next_state = self.preprocessor.step(next_obs)
                ########## YOUR CODE HERE (push into replay) ##########
                self.memory.append(
                    (state, action, reward, next_state, float(done))
                )
                ########## END OF YOUR CODE ##########

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

            print(f"[Ep {ep}] total_reward={total_reward:.1f} "
                  f"env_steps={self.env_count} epsilon={self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Steps": self.env_count,
                "Epsilon": self.epsilon
            })

            if ep % 20 == 0:         # evaluate
                eval_reward = self.evaluate()
                print(f"  > Eval reward={eval_reward:.1f}")
                wandb.log({"Eval Reward": eval_reward})

            if ep % 100 == 0:        # checkpoint
                path = os.path.join(self.save_dir, f"cartpole_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), path)

    def evaluate(self, episodes: int = 5):
        rewards = []
        for _ in range(episodes):
            obs, _ = self.test_env.reset()
            state = self.preprocessor.reset(obs)
            done, total_r = False, 0
            while not done:
                s = torch.from_numpy(np.array(state))\
                        .float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    a = self.q_net(s).argmax().item()
                next_obs, r, term, trunc, _ = self.test_env.step(a)
                done = term or trunc
                total_r += r
                state = self.preprocessor.step(next_obs)
            rewards.append(total_r)
        return np.mean(rewards)

    def train(self):
        if len(self.memory) < max(self.replay_start_size, self.batch_size):
            return

        # ε-greedy decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        ########## YOUR CODE HERE (<5 lines) ##########
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        ########## END OF YOUR CODE ##########

        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states))\
                          .float().to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # current Q
        q_values = self.q_net(states)\
                     .gather(1, actions.unsqueeze(1)).squeeze(1)

        ########## YOUR CODE HERE (~10 lines) ##########
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        ########## END OF YOUR CODE ##########

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            print(f"[Train {self.train_count}] loss={loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="task1_cartpole")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--train-per-step", type=int, default=1)
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-CartPole",
               name=args.wandb_run_name, save_code=True)

    agent = DQNAgent(args=args)
    agent.run()
