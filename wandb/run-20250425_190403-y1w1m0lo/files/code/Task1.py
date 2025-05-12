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
    if isinstance(m, (nn.Conv2d, nn.Linear)) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        input_dim = 4          # CartPole state = (x, v, θ, ω)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        return self.network(x)


class AtariPreprocessor:
    """
        用於 Atari ,CartPole 用不到。
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
        #self.preprocessor = AtariPreprocessor() 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.q_net = DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        #新增雙向佇列 用於簡單記憶儲存
        self.memory = deque(maxlen=args.memory_size)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = 0  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps #ok
        self.replay_start_size = args.replay_start_size #ok
        self.target_update_frequency = args.target_update_frequency #ok
        self.train_per_step = args.train_per_step # not
        self.save_dir = args.save_dir #ok
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
            state, _ = self.env.reset()

            #state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                if done:
                    break

                next_state = next_obs
                self.memory.append((state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1  #step
                step_count += 1 #env_steps

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########   
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        state, _ = self.test_env.reset()
        #state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_obs

        return total_reward
    
    def train(self):

        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, ns, d = zip(*batch)

        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        #states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        #next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        #actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        #rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        #dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        #q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        s  = torch.from_numpy(np.vstack(s)).float().to(self.device)
        ns = torch.from_numpy(np.vstack(ns)).float().to(self.device)
        a  = torch.tensor(a, dtype=torch.int64).to(self.device)
        r  = torch.tensor(r, dtype=torch.float32).to(self.device)
        d  = torch.tensor(d, dtype=torch.float32).to(self.device)
        q_sa = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 
      
        with torch.no_grad():
            tgt = r + self.gamma * self.target_net(ns).max(1)[0] * (1 - d)
        loss = nn.functional.mse_loss(q_sa, tgt)

        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
      
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_sa.mean().item():.3f} std: {q_values.std().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="task1_cartpole")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--train-per-step", type=int, default=1)
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-CartPole",
               name=args.wandb_run_name, save_code=True)

    agent = DQNAgent(args=args)
    agent.run()
