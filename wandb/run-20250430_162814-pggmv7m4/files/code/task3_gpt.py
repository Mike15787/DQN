# 535507 Spring 2025 – Lab5 Task‑3: Enhanced DQN for Pong‑v5
# Techniques: Double DQN + Prioritized Experience Replay + n‑step return
# Author: (fill your student ID / name)
# -----------------------------------------------------------------------------
# Usage example (2 M env‑steps max):
#   python test_task3.py --total-env-steps 2000000 --wandb-run-name task3_pong
# -----------------------------------------------------------------------------

import os
import random, time, argparse
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import cv2, ale_py, wandb

gym.register_envs(ale_py)

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'done')
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DQN(nn.Module):
    """CNN ⟶ FC architecture for 84×84 × 4 Atari observations."""
    def __init__(self, num_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512), nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.network(x)


class AtariPreprocessor:
    """Gray‑scale → resize 84 × 84 → frame‑stack (4)."""
    def __init__(self, frame_stack: int = 4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def _preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized  # uint8 (84, 84)

    def reset(self, obs):
        frame = self._preprocess(obs)
        self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self._preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


# -----------------------------------------------------------------------------
# Prioritized Replay Buffer
# -----------------------------------------------------------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 1_000_000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.buffer = [None] * capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos, self.size, self.frame = 0, 0, 1
        self.max_priority = 1.0

    def _beta(self):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

    def add(self, transition: Transition, td_error: float = None):
        p = self.max_priority if td_error is None else (abs(td_error) + 1e-6)
        self.buffer[self.pos] = transition
        self.priorities[self.pos] = p

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        if self.size == 0:
            raise ValueError('Buffer empty!')
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # importance‑sampling weights
        beta = self._beta()
        self.frame += 1
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            p = abs(err.item()) + 1e-6
            self.priorities[idx] = p
            self.max_priority = max(self.max_priority, p)


# -----------------------------------------------------------------------------
# DQN Agent (Double DQN + PER + n‑step)
# -----------------------------------------------------------------------------
class DQNAgent:
    def __init__(self, args):
        self.args = args
        self.env = gym.make(args.env_name, render_mode='rgb_array')
        self.test_env = gym.make(args.env_name, render_mode='rgb_array')
        self.n_actions = self.env.action_space.n
        self.prep = AtariPreprocessor()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.q_net = DQN(self.n_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.tgt_net = DQN(self.n_actions).to(self.device)
        self.tgt_net.load_state_dict(self.q_net.state_dict())

        self.opt = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=1.5e-4)

        # Replay buffer (PER)
        self.buffer = PrioritizedReplayBuffer(capacity=args.memory_size,
                                              alpha=args.per_alpha,
                                              beta_start=args.per_beta_start,
                                              beta_frames=args.per_beta_frames)

        # epsilon‑greedy
        self.eps = args.epsilon_start
        self.env_steps = 0
        self.train_steps = 0

        # n‑step helper
        self.n = args.n_step
        self.gamma = args.discount_factor
        self.n_buffer = deque(maxlen=self.n)
        self.checkpoints = [200000, 400000, 600000, 800000, 1000000]
        self.next_ckpt_idx = 0
        self.best_eval = -float('inf')
        os.makedirs(args.save_dir, exist_ok=True)

    # ----------------------------------------------------- interaction
    def select_action(self, state):
        if random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device) / 255.0
            return self.q_net(s).argmax().item()

    def _store_n_step(self, transition):
        self.n_buffer.append(transition)
        if len(self.n_buffer) < self.n:
            return None
        R = sum([self.gamma ** i * self.n_buffer[i].reward for i in range(self.n)])
        next_state = self.n_buffer[-1].next_state
        done = self.n_buffer[-1].done
        first = self.n_buffer[0]
        return Transition(first.state, first.action, R, next_state, done)

    # ----------------------------------------------------- main loop
    def run(self):
        ep = 0
        while self.env_steps < self.args.total_env_steps:
            obs, _ = self.env.reset()
            state = self.prep.reset(obs)
            done, total_r, t = False, 0, 0
            while not done and t < self.args.max_episode_steps:
                act = self.select_action(state)
                next_obs, r, term, trunc, _ = self.env.step(act)
                done = term or trunc
                next_state = self.prep.step(next_obs)

                tr = Transition(state, act, r, next_state, done)
                packed = self._store_n_step(tr)
                if packed:
                    self.buffer.add(packed)

                state = next_state
                total_r += r
                t += 1
                self.env_steps += 1

                # training
                for _ in range(self.args.train_per_step):
                    self.train()

                # eps decay
                if self.eps > self.args.epsilon_min:
                    self.eps *= self.args.epsilon_decay

                if self.env_steps >= self.args.total_env_steps:
                    break
            # end episode
            wandb.log({"Episode Reward": total_r, "Env Steps": self.env_steps})
            print(f"[Ep {ep}] R={total_r:4.1f}  Steps={self.env_steps}  ε={self.eps:.4f}")
            ep += 1

            # save 200000 models
            if (self.next_ckpt_idx < len(self.checkpoints) and self.env_steps >= self.checkpoints[self.next_ckpt_idx]):
                step_mark = self.checkpoints[self.next_ckpt_idx]
                fname = f"LAB5_B11017015_task3_pong{step_mark}.pt"
                self.next_ckpt_idx+=1
                torch.save(self.q_net.state_dict(), os.path.join(self.args.save_dir, fname))
                print(f"*** model saved: {fname} ***")
            
            # evaluation every 10 episodes
            if ep % 20 == 0:
                avg_r = self.evaluate()
                wandb.log({"Eval Reward": avg_r, "Env Steps": self.env_steps})
                print(f"*** (avg_R={avg_r:.2f}) ***")
                
                if avg_r > self.best_eval:
                    self.best_eval = avg_r
                    fname = f"LAB5_B11017015_task3_pong_{self.env_steps}.pt"
                    torch.save(self.q_net.state_dict(), os.path.join(self.args.save_dir, fname))
                    print(f"*** New best model saved: {fname} ***")    

    # ----------------------------------------------------- evaluation
    def evaluate(self, episodes: int = 20):
        rs = []
        for _ in range(episodes):
            obs, _ = self.test_env.reset()
            state = self.prep.reset(obs)
            done, total = False, 0
            while not done:
                with torch.no_grad():
                    s = torch.from_numpy(state).float().unsqueeze(0).to(self.device) / 255.0
                    a = self.q_net(s).argmax().item()
                next_obs, r, term, trunc, _ = self.test_env.step(a)
                done = term or trunc
                total += r
                state = self.prep.step(next_obs)
            rs.append(total)
        return float(np.mean(rs))

    # ----------------------------------------------------- learning
    def train(self):
        if self.buffer.size < max(self.args.replay_start_size, self.args.batch_size):
            return
        self.train_steps += 1

        batch, indices, w = self.buffer.sample(self.args.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.from_numpy(np.stack(states)).float().to(self.device) / 255.0
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device) / 255.0
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        w = w.to(self.device)

        q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_act = self.q_net(next_states).argmax(1) 
            next_q = self.tgt_net(next_states).gather(1, next_act.unsqueeze(1)).squeeze(1)
            tgt = rewards + (self.gamma ** self.n) * next_q * (1 - dones)

        td_errors = tgt - q
        loss = (w * td_errors.pow(2)).mean()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.opt.step()

        # update priorities
        self.buffer.update_priorities(indices, td_errors.detach().abs().cpu())

        # target network update
        if self.train_steps % self.args.target_update_frequency == 0:
            self.tgt_net.load_state_dict(self.q_net.state_dict())

        if self.train_steps % 10000 == 0:
            wandb.log({"Loss": loss.item(), "Env Steps": self.env_steps})
            print(f"[Train {self.train_steps}] loss={loss.item():.4f}")


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, default='ALE/Pong-v5')
    parser.add_argument('--save-dir', type=str, default='./results_pong_task3')
    parser.add_argument('--wandb-run-name', type=str, default='task3_pong')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--memory-size', type=int, default=200_000)
    parser.add_argument('--lr', type=float, default=1e-4) #lr 太低的話會很慢
    parser.add_argument('--discount-factor', type=float, default=0.99)
    parser.add_argument('--epsilon-start', type=float, default=1.0)
    parser.add_argument('--epsilon-decay', type=float, default=0.999995) #3 0.999967
    parser.add_argument('--epsilon-min', type=float, default=0.01)
    parser.add_argument('--target-update-frequency', type=int, default=10000)
    parser.add_argument('--replay-start-size', type=int, default=50_000) #2 改50000
    parser.add_argument('--max-episode-steps', type=int, default=10_000)
    parser.add_argument('--train-per-step', type=int, default=1)
    parser.add_argument('--total-env-steps', type=int, default=2_000_000)
    # PER
    parser.add_argument('--per-alpha', type=float, default=0.6)
    parser.add_argument('--per-beta-start', type=float, default=0.4)
    parser.add_argument('--per-beta-frames', type=int, default=100_000)
    # n‑step
    parser.add_argument('--n-step', type=int, default=3) #1 5真的不好

    args = parser.parse_args()
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    wandb.init(project='DLP-Lab5-DQN-Pong', name=args.wandb_run_name, save_code=True)
    wandb.define_metric('Env Steps')
    wandb.define_metric('Eval Reward', step_metric='Env Steps')

    agent = DQNAgent(args)
    agent.run()
