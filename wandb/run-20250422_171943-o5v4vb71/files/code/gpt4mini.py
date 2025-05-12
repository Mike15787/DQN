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

gym.register_envs(ale_py)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
    Q-network: default MLP for low-dimensional (CartPole) or overridden CNN for Atari.
    """
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        # placeholder; actual architecture may be swapped in agent
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.network(x)

class AtariPreprocessor:
    """
    Preprocess Atari frames: grayscale + resize + stack frame_stack frames
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84,84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (Schaul et al., 2016)
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        prio = (abs(error) + 1e-6) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios / prios.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        return indices, samples, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-6) ** self.alpha

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        # environments
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.is_atari = 'Pong' in env_name
        # preprocessor for Atari
        self.preprocessor = AtariPreprocessor()

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # networks
        self.q_net = DQN(self.num_actions).to(self.device)
        self.target_net = DQN(self.num_actions).to(self.device)
        # override architecture for Atari
        if self.is_atari:
            cnn = nn.Sequential(
                nn.Conv2d(4,32,kernel_size=8,stride=4), nn.ReLU(),
                nn.Conv2d(32,64,kernel_size=4,stride=2), nn.ReLU(),
                nn.Conv2d(64,64,kernel_size=3,stride=1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64*7*7,512), nn.ReLU(),
                nn.Linear(512,self.num_actions)
            )
            self.q_net.network = cnn
            self.target_net.network = cnn.__class__(*cnn.__dict__.values()) if False else cnn  # ensure same structure
        self.q_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        # replay buffers
        self.memory = deque(maxlen=args.memory_size)
        self.per_buffer = PrioritizedReplayBuffer(args.memory_size)
        # n-step buffer
        self.n_step = args.n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

        # hyperparams
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.use_ddqn = args.use_ddqn
        self.use_per = args.use_per
        self.use_multistep = args.use_multistep

        self.env_count = 0
        self.train_count = 0
        self.best_reward = float('-inf')
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        st = self._to_tensor(state)
        with torch.no_grad():
            qv = self.q_net(st)
        return qv.argmax().item()

    def _to_tensor(self, state):
        arr = np.array(state, dtype=np.float32)
        if self.is_atari:
            t = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        else:
            t = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        return t

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs,_ = self.env.reset()
            state = self.preprocessor.reset(obs) if self.is_atari else obs
            done=False; total_reward=0; step_count=0
            self.n_step_buffer.clear()
            while not done and step_count<self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc
                next_state = self.preprocessor.step(next_obs) if self.is_atari else next_obs
                # handle n-step
                self.n_step_buffer.append((state,action,reward,next_state,done))
                if self.use_multistep and len(self.n_step_buffer)==self.n_step:
                    R = sum([self.gamma**i*self.n_step_buffer[i][2] for i in range(self.n_step)])
                    s0,a0,_,_,d0 = self.n_step_buffer[0]
                    sn = self.n_step_buffer[-1][3]; dn=self.n_step_buffer[-1][4]
                    transition = (s0,a0,R,sn,dn)
                    self._store(transition)
                else:
                    self._store((state,action,reward,next_state,done))
                state=next_state; total_reward+=reward; self.env_count+=1; step_count+=1
                for _ in range(self.train_per_step): self.train()
                if self.env_count%1000==0:
                    wandb.log({"Env Step":self.env_count, "Eps":self.epsilon})
            wandb.log({"Episode":ep, "Reward":total_reward, "Eps":self.epsilon})
            if ep%20==0:
                er=self.evaluate()
                if er>self.best_reward:
                    self.best_reward=er
                    torch.save(self.q_net.state_dict(), os.path.join(self.save_dir,"best_model.pt"))
                wandb.log({"EvalReward":er})
        
    def _store(self, transition):
        if self.use_per:
            # initial error approx reward
            _,_,r,_,_ = transition
            self.per_buffer.add(transition, error=r)
        else:
            self.memory.append(transition)

    def evaluate(self):
        obs,_=self.test_env.reset()
        state = self.preprocessor.reset(obs) if self.is_atari else obs
        done=False; tot=0
        while not done:
            action = self.select_action(state)
            nxt, r, term, trunc, _ = self.test_env.step(action)
            state = self.preprocessor.step(nxt) if self.is_atari else nxt
            tot+=r; done=term or trunc
        return tot

    def train(self):
        buf = self.per_buffer if self.use_per else self.memory
        if len(buf.buffer if self.use_per else buf) < self.replay_start_size: return
        # epsilon decay
        if self.epsilon>self.epsilon_min: self.epsilon*=self.epsilon_decay
        self.train_count+=1
        # sample
        if self.use_per:
            idxs, batch, weights = buf.sample(self.batch_size)
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        else:
            batch = random.sample(buf, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        st = torch.tensor(np.array(states, dtype=np.float32)).to(self.device)
        nxt = torch.tensor(np.array(next_states, dtype=np.float32)).to(self.device)
        a = torch.tensor(actions, dtype=torch.int64).to(self.device)
        r = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        d = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q = self.q_net(st).gather(1,a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.use_ddqn:
                a_sel = self.q_net(nxt).argmax(1)
                q_next = self.target_net(nxt).gather(1,a_sel.unsqueeze(1)).squeeze(1)
            else:
                q_next = self.target_net(nxt).max(1)[0]
            tgt = r + self.gamma * q_next * (1-d)
        td = tgt - q
        if self.use_per:
            buf.update_priorities(idxs, td.abs().cpu().numpy())
            loss = (weights * td.pow(2)).mean()
        else:
            loss = td.pow(2).mean()
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--save-dir",type=str,default="./results")
    p.add_argument("--wandb-run-name",type=str,default="dqn-run")
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--memory-size",type=int,default=100000)
    p.add_argument("--n-step",type=int,default=3)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--discount-factor",type=float,default=0.99)
    p.add_argument("--epsilon-start",type=float,default=1.0)
    p.add_argument("--epsilon-decay",type=float,default=0.99999)
    p.add_argument("--epsilon-min",type=float,default=0.05)
    p.add_argument("--target-update-frequency",type=int,default=1000)
    p.add_argument("--replay-start-size",type=int,default=50000)
    p.add_argument("--max-episode-steps",type=int,default=10000)
    p.add_argument("--train-per-step",type=int,default=1)
    p.add_argument("--use-ddqn",action="store_true")
    p.add_argument("--use-per",action="store_true")
    p.add_argument("--use-multistep",action="store_true")
    p.add_argument("--task",choices=["Task1","Task2","Task3"],default="Task1")
    args=p.parse_args()
    wandb.init(project="DLP-Lab5-DQN",name=args.wandb_run_name,save_code=True)
    env_map={"Task1":"CartPole-v1","Task2":"ALE/Pong-v5","Task3":"ALE/Pong-v5"}
    agent=DQNAgent(env_map[args.task],args)
    agent.run()
