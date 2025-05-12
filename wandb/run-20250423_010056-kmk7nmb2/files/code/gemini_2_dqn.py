# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # Added for loss function
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
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Use Xavier Initialization for Linear layers, Kaiming for Conv layers
        if isinstance(m, nn.Linear):
             nn.init.xavier_uniform_(m.weight)
        else: # nn.Conv2d
             nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):

    def __init__(self, input_shape, num_actions, env_name="CartPole-v1"):
        super(DQN, self).__init__()
        self.env_name = env_name
        self.num_actions = num_actions

        if "CartPole" in self.env_name:
            # Simple MLP for low-dimensional state space (CartPole)
            state_dim = input_shape[0]
            self.network = nn.Sequential(
                nn.Linear(state_dim, 128), # Increased width
                nn.ReLU(),
                nn.Linear(128, 128),       # Added a layer
                nn.ReLU(),
                nn.Linear(128, num_actions)
            )
        elif "Pong" in self.env_name or "ALE/" in self.env_name: # Check for Atari envs
             # CNN for high-dimensional visual input (Atari)
             # Input shape: (stack_size, height, width) e.g., (4, 84, 84)
            in_channels = input_shape[0] # Should be frame stack size
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), # (N, 4, 84, 84) -> (N, 32, 20, 20)
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),          # (N, 32, 20, 20) -> (N, 64, 9, 9)
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),          # (N, 64, 9, 9) -> (N, 64, 7, 7)
                nn.ReLU(),
                nn.Flatten() # Flatten the output for the FC layers: 64 * 7 * 7 = 3136
            )
            # Calculate the flattened size automatically (more robust)
            with torch.no_grad():
                 dummy_input = torch.zeros(1, *input_shape)
                 cnn_out_dim = self.cnn(dummy_input).shape[1]

            self.fc = nn.Sequential(
                nn.Linear(cnn_out_dim, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )
            self.network = lambda x: self.fc(self.cnn(x / 255.0)) # Combine CNN and FC, normalize pixels
        else:
            raise ValueError(f"Unsupported environment name for DQN architecture: {self.env_name}")


    def forward(self, x):
        # Normalize input if it's image-based (Atari)
        # Normalization moved into network definition for Atari case
        # if "Pong" in self.env_name or "ALE/" in self.env_name:
        #     x = x / 255.0 # Normalize pixel values
        return self.network(x)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        # Check if obs is already grayscale (shape might be (H, W) or (H, W, 1))
        if len(obs.shape) == 2 or obs.shape[2] == 1:
             gray = obs.squeeze() # Remove channel dim if it exists
        else: # Assume RGB
             gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized # shape (84, 84)

    def reset(self, obs):
        # obs should be the initial observation from env.reset()
        frame = self.preprocess(obs) # shape (84, 84)
        # Initialize the deque with the first frame repeated frame_stack times
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        # Stack frames along a new dimension (channel dimension for CNN)
        return np.stack(self.frames, axis=0) # shape (4, 84, 84)

    def step(self, obs):
        # obs is the observation from env.step()
        frame = self.preprocess(obs) # shape (84, 84)
        self.frames.append(frame)
        # Stack frames along the channel dimension
        return np.stack(self.frames, axis=0) # shape (4, 84, 84)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
        Uses basic numpy arrays for priorities, not optimized SumTree.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.0001, epsilon=1e-5):
        self.capacity = capacity
        self.alpha = alpha # Priority exponent
        self.beta = beta # Initial importance sampling exponent
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon # Small constant to ensure non-zero priority
        self.buffer = [None] * capacity # Use list with fixed size for easier indexing
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0 # Current insertion position
        self.size = 0 # Current number of elements in buffer

    def add(self, transition, error=None):
        ########## YOUR CODE HERE (for Task 3) ##########
        # If error is None (first time adding), use max priority
        # Otherwise, use the provided error to calculate priority
        max_prio = np.max(self.priorities) if self.size > 0 else 1.0
        priority = max_prio if error is None else (abs(error) + self.epsilon) ** self.alpha

        self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        ########## END OF YOUR CODE (for Task 3) ##########
        return

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########
        if self.size == 0:
             return [], [], [] # Return empty lists if buffer is empty

        # Calculate sampling probabilities P(i) = p_i^alpha / sum(p_k^alpha)
        priorities_subset = self.priorities[:self.size]
        probs = priorities_subset ** self.alpha
        probs /= probs.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(self.size, batch_size, p=probs)

        # Get transitions for sampled indices
        transitions = [self.buffer[i] for i in indices]

        # Calculate Importance Sampling (IS) weights: w_i = (N * P(i))^(-beta) / max(w_k)
        total_n = self.size
        sampling_probabilities = probs[indices]

        # Anneal beta towards 1.0
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        is_weights = np.power(total_n * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max() # Normalize weights for stability
        is_weights = torch.tensor(is_weights, dtype=torch.float32) # Convert to tensor

        ########## END OF YOUR CODE (for Task 3) ##########
        return transitions, indices, is_weights

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########
        for idx, error in zip(indices, errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority
        ########## END OF YOUR CODE (for Task 3) ##########
        return

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env_name = env_name
        self.args = args
        self.env = gym.make(env_name) # Use default render mode unless specified
        self.test_env = gym.make(env_name, render_mode="rgb_array") # Render test env for video

        # Determine state shape and action space size
        if "CartPole" in self.env_name:
            self.state_shape = self.env.observation_space.shape
            self.is_atari = False
            self.preprocessor = None # No preprocessor needed for CartPole
            self.best_reward = 0 # Or float('-inf')
        elif "Pong" in self.env_name or "ALE/" in self.env_name:
             self.is_atari = True
             self.preprocessor = AtariPreprocessor(frame_stack=args.frame_stack)
             # Manually define shape after preprocessing and stacking
             self.state_shape = (args.frame_stack, 84, 84)
             self.best_reward = -21 # Standard baseline for Pong
        else:
            raise ValueError(f"Unsupported environment: {env_name}")

        self.num_actions = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"Environment: {self.env_name}")
        print(f"State shape: {self.state_shape}, Num actions: {self.num_actions}")


        # Initialize Networks
        self.q_net = DQN(self.state_shape, self.num_actions, self.env_name).to(self.device)
        self.target_net = DQN(self.state_shape, self.num_actions, self.env_name).to(self.device)
        self.q_net.apply(init_weights) # Apply weight initialization
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval() # Target network is only for inference

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=1.5e-4 if self.is_atari else 1e-8) # Use eps for Adam as in Rainbow for Atari

        # Replay Buffer
        if args.use_per:
             print("Using Prioritized Experience Replay")
             self.memory = PrioritizedReplayBuffer(
                 args.memory_size,
                 alpha=args.per_alpha,
                 beta=args.per_beta_start,
                 beta_increment_per_sampling= (1.0 - args.per_beta_start) / args.total_steps # Linear annealing
             )
        else:
             print("Using Standard Experience Replay (Uniform Sampling)")
             self.memory = deque(maxlen=args.memory_size)

        # Hyperparameters
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.n_steps = args.n_steps # For multi-step return (currently 1-step logic is implemented)
        self.epsilon = args.epsilon_start
        # Use linear epsilon decay based on total steps for more control
        self.epsilon_decay_rate = (args.epsilon_start - args.epsilon_min) / args.epsilon_decay_steps
        self.epsilon_min = args.epsilon_min

        self.env_count = 0 # Total environment steps taken
        self.train_count = 0 # Total training updates performed
        self.max_episode_steps = args.max_episode_steps # Max steps per episode
        self.replay_start_size = args.replay_start_size # Steps before starting training
        self.target_update_frequency = args.target_update_frequency # Steps between target net updates
        self.train_frequency = args.train_frequency # Environment steps per training update
        self.save_frequency = args.save_frequency # Environment steps per model save
        self.eval_frequency = args.eval_frequency # Environment steps per evaluation run
        self.eval_episodes = args.eval_episodes # Number of episodes for evaluation

        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)


    def _preprocess_state(self, obs, reset=False):
        """Helper to handle preprocessing based on environment type."""
        if self.is_atari:
            if reset:
                return self.preprocessor.reset(obs)
            else:
                return self.preprocessor.step(obs)
        else:
            # For CartPole, observation is already the state
            return obs # No preprocessing needed

    def select_action(self, state, evaluation=False):
        """Selects action using epsilon-greedy policy or greedily for evaluation."""
        # Decay epsilon based on env steps
        # Only decay if not evaluating
        if not evaluation:
             self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay_rate)

             # Epsilon-greedy for training
             if random.random() < self.epsilon:
                 return random.randint(0, self.num_actions - 1)

        # Greedy action selection (for exploitation or evaluation)
        try:
             # Ensure state is numpy array before converting to tensor
             if not isinstance(state, np.ndarray):
                   state = np.array(state)

             # Add batch dimension if missing
             if state.ndim == len(self.state_shape): # e.g., (4, 84, 84) vs expected (1, 4, 84, 84)
                   state = np.expand_dims(state, axis=0)

             state_tensor = torch.from_numpy(state).float().to(self.device)
             with torch.no_grad():
                 q_values = self.q_net(state_tensor)
             return q_values.argmax().item()
        except Exception as e:
             print(f"Error during action selection:")
             print(f"  State type: {type(state)}")
             if isinstance(state, np.ndarray):
                  print(f"  State shape: {state.shape}")
             print(f"  Evaluation flag: {evaluation}")
             print(f"  Error: {e}")
             # Fallback to random action in case of error
             return random.randint(0, self.num_actions - 1)


    def run(self, total_steps=1000000):
        """Main training loop."""
        start_time = time.time()
        obs, _ = self.env.reset()
        state = self._preprocess_state(obs, reset=True)
        total_reward = 0
        episode_reward = 0
        episode_steps = 0
        episode_count = 0

        # Fill replay buffer partially before starting training
        print(f"Collecting initial experiences ({self.replay_start_size} steps)...")
        for _ in range(self.replay_start_size):
            action = self.env.action_space.sample() # Random actions to fill buffer
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            next_state = self._preprocess_state(next_obs, reset=False)

            # Store transition (handle PER vs standard deque)
            if self.args.use_per:
                 self.memory.add((state, action, reward, next_state, done)) # Add with max priority initially
            else:
                 self.memory.append((state, action, reward, next_state, done))

            self.env_count += 1
            state = next_state if not done else self._preprocess_state(self.env.reset()[0], reset=True)
        print(f"Initial experience collection complete. Starting training...")
        state = self._preprocess_state(self.env.reset()[0], reset=True) # Reset again to start properly

        while self.env_count < total_steps:
            action = self.select_action(state)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            next_state = self._preprocess_state(next_obs, reset=False)

            # Store transition
            if self.args.use_per:
                 # Calculate initial error for priority? Or just add with max?
                 # Adding with max priority is simpler and common. Update happens after sampling.
                 self.memory.add((state, action, reward, next_state, done))
            else:
                 self.memory.append((state, action, reward, next_state, done))


            state = next_state
            total_reward += reward
            episode_reward += reward
            episode_steps += 1
            self.env_count += 1


            # Train the network
            if self.env_count % self.train_frequency == 0:
                loss, mean_q = self.train()
                if loss is not None and self.train_count % 1000 == 0: # Log training loss less frequently
                     wandb.log({
                         "Metrics/Training Loss": loss,
                         "Metrics/Mean Q-value": mean_q,
                         "Progress/Environment Steps": self.env_count,
                         "Progress/Training Updates": self.train_count,
                         "Progress/Epsilon": self.epsilon,
                     })


            # Update target network
            if self.env_count % self.target_update_frequency == 0:
                 self.target_net.load_state_dict(self.q_net.state_dict())
                 # print(f"Step {self.env_count}: Target network updated.")

            # Evaluate performance
            if self.env_count % self.eval_frequency == 0:
                 eval_reward = self.evaluate()
                 wandb.log({
                     "Result/Evaluation Reward": eval_reward,
                     "Progress/Environment Steps": self.env_count,
                     "Progress/Training Updates": self.train_count,
                 })
                 print(f"[Eval] Step: {self.env_count} | Eval Reward: {eval_reward:.2f} | Epsilon: {self.epsilon:.3f}")

                 if eval_reward > self.best_reward:
                      self.best_reward = eval_reward
                      model_path = os.path.join(self.save_dir, "best_model.pt")
                      torch.save(self.q_net.state_dict(), model_path)
                      print(f"*** Saved new best model to {model_path} with reward {eval_reward:.2f} ***")

            # Save model periodically
            if self.env_count % self.save_frequency == 0:
                model_path = os.path.join(self.save_dir, f"model_step{self.env_count}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                # print(f"Saved model checkpoint to {model_path}")

                # Specific saves for Task 3 grading
                if self.is_atari and self.env_count in [200000, 400000, 600000, 800000, 1000000]:
                     task3_path = os.path.join(self.save_dir, f"LAB5_{wandb.run.id}_task3_pong{self.env_count}.pt")
                     torch.save(self.q_net.state_dict(), task3_path)
                     print(f"Saved Task 3 snapshot: {task3_path}")


            # Handle episode end
            if done or episode_steps >= self.max_episode_steps:
                 episode_count += 1
                 time_elapsed = time.time() - start_time
                 steps_per_sec = self.env_count / time_elapsed if time_elapsed > 0 else 0
                 print(f"Ep: {episode_count} | Step: {self.env_count} | Ep Reward: {episode_reward} | Ep Steps: {episode_steps} | Epsilon: {self.epsilon:.3f} | SPS: {steps_per_sec:.2f}")
                 wandb.log({
                     "Result/Episode Reward": episode_reward,
                     "Result/Episode Steps": episode_steps,
                     "Progress/Episode Count": episode_count,
                     "Progress/Environment Steps": self.env_count,
                     "Progress/Steps Per Second": steps_per_sec,
                 })
                 # --- Specific model saves for Task 1 & 2 based on best eval reward ---
                 if self.env_count >= self.eval_frequency: # Ensure at least one eval has run
                      if not self.is_atari: # Task 1 CartPole
                           task1_path = os.path.join(self.save_dir, f"LAB5_{wandb.run.id}_task1_cartpole.pt")
                           if os.path.exists(os.path.join(self.save_dir, "best_model.pt")):
                                torch.save(torch.load(os.path.join(self.save_dir, "best_model.pt")), task1_path)
                                print(f"Saved best model snapshot for Task 1: {task1_path}")
                      elif not self.args.use_ddqn and not self.args.use_per: # Task 2 Vanilla Pong
                           task2_path = os.path.join(self.save_dir, f"LAB5_{wandb.run.id}_task2_pong.pt")
                           if os.path.exists(os.path.join(self.save_dir, "best_model.pt")):
                                torch.save(torch.load(os.path.join(self.save_dir, "best_model.pt")), task2_path)
                                print(f"Saved best model snapshot for Task 2: {task2_path}")


                 # Reset episode variables
                 obs, _ = self.env.reset()
                 state = self._preprocess_state(obs, reset=True)
                 episode_reward = 0
                 episode_steps = 0



    def evaluate(self, render=False):
        """Evaluates the agent's performance over several episodes."""
        print("--- Starting Evaluation ---")
        total_eval_reward = 0
        temp_test_env = gym.make(self.env_name, render_mode="human" if render else "rgb_array")

        for i in range(self.eval_episodes):
            obs, _ = temp_test_env.reset()
            state = self._preprocess_state(obs, reset=True) # Use the agent's preprocessor
            done = False
            episode_reward = 0
            episode_steps = 0
            while not done and episode_steps < self.max_episode_steps:
                 action = self.select_action(state, evaluation=True) # Use greedy policy
                 next_obs, reward, terminated, truncated, _ = temp_test_env.step(action)
                 done = terminated or truncated
                 episode_reward += reward
                 state = self._preprocess_state(next_obs, reset=False)
                 episode_steps += 1
                 if render:
                    time.sleep(0.01) # Slow down rendering a bit
            total_eval_reward += episode_reward
            # print(f"Eval Ep {i+1}/{self.eval_episodes} Reward: {episode_reward}")

        temp_test_env.close()
        avg_eval_reward = total_eval_reward / self.eval_episodes
        print(f"--- Evaluation Complete | Average Reward: {avg_eval_reward:.2f} ---")
        return avg_eval_reward


    def train(self):
        """Samples a batch from replay buffer and performs a gradient update."""
        if len(self.memory) < self.batch_size: # Should use replay_start_size check before calling train
            return None, None # Not enough samples yet

        self.train_count += 1

        # Sample mini-batch
        if self.args.use_per:
             transitions, indices, is_weights = self.memory.sample(self.batch_size)
             is_weights = is_weights.to(self.device) # Move weights to device
        else:
             transitions = random.sample(self.memory, self.batch_size)
             indices = None # Not needed for uniform sampling
             is_weights = torch.ones(self.batch_size).to(self.device) # Uniform weights

        # Unpack transitions
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Convert to torch tensors
        # Handle potential inconsistencies in numpy array creation
        try:
             states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
             next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        except ValueError as e:
             print("Error converting states/next_states to tensor. Check shapes:")
             for i, s in enumerate(states): print(f"State {i} shape: {s.shape}")
             for i, ns in enumerate(next_states): print(f"Next State {i} shape: {ns.shape}")
             raise e

        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # Convert dones (boolean/int) to float (0.0 or 1.0) for multiplication
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # --- Calculate Target Q-values ---
        with torch.no_grad():
            if self.args.use_ddqn:
                # Double DQN: Select action using Q_net, evaluate using Target_net
                next_q_values_qnet = self.q_net(next_states) # Q_main(s')
                best_next_actions = next_q_values_qnet.argmax(1) # argmax_a' Q_main(s', a')
                next_q_values_target = self.target_net(next_states) # Target_net(s')
                # Q_target(s', argmax_a' Q_main(s', a'))
                target_max_q = next_q_values_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: Select and evaluate using Target_net
                next_q_values_target = self.target_net(next_states) # Target_net(s')
                target_max_q = next_q_values_target.max(1)[0] # max_a' Q_target(s', a')

            # Calculate target: r + gamma * Q_target(s', a*) * (1 - done)
            # NOTE: For N-step returns, 'rewards' and 'gamma' would need to be adjusted based on the n-step trajectory.
            # 'next_states' would become s_{t+n} and 'dones' would be done_{t+n}.
            # Requires modifying buffer or sampling logic significantly.
            target_q_values = rewards + (self.gamma * target_max_q * (1 - dones)) # Using gamma for 1-step

        # --- Calculate Current Q-values ---
        # Q(s, a) for the actions actually taken
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- Calculate Loss ---
        # TD Error: target_q - current_q
        td_errors = target_q_values - q_values

        # Apply Importance Sampling weights (for PER)
        # Loss = mean( IS_weight * (TD_Error)^2 ) -> Use Huber loss for stability
        loss = (is_weights * F.smooth_l1_loss(q_values, target_q_values, reduction='none')).mean()
        # loss = (is_weights * (td_errors ** 2)).mean() # MSE Loss weighted

        # --- Gradient Update ---
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping (helps stabilize training, especially with CNNs)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=self.args.grad_clip_norm)
        self.optimizer.step()

        # --- Update Priorities in PER Buffer ---
        if self.args.use_per:
            # Use absolute TD errors for priorities
            abs_td_errors = td_errors.abs().detach().cpu().numpy()
            self.memory.update_priorities(indices, abs_td_errors)

        return loss.item(), q_values.mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Agent Training")

    # --- Task Selection (REQUIRED) ---
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3],
                        help="Task to run: 1 (Vanilla DQN CartPole), 2 (Vanilla DQN Pong), 3 (Enhanced DQN Pong)")

    # --- Environment and Logging ---
    # env_name, use_ddqn, use_per will be set based on --task
    parser.add_argument("--save-dir", type=str, default="./results", help="Directory to save models and logs")
    parser.add_argument("--wandb-project-name", type=str, default="DLP-Lab5-DQN", help="WandB project name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # --- Core Training Hyperparameters ---
    parser.add_argument("--total-steps", type=int, default=1000000, help="Total environment steps to train for (Adjust per task!)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (Adjust per task!)")
    parser.add_argument("--discount-factor", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument("--train-frequency", type=int, default=4, help="Env steps between training updates (Adjust per task!)")
    parser.add_argument("--target-update-frequency", type=int, default=5000, help="Env steps between target net updates (Adjust per task!)")
    parser.add_argument("--grad-clip-norm", type=float, default=10.0, help="Max norm for gradient clipping")

    # --- Replay Buffer ---
    parser.add_argument("--memory-size", type=int, default=100000, help="Replay buffer size (Adjust per task!)")
    parser.add_argument("--replay-start-size", type=int, default=10000, help="Steps before training starts (Adjust per task!)")

    # --- Epsilon Greedy Exploration ---
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Starting value for epsilon")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum value for epsilon (Adjust per task!)")
    parser.add_argument("--epsilon-decay-steps", type=int, default=200000, help="Steps for linear epsilon decay (Adjust per task!)")

    # --- Prioritized Experience Replay (PER) ---
    # use_per is set by --task 3
    parser.add_argument("--per-alpha", type=float, default=0.6, help="Alpha parameter for PER (priority exponent)")
    parser.add_argument("--per-beta-start", type=float, default=0.4, help="Starting beta parameter for PER (IS exponent)")

    # --- N-Step Returns ---
    # use_ddqn is set by --task 3
    parser.add_argument("--n-steps", type=int, default=1, help="N-step returns (NOTE: only 1-step buffer logic currently implemented)")

    # --- Atari Specific ---
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of frames to stack for Atari")

    # --- Evaluation and Saving ---
    parser.add_argument("--eval-frequency", type=int, default=25000, help="Steps between evaluations (Adjust per task!)")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Number of episodes for evaluation (Lab requires 20)")
    parser.add_argument("--save-frequency", type=int, default=100000, help="Steps between saving model checkpoints")
    parser.add_argument("--max-episode-steps", type=int, default=50000, help="Maximum steps per episode (Adjust per task! e.g., 500 for CartPole-v1)")

    args = parser.parse_args()

    # --- Set task-specific configurations ---
    task_label = ""
    if args.task == 1:
        args.env_name = "CartPole-v1"
        args.use_ddqn = False
        args.use_per = False
        task_label = "Task1-CartPole"
        # User MUST provide appropriate HPs via CLI for CartPole
        print("--- Running Task 1: Vanilla DQN on CartPole ---")
        print("!!! Ensure command line arguments (lr, memory-size, total-steps, etc.) are set appropriately for CartPole !!!")
    elif args.task == 2:
        args.env_name = "ALE/Pong-v5"
        args.use_ddqn = False
        args.use_per = False
        task_label = "Task2-Pong-Vanilla"
        print("--- Running Task 2: Vanilla DQN on Pong ---")
        print("!!! Ensure command line arguments are set appropriately for Vanilla Pong !!!")
    elif args.task == 3:
        args.env_name = "ALE/Pong-v5"
        args.use_ddqn = True
        args.use_per = True
        task_label = "Task3-Pong-Enhanced"
        print("--- Running Task 3: Enhanced DQN (DDQN+PER) on Pong ---")
        print("!!! Ensure command line arguments are set appropriately for Enhanced Pong !!!")

    # --- Final Setup ---
    # Calculate PER beta annealing factor if PER is used
    if args.use_per:
        # Avoid division by zero if train_frequency is 0 or total_steps is small
        total_updates = (args.total_steps / args.train_frequency) if args.train_frequency > 0 else 0
        args.per_beta_increment_per_sampling = (1.0 - args.per_beta_start) / total_updates if total_updates > 0 else 0
    else:
        args.per_beta_increment_per_sampling = 0

    # Set WandB run name based on actual settings
    run_name = f"{task_label}"
    if args.use_ddqn: run_name += "-DDQN"
    if args.use_per: run_name += "-PER"
    run_name += f"-lr{args.lr}-bs{args.batch_size}-mem{args.memory_size}"
    args.wandb_run_name = run_name

    print("\n--- Final Configuration (from command line or defaults) ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("-----------------------------------------------------------\n")

    # Initialize WandB
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=args, save_code=True)

    # Set seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Create and run the agent
    agent = DQNAgent(env_name=args.env_name, args=args)
    try:
        agent.run(total_steps=args.total_steps)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # Save final model
        final_model_path = os.path.join(args.save_dir, f"{task_label}_final_model.pt")
        torch.save(agent.q_net.state_dict(), final_model_path)
        print(f"Saved final model to {final_model_path}")

        # --- Save task-specific 'best' models (if best_model.pt exists) ---
        best_model_src = os.path.join(args.save_dir, "best_model.pt")
        if os.path.exists(best_model_src):
            if args.task == 1:
                task1_path = os.path.join(args.save_dir, f"LAB5_{wandb.run.id}_task1_cartpole.pt")
                torch.save(torch.load(best_model_src), task1_path)
                print(f"Saved BEST model snapshot for Task 1: {task1_path}")
            elif args.task == 2:
                 task2_path = os.path.join(args.save_dir, f"LAB5_{wandb.run.id}_task2_pong.pt")
                 torch.save(torch.load(best_model_src), task2_path)
                 print(f"Saved BEST model snapshot for Task 2: {task2_path}")
            # Task 3 relies on step-based snapshots saved during training loop
        else:
            print("Warning: best_model.pt not found for final task-specific saving.")

        wandb.finish()
        if hasattr(agent, 'env'): agent.env.close()
        if hasattr(agent, 'test_env'): agent.test_env.close()