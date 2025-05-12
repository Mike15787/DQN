# --- START OF MODIFIED FILE dqn.py ---

# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh
# Implementation by: AI Assistant based on instructions

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Import F for loss functions
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque, namedtuple
import wandb
import argparse
import time

# Register Atari environments
try:
    gym.register_envs(ale_py)
except ImportError:
    print("ale_py not found or registration failed. Atari envs might not work.")
except Exception as e:
    print(f"Error registering ale_py envs: {e}")


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5), nonlinearity='relu') # Kaiming for ReLU
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound) # Default Pytorch Linear init for bias


class DQN(nn.Module):
    """
        Deep Q Network model. Handles both FCN (CartPole) and CNN (Atari).
    """
    def __init__(self, input_shape, num_actions, is_atari=False):
        super(DQN, self).__init__()
        self.is_atari = is_atari

        if self.is_atari:
            # CNN for Atari (Based on test_model.py and Nature DQN paper)
            # Input shape: (C, H, W) e.g. (4, 84, 84)
            input_channels = input_shape[0]
            self.network = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                # Calculate the flattened size dynamically
                # Need a dummy forward pass to determine size, or calculate manually
                # Manual calculation:
                # Conv1 out: floor(((84 - 8) / 4) + 1) = floor(19 + 1) = 20
                # Conv2 out: floor(((20 - 4) / 2) + 1) = floor(8 + 1) = 9
                # Conv3 out: floor(((9 - 3) / 1) + 1) = floor(6 + 1) = 7
                # Flattened size: 64 * 7 * 7 = 3136
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )
        else:
            # FCN for CartPole
            # Input shape: (State Dim,) e.g. (4,)
            input_dim = input_shape[0]
            # Simple FCN architecture (adjustable)
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128), # Increased size
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions)
            )

    def forward(self, x):
        if self.is_atari:
            # Normalize pixel values for Atari
            x = x / 255.0
        return self.network(x)


class AtariPreprocessor:
    """
        Preprocesses the state input of DQN for Atari (Grayscale, Resize, Stack frames)
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        # Ensure obs is uint8
        if obs.dtype != np.uint8:
            obs = obs.astype(np.uint8)

        if len(obs.shape) == 3 and obs.shape[2] == 3: # Check if RGB
             gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        elif len(obs.shape) == 2: # Already grayscale
             gray = obs
        else:
            raise ValueError(f"Unexpected observation shape: {obs.shape}")

        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized # Shape (84, 84)

    def reset(self, obs):
        """Resets the processor and fills the frame deque with the first observation."""
        frame = self.preprocess(obs)
        # Initialize the deque with the first frame replicated frame_stack times
        self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack)
        # Stack along the first axis (channel axis for PyTorch Conv2d NCHW)
        return np.stack(self.frames, axis=0) # Shape (4, 84, 84)

    def step(self, obs):
        """Processes a new observation, adds it to the deque, and returns the stacked frames."""
        frame = self.preprocess(obs)
        self.frames.append(frame)
        # Stack along the first axis
        return np.stack(self.frames, axis=0) # Shape (4, 84, 84)

# Define a named tuple for transitions
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))
NStepTransition = namedtuple('NStepTransition',
                        ('state_t', 'action_t', 'n_step_reward', 'next_state_tn', 'done_tn'))


class ReplayBuffer:
    """Standard Replay Buffer using deque."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class PrioritizedReplayBuffer:
    """
        Prioritized Experience Replay (PER) buffer.
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=1000000):
        self.capacity = capacity
        self.alpha = alpha # Priority exponent
        self.beta_start = beta_start # Initial IS weight exponent
        self.beta = beta_start
        self.beta_frames = beta_frames # Steps over which beta anneals to 1.0
        self.frame_idx = 0 # Keep track of frames for beta annealing
        self.buffer = [None] * capacity # Store transitions
        self.priorities = np.zeros((capacity,), dtype=np.float64) # Store priorities (use float64 for stability)
        self.pos = 0 # Current position to insert next transition
        self.size = 0 # Current number of transitions in buffer
        self.max_priority = 1.0 # Initial max priority

        # Epsilon added to priorities to ensure non-zero probability
        self.eps = 1e-6

    def push(self, *args):
        """Save a transition. Use max priority for new transitions."""
        ########## YOUR CODE HERE (Task 3: PER add) ##########
        priority = self.max_priority # Give new samples max priority initially
        self.buffer[self.pos] = Transition(*args)
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        ########## END OF YOUR CODE ##########

    def _anneal_beta(self):
        """Anneal beta from beta_start to 1.0 over beta_frames steps."""
        fraction = min(self.frame_idx / self.beta_frames, 1.0)
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)
        self.frame_idx += 1

    def sample(self, batch_size):
        """Sample a batch of transitions based on priorities."""
        if self.size == 0:
            return [], [], [] # Return empty lists if buffer is empty

        ########## YOUR CODE HERE (Task 3: PER sample) ##########
        # Get priorities for sampling (only for filled slots)
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(self.size, batch_size, p=probs, replace=True) # Allow replacement

        # Calculate Importance Sampling (IS) weights
        self._anneal_beta() # Update beta before calculating weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        # Normalize weights for stability (max weight is 1)
        weights /= weights.max()

        # Retrieve transitions and convert to batch format
        batch = [self.buffer[idx] for idx in indices]
        ########## END OF YOUR CODE ##########

        return batch, indices, np.array(weights, dtype=np.float32) # Return weights as numpy array

    def update_priorities(self, indices, errors):
        """Update priorities of sampled transitions."""
        ########## YOUR CODE HERE (Task 3: PER update) ##########
        # Ensure errors are positive and add epsilon
        errors = np.abs(errors) + self.eps
        # Clip priorities? Original paper does not, Rainbow does. Let's not clip for now.
        self.priorities[indices] = errors # Update priorities
        # Update max priority seen so far
        self.max_priority = max(self.max_priority, errors.max())
        ########## END OF YOUR CODE ##########

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env_name = env_name
        self.is_atari = "ALE/" in env_name
        self.args = args

        self.env = gym.make(env_name, render_mode="rgb_array" if args.render else None)
        # Separate test env to ensure consistent evaluation
        self.test_env = gym.make(env_name, render_mode="rgb_array")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Determine state shape and number of actions
        if self.is_atari:
            self.preprocessor = AtariPreprocessor(frame_stack=4)
            # Reset env to get initial observation shape
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs) # State shape (4, 84, 84)
            self.state_shape = state.shape
            print(f"Atari State Shape: {self.state_shape}")
        else:
            self.preprocessor = None # No preprocessing for CartPole
            self.state_shape = self.env.observation_space.shape # e.g., (4,)
            print(f"Classic Control State Shape: {self.state_shape}")

        self.num_actions = self.env.action_space.n
        print(f"Number of Actions: {self.num_actions}")

        # Initialize Networks
        self.q_net = DQN(self.state_shape, self.num_actions, is_atari=self.is_atari).to(self.device)
        self.target_net = DQN(self.state_shape, self.num_actions, is_atari=self.is_atari).to(self.device)
        self.q_net.apply(init_weights) # Apply custom weight initialization
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval() # Target network is only for inference
        print("Networks initialized:")
        print("Q-Network:", self.q_net)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=1.5e-4 if self.is_atari else 1e-8) # Use Adam, smaller eps for Atari stability

        # Hyperparameters
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_min # Renamed from epsilon_min for clarity
        self.epsilon_decay_frames = args.epsilon_decay_frames # Total frames for decay
        self.epsilon = self.epsilon_start # Current epsilon

        # Replay Buffer
        self.use_per = args.use_per
        if self.use_per:
            print("Using Prioritized Experience Replay (PER)")
            beta_frames = args.total_env_steps # Anneal beta over total training steps
            self.memory = PrioritizedReplayBuffer(args.memory_size, alpha=args.per_alpha, beta_start=args.per_beta, beta_frames=beta_frames)
        else:
            print("Using Standard Experience Replay")
            self.memory = ReplayBuffer(args.memory_size)

        # Multi-step Learning
        self.n_steps = args.n_steps
        if self.n_steps > 1:
            print(f"Using {self.n_steps}-step returns")
            self.n_step_buffer = deque(maxlen=self.n_steps)
            self.gamma_n = self.gamma ** self.n_steps # Precompute n-step gamma

        # Double DQN
        self.use_ddqn = args.use_ddqn
        if self.use_ddqn:
            print("Using Double DQN (DDQN)")

        # Training Control
        self.env_count = 0 # Total steps taken in the environment
        self.train_count = 0 # Total training updates performed
        self.target_update_frequency = args.target_update_frequency # Steps between target net updates
        self.train_frequency = args.train_frequency # Environment steps per training update
        self.replay_start_size = args.replay_start_size # Min buffer size before training
        self.max_episode_steps = args.max_episode_steps # Max steps per episode
        self.total_env_steps = args.total_env_steps # Total steps for training

        # Evaluation and Saving
        self.eval_frequency = args.eval_frequency # Env steps between evaluations
        self.save_frequency = args.save_frequency # Env steps between model saves
        self.best_eval_reward = -float('inf') # Initialize with negative infinity
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_frequency = 1000 # How often to print console logs

    def _get_state(self, obs):
        """Get the current state from observation (preprocess if needed)."""
        if self.preprocessor:
            return self.preprocessor.step(obs)
        else:
            return obs

    def _reset_state(self, obs):
        """Reset the state (preprocess if needed)."""
        if self.preprocessor:
            return self.preprocessor.reset(obs)
        else:
            return obs # For CartPole, obs is the state

    def _epsilon_decay(self):
        """Linearly decay epsilon from start to end over decay_frames."""
        fraction = min(1.0, self.env_count / self.epsilon_decay_frames)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state, evaluation=False):
        """Selects an action using epsilon-greedy policy during training, or greedy during evaluation."""
        if not evaluation and random.random() < self.epsilon:
            return self.env.action_space.sample() # Explore
        else:
            # Exploit
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax().item()

    def _process_n_step_buffer(self):
        """Calculates the n-step return and pushes the transition to the main buffer."""
        if len(self.n_step_buffer) < self.n_steps:
            return # Not enough transitions yet

        # Get the relevant transitions
        first_trans = self.n_step_buffer[0] # (s_t, a_t, r_t, next_s_{t+1}, done_t)
        last_trans = self.n_step_buffer[-1] # (s_{t+n-1}, a_{t+n-1}, r_{t+n-1}, next_s_{t+n}, done_{t+n-1})

        # Calculate n-step reward
        n_step_reward = 0.0
        for i in range(self.n_steps):
            r = self.n_step_buffer[i].reward
            n_step_reward += (self.gamma ** i) * r
            # If an episode finished mid-sequence, the effective n is shorter
            if self.n_step_buffer[i].done:
                # This transition marks the end, future rewards are 0.
                # The effective 'n' stops here for reward calculation.
                # The state `next_s_{t+n}` should still be the state after the i-th step.
                last_trans = self.n_step_buffer[i] # Use the state where done occurred
                self.gamma_n = self.gamma ** (i + 1) # Adjust gamma power
                break # Stop accumulating reward
        else:
             # Reset gamma_n if loop completed normally (no early done)
             self.gamma_n = self.gamma ** self.n_steps


        # Extract components for the n-step transition
        state_t = first_trans.state
        action_t = first_trans.action
        next_state_tn = last_trans.next_state # This is s_{t+n}
        done_tn = last_trans.done # Whether the episode ended *at* step t+n-1

        # Create the NStepTransition tuple (or adapt Transition tuple if needed)
        # Using a standard Transition tuple works fine if we understand the fields:
        # state = state_t
        # action = action_t
        # reward = n_step_reward
        # next_state = next_state_tn
        # done = done_tn

        # Push to the main replay buffer
        self.memory.push(state_t, action_t, n_step_reward, next_state_tn, done_tn)

        # The n_step_buffer automatically discards the oldest when full


    def run(self):
        """Main training loop."""
        start_time = time.time()
        ep = 0
        total_reward_window = deque(maxlen=100) # For logging average reward

        while self.env_count < self.total_env_steps:
            ep += 1
            obs, info = self.env.reset()
            state = self._reset_state(obs)
            done = False
            truncated = False
            ep_reward = 0
            ep_steps = 0

            # Clear n-step buffer at the start of each episode
            if self.n_steps > 1:
                self.n_step_buffer.clear()

            while not (done or truncated):
                # Check termination condition
                if self.env_count >= self.total_env_steps:
                    break

                # Select Action
                action = self.select_action(state)

                # Environment Step
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated # Combine termination conditions
                next_state = self._get_state(next_obs)
                ep_reward += reward
                ep_steps += 1
                self.env_count += 1

                # Store transition (handle n-step or 1-step)
                # We always store the 1-step transition temporarily
                temp_transition = Transition(state, action, reward, next_state, done)

                if self.n_steps > 1:
                    self.n_step_buffer.append(temp_transition)
                    # Process the buffer if it's full or episode ended
                    # Note: The logic processes the transition starting `n` steps ago.
                    if len(self.n_step_buffer) == self.n_steps:
                         self._process_n_step_buffer()
                else:
                    # 1-step: push directly to memory
                    self.memory.push(state, action, reward, next_state, done)

                state = next_state

                # Decay Epsilon
                self._epsilon_decay()

                # Train the network
                if self.env_count >= self.replay_start_size and self.env_count % self.train_frequency == 0:
                    self.train() # Call train method

                # Update Target Network
                if self.env_count % self.target_update_frequency == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())
                    # print(f"Step {self.env_count}: Target network updated.")

                 # Evaluate Model
                if self.env_count % self.eval_frequency == 0:
                    eval_reward = self.evaluate()
                    is_best = eval_reward > self.best_eval_reward
                    if is_best:
                        self.best_eval_reward = eval_reward
                        model_path = os.path.join(self.save_dir, "best_model.pt")
                        torch.save(self.q_net.state_dict(), model_path)
                        print(f"Step {self.env_count}: New best model saved with reward {eval_reward:.2f}")
                    print(f"[Eval] Step: {self.env_count} | Eval Reward: {eval_reward:.2f} {'(New Best)' if is_best else ''}")
                    wandb.log({
                        "Eval/Reward": eval_reward,
                        "Eval/Best Reward": self.best_eval_reward,
                        "Step": self.env_count,
                        "Episode": ep,
                    })

                # Save Model Periodically
                if self.env_count % self.save_frequency == 0:
                     model_path = os.path.join(self.save_dir, f"model_step{self.env_count}.pt")
                     torch.save(self.q_net.state_dict(), model_path)
                     print(f"Step {self.env_count}: Model checkpoint saved to {model_path}")

                     # Specific saves for Task 3 grading
                     if self.is_atari and self.env_count in [200000, 400000, 600000, 800000, 1000000]:
                         task3_save_path = os.path.join(self.save_dir, f"LAB5_{wandb.run.id}_task3_pong{self.env_count}.pt")
                         torch.save(self.q_net.state_dict(), task3_save_path)
                         print(f"Step {self.env_count}: Task 3 model saved to {task3_save_path}")


                # Log Progress (less frequently)
                if self.env_count % self.log_frequency == 0:
                    elapsed_time = time.time() - start_time
                    steps_per_sec = self.log_frequency / elapsed_time if elapsed_time > 0 else 0
                    avg_reward = np.mean(total_reward_window) if len(total_reward_window) > 0 else 0.0
                    print(f"[Progress] Step: {self.env_count}/{self.total_env_steps} | Ep: {ep} | Eps: {self.epsilon:.4f} | Avg Reward (100ep): {avg_reward:.2f} | Steps/sec: {steps_per_sec:.2f}")
                    wandb.log({
                        "Progress/Epsilon": self.epsilon,
                        "Progress/Steps Per Second": steps_per_sec,
                        "Progress/Updates": self.train_count,
                        "Step": self.env_count,
                        "Episode": ep,
                    })
                    start_time = time.time() # Reset timer for next interval

            # End of episode
            total_reward_window.append(ep_reward)
            # If n-step, flush remaining transitions from buffer
            if self.n_steps > 1:
                 while len(self.n_step_buffer) > 0:
                     self._process_n_step_buffer()
                     # Need to remove the oldest element manually now if not using maxlen side effect
                     if len(self.n_step_buffer) > 0: # Check required after pop
                         self.n_step_buffer.popleft()


            # Log episode reward
            # print(f"[Episode End] Ep: {ep} | Reward: {ep_reward} | Steps: {ep_steps} | Total Steps: {self.env_count}")
            wandb.log({
                "Train/Episode Reward": ep_reward,
                "Train/Episode Length": ep_steps,
                "Step": self.env_count,
                "Episode": ep
            })

        print("Training finished.")
        self.env.close()
        self.test_env.close()


    def evaluate(self, num_episodes=10): # Evaluate over 10 episodes for stability
        """Evaluates the agent's performance."""
        print(f"\n--- Starting Evaluation ({num_episodes} episodes) ---")
        total_rewards = []
        for ep in range(num_episodes):
            obs, _ = self.test_env.reset()
            state = self._reset_state(obs)
            done = False
            truncated = False
            ep_reward = 0
            while not (done or truncated):
                action = self.select_action(state, evaluation=True) # Use greedy policy
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                ep_reward += reward
                state = self._get_state(next_obs) # Use _get_state for consistency
            total_rewards.append(ep_reward)
            # print(f"Eval Ep {ep+1}/{num_episodes} Reward: {ep_reward}")

        avg_reward = np.mean(total_rewards)
        print(f"--- Evaluation Complete --- Avg Reward: {avg_reward:.2f}\n")
        return avg_reward


    def train(self):
        """Performs a single training step."""
        if len(self.memory) < self.replay_start_size:
            return # Not enough samples yet

        self.train_count += 1

        # Sample a mini-batch
        if self.use_per:
            transitions, indices, weights = self.memory.sample(self.batch_size)
            weights = torch.from_numpy(weights).float().to(self.device) # Convert weights to tensor
        else:
            transitions = self.memory.sample(self.batch_size)
            indices = None # No indices needed for standard buffer
            weights = torch.ones(self.batch_size, device=self.device) # Uniform weights

        # Unpack batch
        batch = Transition(*zip(*transitions)) # Converts list of Transitions to Transition of lists/tuples

        # Convert batch elements to tensors
        # Need special handling for states and next_states due to potential stacking
        states = torch.from_numpy(np.array(batch.state)).float().to(self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64).to(self.device).unsqueeze(1) # Shape (B, 1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_states = torch.from_numpy(np.array(batch.next_state)).float().to(self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32).to(self.device) # 1.0 if done, 0.0 otherwise

        # --- Calculate Q-values for current states ---
        # q_net(states) gives Q-values for all actions: shape (B, num_actions)
        # gather(1, actions) selects the Q-value corresponding to the action taken: shape (B, 1)
        # squeeze(1) removes the trailing dimension: shape (B,)
        current_q_values = self.q_net(states).gather(1, actions).squeeze(1)

        # --- Calculate target Q-values ---
        with torch.no_grad(): # No gradients needed for target calculation
            # Get Q-values for next states from target network
            next_q_values_target = self.target_net(next_states)

            if self.use_ddqn:
                # Double DQN: Select best action using online network, evaluate with target network
                # 1. Find best actions in next_states using the *online* network
                best_next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True) # Shape (B, 1)
                # 2. Get Q-values for these actions using the *target* network
                next_q_vals = next_q_values_target.gather(1, best_next_actions).squeeze(1) # Shape (B,)
            else:
                # Standard DQN: Select max Q-value directly from target network
                next_q_vals = next_q_values_target.max(dim=1)[0] # Shape (B,)

            # Calculate target: R + gamma * Q_target(s', argmax_a' Q(s', a')) (or R + gamma * Q_target(s', max_a' Q_target(s', a')))
            # Use gamma_n if n_steps > 1, otherwise use gamma
            current_gamma = self.gamma_n if self.n_steps > 1 else self.gamma
            target_q_values = rewards + (current_gamma * next_q_vals * (1 - dones)) # (1 - dones) ensures target is just reward if done

        # --- Calculate Loss ---
        # Huber loss is often more robust than MSE for DQN
        loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none') # Calculate element-wise loss

        # Apply Importance Sampling weights if using PER
        loss = (loss * weights).mean() # Weighted average

        # --- Update Priorities (if using PER) ---
        if self.use_per:
            # Calculate TD errors (absolute difference) for priority update
            td_errors = (target_q_values - current_q_values).abs().detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors)

        # --- Optimize the Model ---
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (optional but often helpful)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # --- Logging (optional, less frequent) ---
        if self.train_count % 1000 == 0:
            wandb.log({
                "Train/Loss": loss.item(),
                "Train/Mean Q-value": current_q_values.mean().item(),
                "Step": self.env_count,
                "Update Count": self.train_count
            })
            # print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {current_q_values.mean().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Environment Args
    parser.add_argument("--env-name", type=str, default="CartPole-v1", help="Environment name (e.g., CartPole-v1, ALE/Pong-v5)")
    parser.add_argument("--render", action="store_true", help="Render the environment during training")

    # Training Control Args
    parser.add_argument("--total-env-steps", type=int, default=500000, help="Total environment steps to train for") # Adjusted default for CartPole
    parser.add_argument("--train-frequency", type=int, default=4, help="Steps between training updates") # Train every 4 steps (common for Atari)
    parser.add_argument("--replay-start-size", type=int, default=10000, help="Minimum replay buffer size before training starts") # Adjusted default
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--target-update-frequency", type=int, default=1000, help="Steps between target network updates") # Adjusted default
    parser.add_argument("--max-episode-steps", type=int, default=10000, help="Maximum steps per episode (mostly relevant for Atari)")

    # Hyperparameter Args
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer") # 1e-4 often good for Adam
    parser.add_argument("--discount-factor", "--gamma", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument("--memory-size", type=int, default=100000, help="Capacity of the replay buffer") # 100k reasonable default

    # Epsilon Greedy Args
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Starting value for epsilon")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum value for epsilon") # Lower min epsilon
    parser.add_argument("--epsilon-decay-frames", type=int, default=100000, help="Number of frames over which epsilon decays linearly") # Linear decay

    # Enhancements (Task 3) Args
    parser.add_argument("--use-ddqn", action="store_true", help="Enable Double DQN")
    parser.add_argument("--use-per", action="store_true", help="Enable Prioritized Experience Replay")
    parser.add_argument("--per-alpha", type=float, default=0.6, help="Alpha parameter for PER")
    parser.add_argument("--per-beta", type=float, default=0.4, help="Starting Beta parameter for PER")
    parser.add_argument("--n-steps", type=int, default=1, help="Enable N-step returns (set > 1)")

    # Logging and Saving Args
    parser.add_argument("--wandb-project", type=str, default="DLP-Lab5-DQN", help="WandB project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="WandB run name (defaults to env_name-config)")
    parser.add_argument("--save-dir", type=str, default="./results", help="Directory to save models and logs")
    parser.add_argument("--eval-frequency", type=int, default=10000, help="Frequency (in env steps) to evaluate the model") # Evaluate every 10k steps
    parser.add_argument("--save-frequency", type=int, default=50000, help="Frequency (in env steps) to save model checkpoints") # Save every 50k steps
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # --- Adjust Defaults Based on Env ---
    is_atari = "ALE/" in args.env_name
    if is_atari:
        # Pong defaults (adjust as needed based on PDF/experience)
        print("Applying Atari specific default adjustments...")
        # Training time
        if args.total_env_steps == 500000: args.total_env_steps = 1000000 # Longer training for Pong
        # Buffer / Start Size
        if args.memory_size == 100000: args.memory_size = 100000 # Pong often uses ~1M, but 100k can work
        if args.replay_start_size == 10000: args.replay_start_size = 50000 # Larger start size for Pong
        # Epsilon Decay
        if args.epsilon_decay_frames == 100000: args.epsilon_decay_frames = 1000000 # Slower decay for Pong
        if args.epsilon_min == 0.01: args.epsilon_min = 0.1 # Nature DQN used 0.1 final epsilon
        # Target update
        if args.target_update_frequency == 1000: args.target_update_frequency = 10000 # Slower target updates for Pong
        # Learning Rate (sometimes lower for Atari)
        if args.lr == 1e-4: args.lr = 1e-4 # Keep 1e-4 or try 2.5e-4 if using RMSprop like original paper
        # N-step default (Rainbow used n=3)
        # if args.n_steps == 1: args.n_steps = 3 # Optional: default to 3-step for Atari if enhanced
        # Batch Size (often larger for Atari)
        if args.batch_size == 32: args.batch_size = 32 # 32 is standard, 64 sometimes used
        # Save / Eval Freq
        if args.eval_frequency == 10000: args.eval_frequency = 25000
        if args.save_frequency == 50000: args.save_frequency = 100000
    else:
        # CartPole defaults (adjust as needed)
        if args.total_env_steps == 500000: args.total_env_steps = 50000 # CartPole trains faster
        if args.memory_size == 100000: args.memory_size = 10000
        if args.replay_start_size == 10000: args.replay_start_size = 1000
        if args.epsilon_decay_frames == 100000: args.epsilon_decay_frames = 5000
        if args.epsilon_min == 0.01: args.epsilon_min = 0.02
        if args.target_update_frequency == 1000: args.target_update_frequency = 500
        if args.lr == 1e-4: args.lr = 5e-4 # CartPole can use higher LR
        if args.eval_frequency == 10000: args.eval_frequency = 1000
        if args.save_frequency == 50000: args.save_frequency = 5000

    # --- Seed Everything ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Seeding action space requires the env instance
    temp_env = gym.make(args.env_name)
    temp_env.action_space.seed(args.seed)
    temp_env.observation_space.seed(args.seed)
    temp_env.close()
    del temp_env
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Potentially disable non-deterministic algorithms for reproducibility
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


    # --- WandB Setup ---
    run_name = args.wandb_run_name or f"{args.env_name}-ddqn{args.use_ddqn}-per{args.use_per}-n{args.n_steps}"
    wandb.init(project=args.wandb_project, name=run_name, config=args, save_code=True)

    # --- Create and Run Agent ---
    agent = DQNAgent(env_name=args.env_name, args=args)
    agent.run()

    wandb.finish()
    print("Run finished.")

# --- END OF MODIFIED FILE dqn.py ---