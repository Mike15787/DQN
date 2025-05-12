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
import time # For timing evaluations

gym.register_envs(ale_py)

def init_weights(m):
    """Initialize weights using Kaiming uniform for Conv and Linear layers."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
    Q-network: Uses MLP for low-dimensional states (CartPole) or CNN for Atari.
    The actual network structure is determined in the DQNAgent initialization.
    """
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        # Placeholder: The actual network is built based on input_shape
        self.network = self._build_network()

    def _build_network(self):
        if len(self.input_shape) == 1: # Assuming 1D state (e.g., CartPole)
            # Simple MLP for CartPole
            return nn.Sequential(
                nn.Linear(self.input_shape[0], 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, self.num_actions)
            )
        elif len(self.input_shape) == 3: # Assuming 3D state (e.g., Atari frames CxHxW)
             # CNN for Atari (Nature DQN architecture)
            # Input shape: (batch_size, 4, 84, 84)
            return nn.Sequential(
                nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512), nn.ReLU(), # Calculate input features for Linear layer
                nn.Linear(512, self.num_actions)
            )
        else:
            raise ValueError("Unsupported input shape")

    def forward(self, x):
        # Normalize Atari frames if needed
        if len(x.shape) > 2 and x.dtype == torch.uint8: # Check if it looks like image data
             x = x.float() / 255.0
        elif len(x.shape) > 2 and x.dtype == torch.float32 and x.max() > 1.1: # Check if float but not normalized
             print("Warning: Input tensor seems to be float but not normalized (max > 1.1). Normalizing...")
             x = x / 255.0
        return self.network(x)

class AtariPreprocessor:
    """
    Preprocesses Atari frames: grayscale + resize + stack frame_stack frames.
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque([], maxlen=frame_stack) # Start empty

    def preprocess(self, obs):
        """Converts an observation to grayscale and resizes it."""
        if obs.ndim == 3: # Check if it's an RGB frame
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        elif obs.ndim == 2: # Already grayscale
            gray = obs
        else:
            raise ValueError("Unexpected observation shape")
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        # Add channel dimension: (84, 84) -> (1, 84, 84) - although stacking handles this implicitly later
        return resized # shape (84, 84)

    def reset(self, obs):
        """Resets the processor and fills the frame deque with the first frame."""
        frame = self.preprocess(obs)
        # Fill the deque with the first frame replicated frame_stack times
        self.frames.extend([frame] * self.frame_stack)
        return self._get_stacked_frames()

    def step(self, obs):
        """Processes a new frame, adds it to the deque, and returns the stacked frames."""
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return self._get_stacked_frames()

    def _get_stacked_frames(self):
        """Stacks frames in the deque along the first dimension."""
        assert len(self.frames) == self.frame_stack, "Frame buffer not full!"
        # Stack along a new dimension (channel dimension for Conv2d)
        # Output shape: (frame_stack, 84, 84)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER).
    Reference: https://arxiv.org/abs/1511.05952
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha # Controls prioritization level (0=uniform, 1=full priority)
        self.beta_start = beta_start # Initial IS weight correction strength
        self.beta_frames = beta_frames # Steps over which beta anneals to 1.0
        self.frame = 1 # Counter for annealing beta

        self.buffer = [None] * capacity
        self.priorities = np.zeros((capacity,), dtype=np.float64)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0 # Initialize max priority

    def add(self, transition):
        """Adds a new transition with maximum priority."""
        self.buffer[self.pos] = transition
        self.priorities[self.pos] = self.max_priority # Use max priority for new samples
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_beta(self):
        """Anneals beta from beta_start to 1.0 over beta_frames steps."""
        beta = self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames
        self.frame += 1
        return min(1.0, beta)

    def sample(self, batch_size):
        """Samples a batch of transitions based on priorities."""
        if self.size == 0:
            return [], [], [] # Return empty if buffer is empty

        # Get priorities of stored transitions
        priorities = self.priorities[:self.size]
        scaled_priorities = priorities ** self.alpha
        probs = scaled_priorities / scaled_priorities.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(self.size, batch_size, p=probs, replace=True)
        samples = [self.buffer[i] for i in indices]

        # Calculate Importance Sampling (IS) weights
        beta = self._get_beta()
        weights = (self.size * probs[indices]) ** (-beta)
        # Normalize weights for stability
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32) # Ensure float32 for PyTorch

        return indices, samples, weights

    def update_priorities(self, indices, errors):
        """Updates priorities of sampled transitions based on their TD errors."""
        errors = np.abs(errors) + 1e-6 # Add epsilon to avoid zero priority
        clipped_errors = np.clip(errors, 0, self.max_priority) # Clip errors for stability if needed
        self.priorities[indices] = clipped_errors ** self.alpha
        # Update max priority seen so far
        self.max_priority = max(self.max_priority, clipped_errors.max())

    def __len__(self):
        return self.size

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        # environments
        print(f"Initializing environment: {env_name}")
        self.env = gym.make(env_name, render_mode="rgb_array" if not args.render else "human")
        self.test_env = gym.make(env_name, render_mode="rgb_array") # Always use rgb_array for evaluation consistency
        self.is_atari = 'ALE/' in env_name or 'Pong' in env_name # More robust check for Atari envs
        self.num_actions = self.env.action_space.n
        self.state_shape = self.env.observation_space.shape

        # preprocessor for Atari
        self.preprocessor = AtariPreprocessor(frame_stack=args.frame_stack) if self.is_atari else None
        if self.is_atari:
            # Determine the input shape for the network after preprocessing
            # Reset env to get an initial observation
            obs, _ = self.env.reset()
            processed_obs = self.preprocessor.reset(obs)
            self.state_shape = processed_obs.shape # Should be (frame_stack, 84, 84)
            print(f"Atari detected. State shape after preprocessing: {self.state_shape}")
        else:
            print(f"Non-Atari env. State shape: {self.state_shape}")


        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # hyperparams from args
        self.args = args
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.use_ddqn = args.use_ddqn
        self.use_per = args.use_per
        self.use_multistep = args.use_multistep
        self.n_step = args.n_step if self.use_multistep else 1
        self.target_update_frequency = args.target_update_frequency
        self.replay_start_size = args.replay_start_size
        self.max_episode_steps = args.max_episode_steps
        self.train_per_env_step = args.train_per_env_step # Renamed for clarity
        self.num_eval_episodes = args.num_eval_episodes
        self.gradient_clip_norm = args.gradient_clip_norm
        self.save_interval = args.save_interval # For Task 3 saving

        # networks
        print(f"Building networks for state shape {self.state_shape} and {self.num_actions} actions.")
        self.q_net = DQN(self.state_shape, self.num_actions).to(self.device)
        self.target_net = DQN(self.state_shape, self.num_actions).to(self.device)
        self.q_net.apply(init_weights) # Apply initialization
        self.target_net.load_state_dict(self.q_net.state_dict()) # Sync initially
        self.target_net.eval() # Target network is only for inference
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=1.5e-4 if self.is_atari else 1e-8) # Use AdamW or specific eps for Atari? Adam default eps=1e-8

        # replay buffers
        if self.use_per:
            print("Using Prioritized Replay Buffer")
            self.memory = PrioritizedReplayBuffer(args.memory_size, alpha=args.per_alpha, beta_start=args.per_beta_start, beta_frames=args.per_beta_frames)
        else:
            print("Using Standard Replay Buffer (deque)")
            self.memory = deque(maxlen=args.memory_size)
        # n-step buffer (temporary storage for multi-step calculation)
        self.n_step_buffer = deque(maxlen=self.n_step)

        # tracking variables
        self.env_step_count = 0
        self.train_step_count = 0
        self.best_eval_reward = float('-inf')
        self.episode_count = 0

        # saving directory
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Models and logs will be saved in: {self.save_dir}")

    def select_action(self, state, evaluation=False):
        """Selects action using epsilon-greedy policy."""
        # Use epsilon_min during evaluation
        epsilon_threshold = self.epsilon_min if evaluation else self.epsilon
        if random.random() < epsilon_threshold:
            return self.env.action_space.sample() # Explore
        else:
            # Exploit
            state_tensor = self._to_tensor(state)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax().item()

    def _to_tensor(self, state):
        """Converts state (numpy array) to tensor and moves to device."""
        # Convert state to float32 BEFORE creating tensor if it's not already
        if not isinstance(state, np.ndarray):
             state = np.array(state) # Ensure it's a numpy array

        if state.dtype != np.float32 and not self.is_atari: # Only convert non-Atari to float32 here
             state = state.astype(np.float32)

        # Atari frames are handled during forward pass (normalization)
        # Add batch dimension
        tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        return tensor

    def _preprocess_state(self, obs, reset=False):
        """Applies Atari preprocessing if needed."""
        if self.is_atari:
            if reset:
                return self.preprocessor.reset(obs)
            else:
                return self.preprocessor.step(obs)
        else:
            return obs # Return observation directly if not Atari

    def _store_transition(self, transition):
        """Stores transition in the appropriate replay buffer."""
        if self.use_per:
            self.memory.add(transition) # PER handles its own priority initialization
        else:
            self.memory.append(transition)

    def _process_n_step_buffer(self, force_flush=False):
        """Processes the n-step buffer to calculate and store transitions."""
        if not self.use_multistep:
            # If not using multi-step, store the single transition immediately
            # (This case is handled directly in the run loop now)
             return

        # Process only when buffer is full or when flushing at episode end
        while len(self.n_step_buffer) >= self.n_step or (force_flush and len(self.n_step_buffer) > 0):
            if len(self.n_step_buffer) < self.n_step and not force_flush:
                break # Need more steps unless flushing

            # Calculate n-step return
            current_n = len(self.n_step_buffer) if force_flush else self.n_step
            R = 0.0
            for i in range(current_n):
                 s, a, r, _, _ = self.n_step_buffer[i]
                 R += (self.gamma**i) * r

            # Get s_0, a_0 from the start and s_n, done_n from the end
            s0, a0, _, _, _ = self.n_step_buffer[0]
            sn, an, rn, next_sn, done_n = self.n_step_buffer[current_n-1] # Get the last actual state and done flag

            # Store the n-step transition (s0, a0, n_step_reward, sn, done_n)
            n_step_transition = (s0, a0, R, next_sn, done_n) # Use next_sn as the final state
            self._store_transition(n_step_transition)

            # Remove the first element from the buffer to slide the window
            self.n_step_buffer.popleft()

            # If flushing, continue until buffer is empty
            if not force_flush:
                 break # Only process one transition if not flushing


    def run(self, total_env_steps=5_000_000):
        """Main training loop."""
        obs, info = self.env.reset()
        state = self._preprocess_state(obs, reset=True)
        episode_reward = 0
        episode_steps = 0

        while self.env_step_count < total_env_steps:
            action = self.select_action(state)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            next_state = self._preprocess_state(next_obs) # Process next observation

            episode_reward += reward
            episode_steps += 1
            self.env_step_count += 1

            # Store experience for n-step calculation or single step
            transition_data = (state, action, reward, next_state, done)
            if self.use_multistep:
                self.n_step_buffer.append(transition_data)
                self._process_n_step_buffer() # Process if buffer is full
            else:
                # Store single-step transition directly if not using multi-step
                self._store_transition(transition_data)

            state = next_state

            # --- Training Phase ---
            # Start training only after buffer is sufficiently filled
            can_train = (len(self.memory) >= self.replay_start_size)
            if can_train:
                 # Perform multiple training steps per environment step if configured
                 for _ in range(self.train_per_env_step):
                     self.train()
                 # Epsilon decay based on environment steps after replay buffer is ready
                 if self.epsilon > self.epsilon_min:
                     self.epsilon -= (self.args.epsilon_start - self.epsilon_min) / self.args.epsilon_decay_steps
                     #self.epsilon *= self.epsilon_decay # Alternative: Multiplicative decay

            # --- End of Episode Handling ---
            if done or episode_steps >= self.max_episode_steps:
                self.episode_count += 1
                print(f"Step: {self.env_step_count}, Episode: {self.episode_count}, Reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.3f}")

                if self.use_multistep:
                    # Flush remaining transitions in n-step buffer at episode end
                    self._process_n_step_buffer(force_flush=True)

                log_data = {
                    "Episode": self.episode_count,
                    "Reward/Train": episode_reward,
                    "Epsilon": self.epsilon,
                    "Buffer Size": len(self.memory),
                    "Env Step": self.env_step_count # Log env step count for x-axis
                }


                # --- Evaluation Phase ---
                if self.episode_count % self.args.eval_frequency == 0 and can_train:
                    eval_start_time = time.time()
                    avg_eval_reward = self.evaluate()
                    eval_duration = time.time() - eval_start_time
                    print(f"--- Evaluation Result (Ep {self.episode_count}, Step {self.env_step_count}) ---")
                    print(f"Avg Reward over {self.num_eval_episodes} episodes: {avg_eval_reward:.2f}")
                    print(f"Evaluation Duration: {eval_duration:.2f} seconds")
                    print("--------------------------------------------------")
                    log_data["Reward/Eval"] = avg_eval_reward

                    # Save best model based on evaluation reward
                    if avg_eval_reward > self.best_eval_reward:
                        self.best_eval_reward = avg_eval_reward
                        best_model_path = os.path.join(self.save_dir, f"{self.args.wandb_run_name}_best.pt")
                        torch.save(self.q_net.state_dict(), best_model_path)
                        print(f"*** New best model saved with reward {avg_eval_reward:.2f} to {best_model_path} ***")

                # --- Task 3 Interval Saving ---
                if self.args.task == "Task3" and can_train and self.env_step_count // self.save_interval > (self.env_step_count - episode_steps) // self.save_interval:
                     # Save if we crossed a save_interval threshold within this episode
                     current_interval = (self.env_step_count // self.save_interval) * self.save_interval
                     if current_interval > 0: # Don't save at step 0
                        save_path = os.path.join(self.save_dir, f"{self.args.wandb_run_name}_{current_interval}.pt")
                        torch.save(self.q_net.state_dict(), save_path)
                        print(f"--- Task 3 model snapshot saved at step {current_interval} to {save_path} ---")

                # Log to WandB
                if wandb.run:
                     wandb.log(log_data, step=self.env_step_count)


                # Reset for next episode
                obs, info = self.env.reset()
                state = self._preprocess_state(obs, reset=True)
                self.n_step_buffer.clear() # Clear n-step buffer for new episode
                episode_reward = 0
                episode_steps = 0

        print("Training finished.")
        self.env.close()
        self.test_env.close()


    def evaluate(self):
        """Evaluates the agent's performance over several episodes."""
        total_rewards = []
        original_epsilon = self.epsilon # Store current epsilon
        self.epsilon = self.epsilon_min # Use minimal exploration for evaluation

        print(f"Starting evaluation for {self.num_eval_episodes} episodes...")
        for i in range(self.num_eval_episodes):
            obs, info = self.test_env.reset()
            state = self._preprocess_state(obs, reset=True)
            episode_reward = 0
            done = False
            step = 0
            while not done and step < self.max_episode_steps:
                 # Use evaluation=True in select_action if it behaves differently
                 action = self.select_action(state, evaluation=True)
                 next_obs, reward, terminated, truncated, info = self.test_env.step(action)
                 done = terminated or truncated
                 state = self._preprocess_state(next_obs)
                 episode_reward += reward
                 step += 1
            total_rewards.append(episode_reward)
            # print(f"Eval episode {i+1}/{self.num_eval_episodes} finished with reward {episode_reward}")

        self.epsilon = original_epsilon # Restore original epsilon
        return np.mean(total_rewards)


    def train(self):
        """Performs a single training step."""
        # Ensure buffer has enough samples
        if len(self.memory) < self.batch_size:
             return # Should ideally be checked against replay_start_size before calling train

        # Sample batch
        if self.use_per:
            indices, batch, weights = self.memory.sample(self.batch_size)
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device).unsqueeze(1) # Add feature dim
        else:
            batch = random.sample(self.memory, self.batch_size)
            indices = None # No indices needed for standard replay
            weights = torch.ones(self.batch_size, 1, dtype=torch.float32).to(self.device) # Uniform weights

        # Unpack batch data
        # Note: Need to handle potential numpy arrays in the batch before converting to tensor
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors efficiently
        # Use np.stack for potentially multi-dimensional states (like Atari frames)
        state_np = np.stack(states, axis=0)
        next_state_np = np.stack(next_states, axis=0)

        st = torch.tensor(state_np, dtype=torch.float32 if not self.is_atari else torch.uint8).to(self.device) # Atari uint8 for memory efficiency
        nxt = torch.tensor(next_state_np, dtype=torch.float32 if not self.is_atari else torch.uint8).to(self.device)
        # Actions, rewards, dones are typically simpler
        a = torch.tensor(actions, dtype=torch.int64).to(self.device).unsqueeze(1) # Shape (batch_size, 1)
        r = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1) # Shape (batch_size, 1)
        d = torch.tensor(dones, dtype=torch.float32).to(self.device).unsqueeze(1)   # Shape (batch_size, 1)

        # --- Calculate Current Q Values ---
        # Q(s, a) for the actions taken
        current_q_values = self.q_net(st).gather(1, a) # Shape (batch_size, 1)

        # --- Calculate Target Q Values ---
        with torch.no_grad():
            if self.use_ddqn:
                # Select best actions a' using the online network Q(s', a'; θ)
                next_q_values_online = self.q_net(nxt) # Shape (batch_size, num_actions)
                best_actions = next_q_values_online.argmax(dim=1, keepdim=True) # Shape (batch_size, 1)
                # Evaluate these actions a' using the target network Q(s', a'; θ-)
                next_q_values_target = self.target_net(nxt).gather(1, best_actions) # Shape (batch_size, 1)
            else:
                # Select and evaluate using the target network: max_{a'} Q(s', a'; θ-)
                next_q_values_target = self.target_net(nxt).max(dim=1, keepdim=True)[0] # Shape (batch_size, 1)

            # Calculate the target: R + γ^n * Q_target(s', a') * (1 - done)
            # Note: If using multi-step, 'r' already contains the n-step return 'R'.
            # The discount factor needs to be gamma^n_step.
            effective_gamma = self.gamma ** self.n_step
            target_q_values = r + effective_gamma * next_q_values_target * (1 - d) # Shape (batch_size, 1)

        # --- Calculate Loss ---
        td_error = target_q_values - current_q_values # Shape (batch_size, 1)

        # Apply Importance Sampling weights for PER
        loss = (weights * td_error.pow(2)).mean()

        # --- Optimization Step ---
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.gradient_clip_norm)
        self.optimizer.step()

        self.train_step_count += 1

        # --- Update PER Priorities ---
        if self.use_per and indices is not None:
            errors_numpy = td_error.abs().squeeze().detach().cpu().numpy()
            self.memory.update_priorities(indices, errors_numpy)

        # --- Update Target Network ---
        if self.train_step_count % self.target_update_frequency == 0:
            print(f"--- Updating target network at train step {self.train_step_count} ---")
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Log loss (optional, can be frequent)
        if wandb.run and self.train_step_count % 100 == 0: # Log loss every 100 train steps
            wandb.log({"Loss": loss.item(), "Train Step": self.train_step_count}, step=self.env_step_count) # Log against env_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Task and General Config
    parser.add_argument("--task", type=str, choices=["Task1", "Task2", "Task3"], default="Task1", help="Select the task to run (Task1: CartPole, Task2: Pong Vanilla, Task3: Pong Enhanced)")
    parser.add_argument("--save-dir", type=str, default="./results", help="Directory to save models and logs")
    parser.add_argument("--wandb-project-name", type=str, default="DLP-Lab5-DQN", help="WandB project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="WandB run name (defaults to task name)")
    parser.add_argument("--render", action="store_true", help="Render the environment during training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total-env-steps", type=int, default=None, help="Total environment steps to train for (overrides default)")

    # Core DQN Hyperparameters (provide defaults, allow override)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate") # Default set based on task
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training") # Default set based on task
    parser.add_argument("--memory-size", type=int, default=None, help="Replay buffer size") # Default set based on task
    parser.add_argument("--discount-factor", "--gamma", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument("--target-update-frequency", type=int, default=None, help="Frequency (train steps) to update target network") # Default set based on task
    parser.add_argument("--replay-start-size", type=int, default=None, help="Number of steps to fill buffer before training starts") # Default set based on task

    # Epsilon Greedy Strategy
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon value")
    parser.add_argument("--epsilon-min", type=float, default=None, help="Minimum epsilon value") # Default set based on task
    parser.add_argument("--epsilon-decay-steps", type=int, default=None, help="Number of env steps over which epsilon decays linearly") # Default set based on task
    # parser.add_argument("--epsilon-decay", type=float, default=0.99999, help="Multiplicative epsilon decay factor (alternative)") # Less common for large step counts

    # Enhancements (can be forced on/off, defaults set by task)
    parser.add_argument("--use-ddqn", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable Double DQN")
    parser.add_argument("--use-per", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable Prioritized Experience Replay")
    parser.add_argument("--use-multistep", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable Multi-step returns")
    parser.add_argument("--n-step", type=int, default=3, help="Number of steps for Multi-step return")

    # PER Specific Hyperparameters
    parser.add_argument("--per-alpha", type=float, default=0.6, help="Alpha parameter for PER")
    parser.add_argument("--per-beta-start", type=float, default=0.4, help="Initial beta parameter for PER IS weights")
    parser.add_argument("--per-beta-frames", type=int, default=None, help="Number of env steps to anneal beta to 1.0") # Default set based on task

    # Training Loop Hyperparameters
    parser.add_argument("--max-episode-steps", type=int, default=None, help="Max steps per episode") # Default set based on task
    parser.add_argument("--train-per-env-step", type=int, default=None, help="Number of training steps per environment step") # Default set based on task
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0, help="Max norm for gradient clipping (0 to disable)")

    # Evaluation Hyperparameters
    parser.add_argument("--eval-frequency", type=int, default=None, help="Frequency (episodes) to run evaluation") # Default set based on task
    parser.add_argument("--num-eval-episodes", type=int, default=20, help="Number of episodes to average over during evaluation")

    # Atari Specific
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of frames to stack for Atari")

    # Task 3 Specific
    parser.add_argument("--save-interval", type=int, default=200000, help="Frequency (env steps) to save model snapshots for Task 3")

    args = parser.parse_args()

    # --- Set Task-Specific Defaults ---
    env_name = ""
    if args.task == "Task1":
        env_name = "CartPole-v1"
        # CartPole defaults (trains fast)
        if args.lr is None: args.lr = 1e-3
        if args.batch_size is None: args.batch_size = 64
        if args.memory_size is None: args.memory_size = 10_000
        if args.target_update_frequency is None: args.target_update_frequency = 500 # Train steps
        if args.replay_start_size is None: args.replay_start_size = 1_000
        if args.epsilon_min is None: args.epsilon_min = 0.01
        if args.epsilon_decay_steps is None: args.epsilon_decay_steps = 5_000 # Env steps
        if args.max_episode_steps is None: args.max_episode_steps = 500 # CartPole specific limit
        if args.train_per_env_step is None: args.train_per_env_step = 1
        if args.eval_frequency is None: args.eval_frequency = 20 # Episodes
        if args.total_env_steps is None: args.total_env_steps = 50_000
        # Enhancements off by default for Task 1
        if args.use_ddqn is None: args.use_ddqn = False
        if args.use_per is None: args.use_per = False
        if args.use_multistep is None: args.use_multistep = False
        if args.wandb_run_name is None: args.wandb_run_name = "dqn_cartpole_vanilla"

    elif args.task == "Task2":
        env_name = "ALE/Pong-v5"
        # Pong Vanilla defaults (Nature DQN settings adjusted)
        if args.lr is None: args.lr = 1e-4 # Lower LR for Atari
        if args.batch_size is None: args.batch_size = 32
        if args.memory_size is None: args.memory_size = 100_000 # Smaller than Rainbow, per assignment suggestion range
        if args.target_update_frequency is None: args.target_update_frequency = 1000 # Train steps (Nature DQN uses 10k env steps)
        if args.replay_start_size is None: args.replay_start_size = 10_000 # Start training earlier than Nature DQN's 50k
        if args.epsilon_min is None: args.epsilon_min = 0.1 # Higher min epsilon for Atari? Nature uses 0.1
        if args.epsilon_decay_steps is None: args.epsilon_decay_steps = 1_000_000 # Env steps (Nature DQN)
        if args.max_episode_steps is None: args.max_episode_steps = 10000 # Limit episode length?
        if args.train_per_env_step is None: args.train_per_env_step = 1 # Train once per env step (can be adjusted e.g., train every 4 steps)
        if args.eval_frequency is None: args.eval_frequency = 50 # Episodes (Evaluation takes longer)
        if args.total_env_steps is None: args.total_env_steps = 2_000_000 # Train longer for Atari
        # Enhancements off by default for Task 2
        if args.use_ddqn is None: args.use_ddqn = False
        if args.use_per is None: args.use_per = False
        if args.use_multistep is None: args.use_multistep = False
        if args.wandb_run_name is None: args.wandb_run_name = "dqn_pong_vanilla"

    elif args.task == "Task3":
        env_name = "ALE/Pong-v5"
        # Pong Enhanced defaults (Rainbow-like settings, but maybe less memory)
        if args.lr is None: args.lr = 6.25e-5 # Often lower for Rainbow components
        if args.batch_size is None: args.batch_size = 32
        if args.memory_size is None: args.memory_size = 100_000 # Assignment range, Rainbow uses 1M
        if args.target_update_frequency is None: args.target_update_frequency = 2000 # Slower updates might be stable (Rainbow uses 8k env steps)
        if args.replay_start_size is None: args.replay_start_size = 10_000 # Start training earlier
        if args.epsilon_min is None: args.epsilon_min = 0.01 # Often lower min epsilon with PER/DDQN
        if args.epsilon_decay_steps is None: args.epsilon_decay_steps = 250_000 # Decay faster if learning faster
        if args.max_episode_steps is None: args.max_episode_steps = 10000
        if args.train_per_env_step is None: args.train_per_env_step = 1
        if args.eval_frequency is None: args.eval_frequency = 50
        if args.total_env_steps is None: args.total_env_steps = 1_000_000 # Aim for 1M steps per requirement
        # Enhancements ON by default for Task 3
        if args.use_ddqn is None: args.use_ddqn = True
        if args.use_per is None: args.use_per = True
        if args.use_multistep is None: args.use_multistep = True
        if args.per_beta_frames is None: args.per_beta_frames = args.total_env_steps # Anneal beta over the whole training
        if args.wandb_run_name is None: args.wandb_run_name = f"dqn_pong_enhanced_ddqn{args.use_ddqn}_per{args.use_per}_multi{args.use_multistep}_n{args.n_step}"

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Potentially disable benchmarking for reproducibility if needed
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True

    # Initialize WandB
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=args, save_code=True)
    wandb.config.update({"environment_name": env_name}) # Add env_name to config

    # Create and run the agent
    agent = DQNAgent(env_name, args)
    agent.run(total_env_steps=args.total_env_steps)

    wandb.finish()
    print("Run finished.")