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
    Q-network: Uses MLP for CartPole or CNN for Atari Pong.
    """
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.network = self._build_network()

    def _build_network(self):
        if len(self.input_shape) == 1: # CartPole state (1D)
            return nn.Sequential(
                nn.Linear(self.input_shape[0], 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, self.num_actions)
            )
        elif len(self.input_shape) == 3: # Atari state (CxHxW)
            # Input shape: (batch_size, 4, 84, 84)
            return nn.Sequential(
                nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
                nn.Linear(512, self.num_actions)
            )
        else:
            # This path shouldn't be reached for CartPole or Pong
            raise ValueError(f"Unsupported input shape: {self.input_shape}")

    def forward(self, x):
        # Normalize Atari frames (uint8 -> float / 255.0)
        # CartPole inputs are expected to be float32 already
        if x.dtype == torch.uint8: # Check if it's Atari frame data
            x = x.float() / 255.0
        # Removed the check for unnormalized float32 inputs for simplicity
        return self.network(x)

class AtariPreprocessor:
    """
    Preprocesses Atari frames: grayscale + resize + stack frame_stack frames.
    Assumes input 'obs' is an RGB frame from ALE.
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque([], maxlen=frame_stack)

    def preprocess(self, obs):
        """Converts an RGB observation to grayscale and resizes it."""
        # Simplified: Assumes obs is always RGB (ndim=3) from ALE
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized # shape (84, 84)

    def reset(self, obs):
        """Resets the processor and fills the frame deque with the first frame."""
        frame = self.preprocess(obs)
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
        # Output shape: (frame_stack, 84, 84)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER).
    Reference: https://arxiv.org/abs/1511.05952
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.buffer = [None] * capacity
        self.priorities = np.zeros((capacity,), dtype=np.float64)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

    def add(self, transition):
        """Adds a new transition with maximum priority."""
        self.buffer[self.pos] = transition
        self.priorities[self.pos] = self.max_priority
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
            return [], [], []

        priorities = self.priorities[:self.size]
        scaled_priorities = priorities ** self.alpha
        probs = scaled_priorities / scaled_priorities.sum()

        indices = np.random.choice(self.size, batch_size, p=probs, replace=True)
        samples = [self.buffer[i] for i in indices]

        beta = self._get_beta()
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max() # Normalize weights
        weights = np.array(weights, dtype=np.float32)

        return indices, samples, weights

    def update_priorities(self, indices, errors):
        """Updates priorities of sampled transitions based on their TD errors."""
        errors = np.abs(errors) + 1e-6 # Epsilon for non-zero priority
        clipped_errors = np.clip(errors, 0, self.max_priority) # Clipping might not be strictly needed but can help stability
        self.priorities[indices] = clipped_errors ** self.alpha
        self.max_priority = max(self.max_priority, clipped_errors.max())

    def __len__(self):
        return self.size

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        # environments
        print(f"Initializing environment: {env_name}")
        self.env = gym.make(env_name, render_mode="rgb_array" if not args.render else "human")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.is_atari = 'ALE/' in env_name or 'Pong' in env_name # Check for Atari
        self.num_actions = self.env.action_space.n
        env_state_shape = self.env.observation_space.shape # Shape before preprocessing

        # preprocessor for Atari
        self.preprocessor = AtariPreprocessor(frame_stack=args.frame_stack) if self.is_atari else None
        if self.is_atari:
            obs, _ = self.env.reset()
            processed_obs = self.preprocessor.reset(obs)
            self.state_shape = processed_obs.shape # Actual shape used by network (e.g., (4, 84, 84))
            print(f"Atari detected. State shape after preprocessing: {self.state_shape}")
        else:
            self.state_shape = env_state_shape # CartPole uses original shape (e.g., (4,))
            print(f"Non-Atari env. State shape: {self.state_shape}")

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # hyperparams from args
        self.args = args # Store args for easy access
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_min = args.epsilon_min
        self.use_ddqn = args.use_ddqn
        self.use_per = args.use_per
        self.use_multistep = args.use_multistep
        self.n_step = args.n_step if self.use_multistep else 1
        self.target_update_frequency = args.target_update_frequency
        self.replay_start_size = args.replay_start_size
        self.max_episode_steps = args.max_episode_steps
        self.train_per_env_step = args.train_per_env_step
        self.num_eval_episodes = args.num_eval_episodes
        self.gradient_clip_norm = args.gradient_clip_norm
        self.save_interval = args.save_interval

        # networks
        print(f"Building networks for state shape {self.state_shape} and {self.num_actions} actions.")
        self.q_net = DQN(self.state_shape, self.num_actions).to(self.device)
        self.target_net = DQN(self.state_shape, self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=1.5e-4 if self.is_atari else 1e-8)

        # replay buffers
        if self.use_per:
            print("Using Prioritized Replay Buffer")
            self.memory = PrioritizedReplayBuffer(args.memory_size, alpha=args.per_alpha, beta_start=args.per_beta_start, beta_frames=args.per_beta_frames)
        else:
            print("Using Standard Replay Buffer (deque)")
            self.memory = deque(maxlen=args.memory_size)
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
        epsilon_threshold = self.epsilon_min if evaluation else self.epsilon
        if random.random() < epsilon_threshold:
            return self.env.action_space.sample()
        else:
            state_tensor = self._to_tensor(state)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax().item()

    def _to_tensor(self, state):
        """Converts state (numpy array) to tensor and moves to device."""
        # Assumes input `state` is already the correct type (float32 for CartPole, uint8 for Atari preprocessed)
        if not isinstance(state, np.ndarray):
             state = np.array(state) # Ensure numpy array for from_numpy

        # CartPole state needs conversion if it wasn't already float32
        if not self.is_atari and state.dtype != np.float32:
             state = state.astype(np.float32)

        tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        return tensor

    def _preprocess_state(self, obs, reset=False):
        """Applies Atari preprocessing if needed."""
        if self.is_atari:
            return self.preprocessor.reset(obs) if reset else self.preprocessor.step(obs)
        else:
            return obs # CartPole uses raw observation

    def _store_transition(self, transition):
        """Stores transition in the appropriate replay buffer."""
        if self.use_per:
            self.memory.add(transition)
        else:
            self.memory.append(transition)

    def _process_n_step_buffer(self, force_flush=False):
        """Processes the n-step buffer to calculate and store transitions."""
        if not self.use_multistep:
             return

        while len(self.n_step_buffer) >= self.n_step or (force_flush and len(self.n_step_buffer) > 0):
            if len(self.n_step_buffer) < self.n_step and not force_flush:
                break

            current_n = len(self.n_step_buffer) if force_flush else self.n_step
            R = sum([(self.gamma**i) * self.n_step_buffer[i][2] for i in range(current_n)]) # Sum discounted rewards

            s0, a0, _, _, _ = self.n_step_buffer[0]
            _, _, _, next_sn, done_n = self.n_step_buffer[current_n-1] # Get final next_state and done

            n_step_transition = (s0, a0, R, next_sn, done_n)
            self._store_transition(n_step_transition)

            self.n_step_buffer.popleft()
            if not force_flush:
                 break


    def run(self, total_env_steps):
        """Main training loop."""
        obs, info = self.env.reset()
        state = self._preprocess_state(obs, reset=True)
        episode_reward = 0
        episode_steps = 0

        while self.env_step_count < total_env_steps:
            action = self.select_action(state)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            next_state = self._preprocess_state(next_obs)

            episode_reward += reward
            episode_steps += 1
            self.env_step_count += 1

            # Store experience
            transition_data = (state, action, reward, next_state, done)
            if self.use_multistep:
                self.n_step_buffer.append(transition_data)
                self._process_n_step_buffer()
            else:
                self._store_transition(transition_data)

            state = next_state

            # --- Training ---
            can_train = (len(self.memory) >= self.replay_start_size)
            if can_train:
                 for _ in range(self.train_per_env_step):
                     if len(self.memory) >= self.batch_size: # Ensure enough samples for a batch
                        self.train()
                 # Linear epsilon decay
                 if self.epsilon > self.epsilon_min:
                     self.epsilon -= (self.args.epsilon_start - self.epsilon_min) / self.args.epsilon_decay_steps
                     self.epsilon = max(self.epsilon_min, self.epsilon) # Ensure not below min

            # --- End of Episode ---
            if done or episode_steps >= self.max_episode_steps:
                self.episode_count += 1
                print(f"Step: {self.env_step_count}, Episode: {self.episode_count}, Reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.3f}")

                if self.use_multistep:
                    self._process_n_step_buffer(force_flush=True) # Flush buffer

                log_data = {
                    "Episode": self.episode_count,
                    "Reward/Train": episode_reward,
                    "Epsilon": self.epsilon,
                    "Buffer Size": len(self.memory),
                    "Env Step": self.env_step_count
                }

                # --- Evaluation ---
                if self.episode_count % self.args.eval_frequency == 0 and can_train:
                    eval_start_time = time.time()
                    avg_eval_reward = self.evaluate()
                    eval_duration = time.time() - eval_start_time
                    print(f"--- Evaluation Result (Ep {self.episode_count}, Step {self.env_step_count}) ---")
                    print(f"Avg Reward over {self.num_eval_episodes} episodes: {avg_eval_reward:.2f} (Duration: {eval_duration:.2f}s)")
                    print("--------------------------------------------------")
                    log_data["Reward/Eval"] = avg_eval_reward

                    # Save best model
                    if avg_eval_reward > self.best_eval_reward:
                        self.best_eval_reward = avg_eval_reward
                        best_model_path = os.path.join(self.save_dir, f"{self.args.wandb_run_name}_best.pt")
                        torch.save(self.q_net.state_dict(), best_model_path)
                        print(f"*** New best model saved: {best_model_path} ***")

                # --- Task 3 Interval Saving ---
                # Check if the current step count crossed a save interval boundary since the last step of the episode
                if self.args.task == "Task3" and can_train and \
                   (self.env_step_count // self.save_interval > (self.env_step_count - episode_steps) // self.save_interval):
                     current_interval_step = (self.env_step_count // self.save_interval) * self.save_interval
                     if current_interval_step > 0:
                        save_path = os.path.join(self.save_dir, f"{self.args.wandb_run_name}_{current_interval_step}.pt")
                        torch.save(self.q_net.state_dict(), save_path)
                        print(f"--- Task 3 model snapshot saved: {save_path} ---")

                # Log to WandB
                if wandb.run:
                     wandb.log(log_data, step=self.env_step_count)

                # Reset episode
                obs, info = self.env.reset()
                state = self._preprocess_state(obs, reset=True)
                self.n_step_buffer.clear()
                episode_reward = 0
                episode_steps = 0

        print("Training finished.")
        self.env.close()
        self.test_env.close()


    def evaluate(self):
        """Evaluates the agent's performance over several episodes."""
        total_rewards = []
        print(f"Starting evaluation for {self.num_eval_episodes} episodes...")
        for i in range(self.num_eval_episodes):
            obs, info = self.test_env.reset()
            state = self._preprocess_state(obs, reset=True)
            episode_reward = 0
            done = False
            step = 0
            while not done and step < self.max_episode_steps:
                 action = self.select_action(state, evaluation=True) # Use greedy policy
                 next_obs, reward, terminated, truncated, info = self.test_env.step(action)
                 done = terminated or truncated
                 state = self._preprocess_state(next_obs)
                 episode_reward += reward
                 step += 1
            total_rewards.append(episode_reward)

        return np.mean(total_rewards)


    def train(self):
        """Performs a single training step."""
        # Sample batch
        if self.use_per:
            indices, batch, weights = self.memory.sample(self.batch_size)
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device).unsqueeze(1)
        else:
            batch = random.sample(self.memory, self.batch_size)
            indices = None
            weights = torch.ones(self.batch_size, 1, dtype=torch.float32).to(self.device)

        # Unpack batch data
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        state_np = np.stack(states, axis=0)
        next_state_np = np.stack(next_states, axis=0)
        # Use uint8 for Atari states on device to save memory, convert in forward pass
        st = torch.tensor(state_np, dtype=torch.uint8 if self.is_atari else torch.float32).to(self.device)
        nxt = torch.tensor(next_state_np, dtype=torch.uint8 if self.is_atari else torch.float32).to(self.device)
        a = torch.tensor(actions, dtype=torch.int64).to(self.device).unsqueeze(1)
        r = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        d = torch.tensor(dones, dtype=torch.float32).to(self.device).unsqueeze(1)

        # --- Current Q Values ---
        current_q_values = self.q_net(st).gather(1, a)

        # --- Target Q Values ---
        with torch.no_grad():
            if self.use_ddqn:
                next_q_values_online = self.q_net(nxt)
                best_actions = next_q_values_online.argmax(dim=1, keepdim=True)
                next_q_values_target = self.target_net(nxt).gather(1, best_actions)
            else:
                next_q_values_target = self.target_net(nxt).max(dim=1, keepdim=True)[0]

            effective_gamma = self.gamma ** self.n_step # Apply n-step discount
            target_q_values = r + effective_gamma * next_q_values_target * (1 - d)

        # --- Loss Calculation ---
        td_error = target_q_values - current_q_values
        loss = (weights * td_error.pow(2)).mean() # Apply IS weights for PER

        # --- Optimization ---
        self.optimizer.zero_grad()
        loss.backward()
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
            # print(f"--- Updating target network at train step {self.train_step_count} ---") # Optional print
            self.target_net.load_state_dict(self.q_net.state_dict())

        # --- Log Loss ---
        if wandb.run and self.train_step_count % 100 == 0:
            wandb.log({"Loss": loss.item(), "Train Step": self.train_step_count}, step=self.env_step_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Agent for CartPole and Atari Pong")
    # Task and General Config
    parser.add_argument("--task", type=str, choices=["Task1", "Task2", "Task3"], required=True, help="Select the task")
    parser.add_argument("--save-dir", type=str, default="./results", help="Directory to save models")
    parser.add_argument("--wandb-project-name", type=str, default="DLP-Lab5-DQN", help="WandB project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="WandB run name (defaults to task name)")
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total-env-steps", type=int, default=None, help="Total environment steps (overrides task defaults)")

    # Core DQN Hyperparameters
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides task defaults)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides task defaults)")
    parser.add_argument("--memory-size", type=int, default=None, help="Replay buffer size (overrides task defaults)")
    parser.add_argument("--discount-factor", "--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--target-update-frequency", type=int, default=None, help="Target network update frequency (train steps, overrides task defaults)")
    parser.add_argument("--replay-start-size", type=int, default=None, help="Min buffer size before training starts (overrides task defaults)")

    # Epsilon Greedy
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-min", type=float, default=None, help="Minimum epsilon (overrides task defaults)")
    parser.add_argument("--epsilon-decay-steps", type=int, default=None, help="Steps for linear epsilon decay (overrides task defaults)")

    # Enhancements
    parser.add_argument("--use-ddqn", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable Double DQN (overrides task defaults)")
    parser.add_argument("--use-per", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable PER (overrides task defaults)")
    parser.add_argument("--use-multistep", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable Multi-step returns (overrides task defaults)")
    parser.add_argument("--n-step", type=int, default=3, help="N-step for Multi-step return")

    # PER Specific
    parser.add_argument("--per-alpha", type=float, default=0.6, help="PER alpha")
    parser.add_argument("--per-beta-start", type=float, default=0.4, help="PER initial beta")
    parser.add_argument("--per-beta-frames", type=int, default=None, help="PER beta annealing steps (overrides task defaults)")

    # Training Loop
    parser.add_argument("--max-episode-steps", type=int, default=None, help="Max steps per episode (overrides task defaults)")
    parser.add_argument("--train-per-env-step", type=int, default=None, help="Train steps per env step (overrides task defaults)")
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0, help="Gradient clipping norm (0 disables)")

    # Evaluation
    parser.add_argument("--eval-frequency", type=int, default=None, help="Evaluation frequency (episodes, overrides task defaults)")
    parser.add_argument("--num-eval-episodes", type=int, default=20, help="Number of episodes for evaluation")

    # Atari Specific
    parser.add_argument("--frame-stack", type=int, default=4, help="Frame stack size for Atari")

    # Task 3 Specific
    parser.add_argument("--save-interval", type=int, default=200000, help="Task 3 model save interval (env steps)")

    args = parser.parse_args()

    # --- Set Task-Specific Defaults ---
    # (Defaults are defined here and can be overridden by command-line args)
    task_defaults = {
        "Task1": {
            "env_name": "CartPole-v1", "lr": 1e-3, "batch_size": 64, "memory_size": 10_000,
            "target_update_frequency": 500, "replay_start_size": 1_000, "epsilon_min": 0.01,
            "epsilon_decay_steps": 10_000, "max_episode_steps": 500, "train_per_env_step": 1,
            "eval_frequency": 20, "total_env_steps": 50_000,
            "use_ddqn": False, "use_per": False, "use_multistep": False,
            "wandb_run_name": "dqn_cartpole_vanilla"
        },
        "Task2": {
            "env_name": "ALE/Pong-v5", "lr": 1e-4, "batch_size": 32, "memory_size": 100_000,
            "target_update_frequency": 1000, "replay_start_size": 10_000, "epsilon_min": 0.1,
            "epsilon_decay_steps": 1_000_000, "max_episode_steps": 10000, "train_per_env_step": 1, # Or maybe train every 4 steps (0.25)?
            "eval_frequency": 50, "total_env_steps": 2_000_000,
            "use_ddqn": False, "use_per": False, "use_multistep": False,
            "per_beta_frames": 1_000_000, # Example: Anneal beta over 1M steps if PER was used
            "wandb_run_name": "dqn_pong_vanilla"
        },
        "Task3": {
            "env_name": "ALE/Pong-v5", "lr": 6.25e-5, "batch_size": 32, "memory_size": 100_000, # Rainbow uses 1M, adjust based on memory
            "target_update_frequency": 2000, "replay_start_size": 10_000, "epsilon_min": 0.01,
            "epsilon_decay_steps": 250_000, "max_episode_steps": 10000, "train_per_env_step": 1,
            "eval_frequency": 50, "total_env_steps": 1_000_000, # Per assignment requirement
            "use_ddqn": True, "use_per": True, "use_multistep": True,
            "per_beta_frames": 1_000_000, # Anneal beta over total steps
            "wandb_run_name": "dqn_pong_enhanced" # Default name, can add details
        }
    }

    defaults = task_defaults[args.task]
    # Apply defaults only if the argument was not provided via command line
    for key, value in defaults.items():
         if getattr(args, key, None) is None:
             setattr(args, key, value)

    # Handle special case for wandb run name default if Task 3 enhancements change
    if args.task == "Task3" and args.wandb_run_name == "dqn_pong_enhanced": # Only update if default name was used
        args.wandb_run_name = f"dqn_pong_enhanced_ddqn{args.use_ddqn}_per{args.use_per}_multi{args.use_multistep}_n{args.n_step}"

    env_name = args.env_name # Get env_name from applied defaults

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed) # Seed all GPUs

    # Initialize WandB
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=vars(args), save_code=True)
    # wandb.config.update({"environment_name": env_name}) # env_name is already in config via vars(args)

    # Create and run the agent
    agent = DQNAgent(env_name, args)
    agent.run(total_env_steps=args.total_env_steps)

    wandb.finish()
    print("Run finished.")