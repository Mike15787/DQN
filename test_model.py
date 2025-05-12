import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse

# --- Reusable Classes (Copied/Adapted from dqn.py) ---

class DQN(nn.Module):
    """
    Q-network: Builds MLP for CartPole or CNN for Atari based on input_shape.
    """
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.network = self._build_network()

    def _build_network(self):
        if len(self.input_shape) == 1: # CartPole state (1D)
            print(f"Building MLP for input shape {self.input_shape}")
            return nn.Sequential(
                nn.Linear(self.input_shape[0], 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, self.num_actions)
            )
        elif len(self.input_shape) == 3: # Atari state (CxHxW)
            print(f"Building CNN for input shape {self.input_shape}")
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
            raise ValueError(f"Unsupported input shape: {self.input_shape}")

    def forward(self, x):
        # Normalize Atari frames (uint8 -> float / 255.0)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.network(x)

class AtariPreprocessor:
    """
    Preprocesses Atari frames: grayscale + resize + stack frames.
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque([], maxlen=frame_stack)

    def preprocess(self, obs):
        """Converts an RGB observation to grayscale and resizes it."""
        # Assumes obs is RGB (ndim=3) from ALE
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized # shape (84, 84)

    def reset(self, obs):
        """Resets and fills the frame deque with the first frame."""
        frame = self.preprocess(obs)
        self.frames.extend([frame] * self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        """Processes a new frame and returns the stacked frames."""
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

# --- Evaluation Function ---

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Task Specific Setup ---
    is_atari = False
    preprocessor = None
    input_shape = None
    if args.task == "Task1":
        env_name = "CartPole-v1"
        input_shape = (4,) # CartPole state shape
    elif args.task in ["Task2", "Task3"]:
        env_name = "ALE/Pong-v5"
        is_atari = True
        # Frame stack is usually 4 for Atari DQN
        frame_stack = 4 # You might need to make this an arg if you changed it in training
        preprocessor = AtariPreprocessor(frame_stack=frame_stack)
        input_shape = (frame_stack, 84, 84) # Input shape for CNN
    else:
        raise ValueError(f"Invalid task: {args.task}")

    print(f"Setting up environment: {env_name}")
    # Use "human" render mode if you want to see the window, "rgb_array" for saving videos
    render_mode = "human" if args.render else "rgb_array"
    env = gym.make(env_name, render_mode=render_mode)

    # Seeding
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # env.action_space.seed(seed) # Recommended way for Gymnasium >= 0.26
    # env.observation_space.seed(seed)

    num_actions = env.action_space.n

    # --- Load Model ---
    print(f"Loading model for input shape {input_shape} and {num_actions} actions.")
    model = DQN(input_shape, num_actions).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded model from: {args.model_path}")
    except Exception as e:
        print(f"Error loading model from {args.model_path}: {e}")
        print("Ensure the model architecture in test_model.py matches the saved model.")
        env.close()
        return
    model.eval() # Set model to evaluation mode

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Helper Functions ---
    def preprocess_state(obs, reset=False):
        if is_atari and preprocessor:
            return preprocessor.reset(obs) if reset else preprocessor.step(obs)
        else:
            return obs # CartPole uses raw observation

    def state_to_tensor(state):
        """Converts state to tensor, handles dtype."""
        if not isinstance(state, np.ndarray):
             state = np.array(state)
        # Use uint8 for Atari frames to match training buffer, float32 for CartPole
        dtype = torch.uint8 if is_atari else torch.float32
        if not is_atari and state.dtype != np.float32:
            state = state.astype(np.float32)

        return torch.from_numpy(state).to(dtype).unsqueeze(0).to(device)

    # --- Evaluation Loop ---
    print(f"Starting evaluation for {args.episodes} episodes...")
    all_rewards = []
    for ep in range(args.episodes):
        # Seed the environment reset for consistent evaluation runs
        obs, _ = env.reset(seed=seed + ep)
        state = preprocess_state(obs, reset=True)
        done = False
        total_reward = 0
        frames = []

        while not done:
            # Render frame BEFORE taking action (for CartPole, state changes after action)
            if render_mode == "rgb_array":
                frame = env.render()
                frames.append(frame)
            elif render_mode == "human":
                env.render() # Let gym handle human rendering timing

            state_tensor = state_to_tensor(state)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocess_state(next_obs) # Preprocess the *next* observation

        all_rewards.append(total_reward)
        print(f"Episode {ep+1}/{args.episodes} finished. Reward: {total_reward}")

        # Save video if using rgb_array
        if render_mode == "rgb_array" and frames:
            out_path = os.path.join(args.output_dir, f"eval_task{args.task}_ep{ep+1}_reward{total_reward:.0f}.mp4")
            try:
                with imageio.get_writer(out_path, fps=30, macro_block_size=1) as video: # Adjust fps/macro_block_size if needed
                    for f in frames:
                        video.append_data(f)
                print(f"Saved episode video -> {out_path}")
            except Exception as e:
                print(f"Error saving video for episode {ep+1}: {e}")
        elif render_mode == "rgb_array" and not frames:
             print(f"Warning: No frames recorded for episode {ep+1}, cannot save video.")


    env.close()
    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    print("\n--- Evaluation Summary ---")
    print(f"Task: {args.task}")
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Average Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
    print("--------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained DQN models for Task 1, 2, or 3.")
    parser.add_argument("--task", type=str, choices=["Task1", "Task2", "Task3"], required=True,
                        help="Specify the task the model was trained for (Task1: CartPole, Task2/3: Pong)")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained .pt model file")
    parser.add_argument("--output-dir", type=str, default="./eval_videos",
                        help="Directory to save evaluation videos")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes to run for evaluation")
    parser.add_argument("--seed", type=int, default=42, # Use a different seed than training potentially
                        help="Random seed for evaluation reproducibility")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment in a window (human mode) instead of saving videos (rgb_array mode)")

    args = parser.parse_args()

    # Ensure ALE is registered if needed (might be redundant if run after training script)
    try:
        gym.register_envs(ale_py)
        print("ALE environments registered.")
    except Exception as e:
        print(f"Could not register ALE environments (maybe already done): {e}")


    evaluate(args)