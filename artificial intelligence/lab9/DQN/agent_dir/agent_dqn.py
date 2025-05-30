import os
import random
import copy
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent
from collections import deque # For ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return np.array(states, dtype=np.float32), \
               np.array(actions, dtype=np.int64), \
               np.array(rewards, dtype=np.float32), \
               np.array(next_states, dtype=np.float32), \
               np.array(dones, dtype=np.float32) # Dtype consistency

    def clean(self):
        self.buffer.clear()


class AgentDQN(Agent):
    def __init__(self, env, args):
        super(AgentDQN, self).__init__(env)
        self.env = env
        self.args = args

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.use_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            self.device = torch.device("cuda")
            print(f"Using CUDA on device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.policy_net = QNetwork(self.state_dim, args.hidden_size, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, args.hidden_size, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        self.replay_buffer = ReplayBuffer(args.buffer_size)

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.target_update_interval_d = args.target_update_interval_d
        self.n_frames = args.n_frames # Total training steps for termination

        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay_frames = args.epsilon_decay_frames
        self.current_epsilon = self.epsilon_start

        self.total_steps = 0
        self.episode_count_for_run = 0 # Renamed to avoid conflict if class is instantiated multiple times

        # --- TensorBoard Writer Setup ---
        log_dir_name = f'dqn_{args.env_name}_h{args.hidden_size}_lr{args.lr}_bs{args.batch_size}_seed{args.seed}'
        self.log_dir = Path('runs_cartpole') / log_dir_name # More organized logs
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(logdir=str(self.log_dir))
        print(f"TensorBoard logs will be saved to: {self.log_dir}")

        self.model_save_path = Path('models_cartpole') / log_dir_name
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        print(f"Models will be saved to: {self.model_save_path}")


    def init_game_setting(self):
        pass # For CartPole, usually nothing special here for testing

    def _update_epsilon(self):
        # Linear decay
        fraction = min(1.0, self.total_steps / self.epsilon_decay_frames)
        self.current_epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)
        self.current_epsilon = max(self.epsilon_end, self.current_epsilon) # Ensure it doesn't go below end

    def make_action(self, observation, test=False):
        # Epsilon is updated based on total_steps, which increments during training
        # For pure testing (test=True), we want exploitation
        # For training (test=False), we want epsilon-greedy
        
        current_epsilon_to_use = 0.001 if test else self.current_epsilon # Small epsilon for pure test

        if random.random() > current_epsilon_to_use:
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
        else:
            action = self.env.action_space.sample()
        return action

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q_values = self.policy_net(states_tensor).gather(1, actions_tensor)

        with torch.no_grad():
            next_q_values_target_net = self.target_net(next_states_tensor).max(1)[0].unsqueeze(1)
        
        target_q_values = rewards_tensor + (self.gamma * next_q_values_target_net * (1 - dones_tensor))

        # loss = nn.SmoothL1Loss()(current_q_values, target_q_values) # Huber loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(current_q_values, target_q_values)


        self.optimizer.zero_grad()
        loss.backward()
        if self.args.grad_norm_clip > 0:
             torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.args.grad_norm_clip)
        self.optimizer.step()
        
        return loss.item()

    def run(self):
        print(f"Starting training for {self.n_frames} total steps...")
        all_episode_rewards = [] # Store all episode rewards for averaging
        episode_rewards_window = deque(maxlen=100) # For rolling average of last 100 episodes
        
        best_avg_reward = -float('inf') # To save the best model

        while self.total_steps < self.n_frames:
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0 # Renamed t_step_in_episode to episode_steps
            
            # Update epsilon at the start of each episode based on total_steps progress
            self._update_epsilon()

            # Determine max steps for this episode. CartPole-v0 is 200, v1 is 500.
            max_steps_this_episode = getattr(self.env.spec, 'max_episode_steps', 200)

            for t in range(1, max_steps_this_episode + 1): # Max steps per episode
                action = self.make_action(state, test=False) # Training mode
                next_state, reward, done, _ = self.env.step(action)
                
                # CartPole specific: Adjust reward if needed for faster learning
                # E.g., penalize for 'done' if not at max steps (though standard env already gives +1 per step)
                # if done and t < max_steps_this_episode:
                #     reward = -10 # Example of custom reward, usually not needed for CartPole

                self.replay_buffer.push(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                self.total_steps += 1
                episode_steps = t

                # Train the Q-network (G=1 from algorithm image, i.e., train every step)
                current_loss = self.train()
                if current_loss is not None: # Log loss if training happened
                    self.writer.add_scalar('Loss/td_loss', current_loss, self.total_steps)

                # Update target network (d from algorithm image, now based on total_steps)
                if self.args.target_update_interval_d > 0 and \
                   self.total_steps % self.args.target_update_interval_d == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    # print(f"Step {self.total_steps}: Target network updated.") # Optional: for debugging

                if done or self.total_steps >= self.n_frames:
                    break
            
            # --- End of Episode ---
            self.episode_count_for_run += 1
            all_episode_rewards.append(episode_reward)
            episode_rewards_window.append(episode_reward)
            avg_reward_last_100 = np.mean(episode_rewards_window)

            # TensorBoard Logging
            self.writer.add_scalar('Reward/Episode_Reward', episode_reward, self.episode_count_for_run)
            self.writer.add_scalar('Reward/Avg_Reward_Last_100_Episodes', avg_reward_last_100, self.episode_count_for_run)
            self.writer.add_scalar('Info/Episode_Length', episode_steps, self.episode_count_for_run)
            self.writer.add_scalar('Params/Epsilon', self.current_epsilon, self.total_steps) # Log epsilon against total steps

            # Console Output
            print(f"Episode: {self.episode_count_for_run} | Total Steps: {self.total_steps}/{self.n_frames} | "
                  f"Ep Reward: {episode_reward:.1f} | Avg Reward (100): {avg_reward_last_100:.2f} | "
                  f"Ep Length: {episode_steps} | Epsilon: {self.current_epsilon:.3f}" +
                  (f" | Loss: {current_loss:.4f}" if current_loss is not None else ""))
            
            # Save model if it's the best average reward so far
            if avg_reward_last_100 > best_avg_reward and len(episode_rewards_window) >= 100: # Ensure we have enough episodes for a stable average
                best_avg_reward = avg_reward_last_100
                model_name = f"dqn_{self.args.env_name}_best_avg_reward_{best_avg_reward:.2f}_step{self.total_steps}.pth"
                torch.save(self.policy_net.state_dict(), self.model_save_path / model_name)
                print(f"*** New best average reward: {best_avg_reward:.2f}. Model saved to {self.model_save_path / model_name} ***")

            # Check for convergence goal
            if avg_reward_last_100 >= 195.0 and len(episode_rewards_window) >= 100: # CartPole-v0 solved at 195 over 100 episodes
                 print(f"CartPole-v0 solved in {self.episode_count_for_run} episodes and {self.total_steps} steps!")
                 # Optionally, you can break training early if desired
                 # break 

            if self.total_steps >= self.n_frames:
                print(f"Training finished after {self.total_steps} total steps.")
                break
        
        final_model_name = f"dqn_{self.args.env_name}_final_steps{self.total_steps}.pth"
        torch.save(self.policy_net.state_dict(), self.model_save_path / final_model_name)
        print(f"Final model saved to {self.model_save_path / final_model_name}")

        self.writer.close()
        self.env.close()