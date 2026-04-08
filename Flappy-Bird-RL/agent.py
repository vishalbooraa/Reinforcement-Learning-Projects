import flappy_bird_gymnasium
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim 
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import os
import argparse

if torch.backends.mps.is_available():
    device="mps"
elif torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"


RUNS_DIR="runs"
os.makedirs(RUNS_DIR,exist_ok=True)

class Agent:
    def __init__(self,param_set):
        self.param_set = param_set

        with open("parameters.yaml","r") as f:
            all_param_set = yaml.safe_load(f)
            params=all_param_set[param_set]

        self.alpha= params["alpha"]
        self.gamma=params["gamma"]

        self.epsilon_init=params["epsilon_init"]
        self.epsilon_min=params["epsilon_min"]
        self.epsilon_decay=params["epsilon_decay"]

        self.replay_memory_size=params["replay_memory_size"]
        self.mini_batch_size=params["mini_batch_size"]

        self.reward_threshold=params["reward_threshold"]
        self.network_sync_rate=params["network_sync_rate"]


        self.loss_fn= nn.MSELoss()
        self.optimizer=None

        self.LOG_FILE= os.path.join(RUNS_DIR,f"{self.param_set}.log")
        self.MODEL_FILE= os.path.join(RUNS_DIR,f"{self.param_set}.pt")


    def run(self, is_training=True, render=False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

            steps = 0
            best_reward = float("-inf")

        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            episode_rewards = 0
            terminated = False

            while not terminated and episode_rewards < self.reward_threshold:

                if is_training and random.random() < epsilon:
                    action = torch.tensor(env.action_space.sample(), device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).argmax()

                next_state, reward, terminated, _, _ = env.step(action.item())

                next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                reward = torch.clamp(reward, -1, 1)

                episode_rewards += reward.item()

                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    steps += 1

                state = next_state

            # ✅ PRINT INSIDE LOOP
            print(f"Episode={episode+1}, Reward={episode_rewards}, Epsilon={epsilon}")
            
            with open(self.LOG_FILE, "a") as f:
                f.write(f"{episode+1},{episode_rewards},{epsilon}\n")

            # ✅ TRAINING PART
            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                if episode_rewards > best_reward:
                    best_reward = episode_rewards
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0
    
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)
        
        with torch.no_grad():
            target_q_values = rewards + self.gamma * target_dqn(next_states).max(1)[0] * (1 - terminations)

        current_q_values = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("hyperparameters",help="")
    parser.add_argument("--train",help="Training mode",action="store_true")
    args=parser.parse_args()
    dql=Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False,render=True)
