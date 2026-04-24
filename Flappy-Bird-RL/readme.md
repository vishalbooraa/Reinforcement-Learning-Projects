# 🎮 Flappy Bird RL with DQN

## 📌 Overview

This project implements a **Deep Q-Network (DQN)** agent to play the **Flappy Bird game** using **Reinforcement Learning**.
The agent learns to navigate the bird through pipes by training on the **FlappyBird-v0 environment** from the `flappy_bird_gymnasium` package.

---

## 🚀 Features

* **DQN Implementation:** Neural network with a hidden layer for Q-value approximation
* **Experience Replay:** Stores and samples past experiences for stable training
* **Epsilon-Greedy Exploration:** Balances exploration vs exploitation
* **Target Network:** Improves training stability with periodic updates
* **Hyperparameter Tuning:** Configurable via `parameters.yaml`

---

## 📂 Project Structure

```bash
.
├── agent.py                # Main agent (training + testing)
├── dqn.py                  # Neural network model
├── experience_replay.py    # Replay buffer implementation
├── parameters.yaml         # Hyperparameter configurations
├── runs/                   # Logs and trained models
└── README.md
```

---

## ⚙️ Dependencies

* Python 3.7+
* PyTorch (`torch`)
* gymnasium
* flappy_bird_gymnasium
* pyyaml
* argparse *(built-in)*

### 📦 Installation

```bash
pip install torch gymnasium flappy_bird_gymnasium pyyaml
```

---

## 🧪 Usage

### ▶️ Training the Agent

```bash
python agent.py flappybirdv0 --train
```

* Starts training using the `flappybirdv0` configuration
* Logs saved in: `runs/flappybirdv0.log`
* Best model saved as: `flappybirdv0.pt`

---

### ▶️ Testing the Agent

```bash
python agent.py flappybirdv0
```

* Loads trained model
* Runs agent in **render mode** (watch gameplay)

---

## ⚙️ Hyperparameters

Defined in `parameters.yaml`.

### 🔹 Example: `flappybirdv0`

* **epsilon_init:** 1.0 *(initial exploration)*
* **epsilon_min:** 0.01 *(minimum exploration)*
* **epsilon_decay:** 0.999995 *(decay rate)*
* **replay_memory_size:** 100000
* **mini_batch_size:** 32
* **network_sync_rate:** 10 *(target update frequency)*
* **alpha:** 0.00025 *(learning rate)*
* **gamma:** 0.99 *(discount factor)*
* **reward_threshold:** 1000

You can create new configurations in the YAML file and pass them as arguments.

---

## 🧠 How It Works

1. **Environment Interaction:**
   Agent observes state and receives rewards

2. **Action Selection:**
   Uses **epsilon-greedy policy** *(flap / no flap)*

3. **Experience Storage:**
   Stores `(state, action, reward, next_state)` in replay memory

4. **Training:**
   Samples mini-batches to update Q-network

5. **Target Network:**
   Stabilizes learning by updating periodically

---

## 📊 Results

* Agent learns to **avoid pipes and achieve higher scores**
* Training progress tracked via:

  * Episode rewards
  * Epsilon decay
* Logs available in `runs/` directory

---

## 🤝 Contributing

Contributions are welcome!

* Experiment with different hyperparameters
* Try new architectures (CNN, deeper networks)
* Improve DQN (Double DQN, Dueling DQN, etc.)

---

## 🛡️ License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Vishal Boora**
B.Tech IT (2026) | AI/ML & Full Stack Developer

---
