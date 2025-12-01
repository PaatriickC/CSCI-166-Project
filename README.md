# 🎮 Deep Q-Networks on Atari Space Invaders
This repository contains my CSCI 166 Deep Reinforcement Learning project, where I implement and compare **Baseline DQN** and **Double DQN** on the Atari game **Space Invaders** using the Gymnasium ALE environment.

Both agents process **RGB stacked frames**, use experience replay, and train inside Google Colab.


# 🚀 **Project Overview**
This project trains an RL agent to play **Space Invaders** using:

# ✔️ Baseline DQN
+ Standard Q-value update
+ Single-network max operator (known to overestimate)
+ Target network synced periodically

# ✔️ Double DQN
+ Online network selects best action
+ Target network evaluates it
+ Reduces overestimation and instability

Both agents share the same:
- Convolutional neural network architecture
- Replay buffer
- Environment wrappers
- Hyperparameters
- Training loop
- Logging system

This ensures a fair, **direct comparison** of algorithmic performance.

# 🧠 Observation & Preprocessing
+ Uses **full RGB images** from ALE(H x W x 3)
+ Converts to **channel-first** (C x H x W)
+ Stacks **4 consecutive frames** ➡️ final shape (12 x H x W)
+ Normalizes pixel values inside the network ( / 255.0)

This increases compute cost but preserves full color information.


# 🏗️ Model Architecture
The CNN architecture is identical for both agents:
``` python
Input -> Conv(32, 8x8, stride 4)
      -> Conv(64, 4x4, stride 2)
      -> Conv(64, 3x3, stride 1)
      -> Flatten
      -> FC(512)
      -> FC(n_actions)
      -> Q-values
```
Implemented in PyTorch

# ⚙️ Hyperparameters (Optimized for Google Colab)
``` python
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 50_000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 5_000
REPLAY_START_SIZE = 10_000

EPSILON_START = 1.0
EPSILON_FINAL = 0.05
EPSILON_DECAY_LAST_FRAME = 300_000

TOTAL_FRAMES = ~1_000_000
```
These parameters allow both models to train in **2-5 hours** on T4 or L4 GPU

# ⚖️ Differences Between Baseline DQN and Double DQN
**Baseline DQN Target**
``` python
next_state_values = tgt_net(next_states).max(1)[0]
target = rewards + GAMMA * next_state_values
```

**Double DQN Target**
``` python
next_actions = net(next_states).argmax(dim=1)
next_q = tgt_net(next_states).gather(1, next_actions.unsqueeze(-1))
target = rewards + GAMMA * next_q
```
Double DQN decouples selection and evalutation, greatly improving stability

# 📈 Learning Curve Results
**Baseline DQN Observations**
+ High variance in rewards
+ Occasional collapses
+ Overestimation bias visible in noisy Q-values
+ Final moving-average score: 101.850

**Double DQN Observations**
+ Smoother learning progression
+ Significantly fewer reward colapses
+ Better long-term strategy
+ Higher final score: 103.750

Graphs are generated in the notebook and included in the report.

# 🎥 Videos
Both agents include:
**🎞️ Early Gameplay**
+ Random policy (before training)

**🎞️ Learned Gameplay**
+ The trained policy acting greedily from Q-values

Videos are saved in:
```python
videos/early/
videos/late/
```
Links:

# 📊 Experiment Log
All runs log hyperparameters, timestamps, and final mean rewards to:
``` python
experiments/experiment_log.csv
```
Includes:
+ Replay size
+ Gamma
+ Epsilon schedule
+ Total training frames
+ Final mean reward

# 📫 Contact
Name: Patrick F. Cortez

Email: patrick24.pc14@gmail.com
