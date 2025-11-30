# Space Invaders DQN / Double DQN (PyTorch + Gymnasium Atari)
This project implements a Deep-Q-Network (DQN) and Double DQN agent to play **Space Invaders** from the Atari Learning Environment (ALE).
It uses **PyTorch**, **Gymnasium**, **AntiPreprocessing**, **frame stacking**, **replay buffer**, and supports **Google Drive mdoel saving**, and **TensorBoard logging and checkpointing**


# 🚀 **Features**
+ Atari-ready DQN model with convolutional layers
+ Double DQN toggle (USE_DOUBLE_DQN = True/False)
+ Frame stacking (4 frames) + grayscale + resize(84 x 84 or 96 x 96)
+ Replay buffer with random sampling
+ Epsilon-greedy exploration
+ Target network synchronization
+ Learning rate / gamma / batch size configuration
+ TensorBoard logging
+ Checkpoint saving every N frames
+ Google Drive syncing for Colab users
+ Automatic best-model saving (based on mean 100-episode reward)
+ Automatic CSV logging + learning curve plot
