"""Deep Q-Network pieces shared by train.py and play.py.

The two scripts import this file so the network that plays the game is the
same network that was trained. Nothing here starts training or opens a window.
"""

import random

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

# Pong from the Arcade Learning Environment, through Gymnasium.
ENV_ID = "ALE/Pong-v5"

# Register the Atari games with Gymnasium once, when this file is imported.
gym.register_envs(ale_py)


def make_env(render_mode=None):
    """Build Pong the way the DQN paper sees it.

    frameskip=1 on the raw game, because AtariPreprocessing repeats each
    action for 4 frames itself. Doing both would skip 16 frames.

    The wrapper then:
    - turns the colour frame into an 84x84 grayscale image
    - keeps the brighter pixel of two consecutive frames (sprites flicker)
    - stacks 4 frames, so the network can see which way the ball is moving
    """
    env = gym.make(
        ENV_ID,
        render_mode=render_mode,
        frameskip=1,
        repeat_action_probability=0.0,
    )
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


def prepare_state(observation):
    """Return one observation as uint8 array of shape (4, 84, 84)."""
    array = np.squeeze(np.asarray(observation))
    if array.ndim != 3:
        raise ValueError(f"Expected a stack of frames, got shape {array.shape}")

    # Some wrapper versions stack on the last axis: (84, 84, 4).
    if array.shape[-1] == 4 and array.shape[0] != 4:
        array = np.transpose(array, (2, 0, 1))

    if array.shape != (4, 84, 84):
        raise ValueError(
            "Expected shape (4, 84, 84) after preprocessing, "
            f"got {tuple(array.shape)}"
        )

    if np.issubdtype(array.dtype, np.floating) and float(array.max()) <= 1.0:
        array = array * 255.0
    return np.ascontiguousarray(array, dtype=np.uint8)


def observations_to_tensor(observations, device):
    """uint8 pixels in [0, 255] -> float pixels in [0, 1]."""
    tensor = torch.tensor(np.asarray(observations), dtype=torch.float32, device=device)
    return tensor / 255.0


class QNetwork(nn.Module):
    """Convolutional network that maps 4 frames to one Q-value per action.

    This is the network from the Nature DQN paper (Mnih et al., 2015).
    Input:  (batch, 4, 84, 84)
    Conv 8x8 stride 4: 84 -> 20, 32 filters
    Conv 4x4 stride 2: 20 -> 9,  64 filters
    Conv 3x3 stride 1: 9  -> 7,  64 filters
    Flatten: 64 * 7 * 7 = 3136
    Linear 512, then one output per action.

    There is no activation on the last layer. Q-values in Pong are often
    negative, because losing a point is a reward of -1.
    """

    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size memory of transitions (state, action, reward, next state, done).

    The update draws a random batch. Consecutive frames are almost the same,
    so learning from them in order would keep training on one moment of the
    game. A random batch mixes different moments together.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.states = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.next_states = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.index = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        self.next_states[self.index] = next_state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = float(done)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        states = observations_to_tensor(self.states[indices], device)
        next_states = observations_to_tensor(self.next_states[indices], device)
        actions = torch.tensor(self.actions[indices], dtype=torch.int64, device=device)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones[indices], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones


def select_action(network, state, epsilon, n_actions, device):
    """Epsilon-greedy: random action with probability epsilon, otherwise argmax Q."""
    if random.random() < epsilon:
        return random.randrange(n_actions)

    # Choosing an action is not a learning step, so no gradient is stored.
    with torch.no_grad():
        q_values = network(observations_to_tensor(state, device).unsqueeze(0))
    return int(q_values.argmax(dim=1).item())


def copy_weights(online, target):
    """Make the target network a frozen copy of the online network."""
    target.load_state_dict(online.state_dict())
    for parameter in target.parameters():
        parameter.requires_grad = False
    target.eval()


def update_q_network(online, target, optimizer, buffer, batch_size, gamma, device):
    """One gradient step on the squared Bellman error.

    y = r + gamma * max_a' Q_target(s', a') * (1 - done)
    loss = mean( (Q_online(s, a) - y) ^ 2 )

    `done` is 1 only when the game has ended. Then y is just r, because
    there is no future. The target network is not trained here. Its weights
    change only when copy_weights is called.
    """
    states, actions, rewards, next_states, dones = buffer.sample(batch_size, device)

    # Q(s, a) for the action that was actually taken. Shape: (batch,)
    q_values = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = target(next_states).max(dim=1).values
        targets = rewards + gamma * next_q * (1.0 - dones)

    loss = nn.functional.mse_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())
