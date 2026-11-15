# Deep Q-Network for Atari Pong

A Deep Q-Network (DQN) learns to play Atari Pong by playing the game and updating its own estimate of how good each action is. The agent is written in PyTorch. It does not use Stable-Baselines3, RLlib, CleanRL, TorchRL, or any other library that already implements DQN.

## Group members

| Member | Experiment file | Model folder |
| --- | --- | --- |
| Emmanuel-NS | `Emmanuel-NSHyperparameter_experiment_results.md` | `models/Emmanuel-NS/` |
| Mukasa | `MukasaHyperparameter_experiment_results.md` | `models/Mukasa/` |
| Henriette | `HenrietteHyperparameter_experiment_results.md` | `models/Henriette/` |

Each member runs 10 experiments with their own `--member` name and fills in only their own table. The shared code is `dqn.py`, `train.py`, and `play.py`.

## Selected environment

**ALE/Pong-v5** from the Gymnasium Atari collection.

Pong is a two-paddle game. The agent controls one paddle. The opponent is the built-in Atari player. A point is scored when the ball passes a paddle. The reward is **+1** when the agent scores and **-1** when the opponent scores. The match ends when one side reaches 21 points.

The useful actions move the paddle. In the ALE names those actions are `RIGHT` and `LEFT` (they come from the joystick). `NOOP` and `FIRE` do little once the rally has started. The network still has one output for every legal action and has to learn which outputs matter.

A random agent loses almost every match. Its episode reward is about **-20 to -21**. A strong agent finishes with a reward closer to **+21**.

## DQN architecture

The input is **4 grayscale frames**, each **84 by 84**. One frame cannot show which way the ball is moving. Four frames in a row can.

Both the online network and the target network use this architecture. They have the same layers. Only the online network is trained with the optimizer. The target network is a copy that is refreshed every 1000 steps.

```text
Input                         (4, 84, 84)
  Conv2d 32 filters, 8x8, stride 4, ReLU     -> (32, 20, 20)
  Conv2d 64 filters, 4x4, stride 2, ReLU     -> (64, 9, 9)
  Conv2d 64 filters, 3x3, stride 1, ReLU     -> (64, 7, 7)
  Flatten                                    -> 3136
  Linear 512, ReLU
  Linear n_actions                           -> one Q-value per action
```

`64 * 7 * 7 = 3136`. The last layer has **no activation**. Q-values are allowed to be negative, because a lost point is a reward of -1.

```text
        state s                         next state s'
           |                                  |
           v                                  v
   +---------------+                  +---------------+
   | Online Q net  |  copy every      | Target Q net  |
   | (trained)     |  1000 steps ---> | (frozen)      |
   +---------------+                  +---------------+
           |                                  |
           v                                  v
   Q(s, a) for the action             max over a' of Q(s', a')
   that was taken                     used only to build the label y
```

The class is `QNetwork` in `dqn.py`.

## Training process

One transition is `(state, action, reward, next state, done)`.

1. **Epsilon-greedy action.** With probability epsilon the agent picks a random action (exploration). Otherwise it picks the action with the largest online Q-value (exploitation). Epsilon starts at 1, so the first episode is random. After each episode:

   `epsilon = max(epsilon_end, epsilon * epsilon_decay)`

   The default decay is **0.992**, which brings epsilon from 1.0 down to 0.05 in about 370 episodes:

   `episodes ≈ ln(epsilon_end / epsilon_start) / ln(decay)`

2. **Store the transition** in the replay buffer. The reward saved for learning is clipped to [-1, 1]. Pong rewards are already in that range. Clipping is there so a different Atari game with a huge score cannot drown the loss.

3. **Replay.** Once the buffer has enough transitions, every 4th step draws a random batch. The original DQN paper also updates every 4 steps. Random batches mix different moments of the game. Consecutive frames are too similar to learn from in order.

4. **Bellman target, from the target network only.**

   `y = r + gamma * max_a' Q_target(s', a') * (1 - done)`

   If the match is over, `done` is 1 and `y` is just `r`. There is no future value to add. `gamma` is how much a later point counts. At 0.99, a point one step from now is worth 99% of a point scored now.

5. **Loss and update.** The online network's Q-value for the action that was taken is compared with `y`:

   `loss = mean( (Q_online(s, a) - y)^2 )`

   That is mean squared error. Backpropagation computes the gradient, and Adam changes only the online weights. The target network does not receive a gradient. Every 1000 steps its weights are replaced with a copy of the online weights, so the labels stay still for a while instead of moving on every step.

6. **What is recorded.** After each episode the script prints the score, the episode length, the total step count, epsilon, and the average loss. It appends the same numbers to `training_log.csv`. The plot has three panels: reward, episode length, and total steps. The saved model is the one with the best average reward over 20 episodes, not the winner of a single lucky game.

The buffer default is **5000** transitions. The paper used 1,000,000. 5000 is small enough for an 8 GB laptop. Each transition stores two stacks of 4 frames.

## Setup

[uv](https://docs.astral.sh/uv/) is the package manager for this project. Do not use pip.

```powershell
# If `uv` is not on PATH (it is installed for this user here):
$env:Path = "$env:USERPROFILE\.local\bin;" + $env:Path

cd C:\Users\PC\formative-3-dqn
uv sync
uv run AutoROM --accept-license
```

`uv sync` creates `.venv` and installs PyTorch, Gymnasium, and the Atari ROMs. Accept the ROM license when AutoROM asks. The ROMs stay inside the environment and are not committed.

## Train

Run this from the project folder. `--promote` copies the run into `dqn_model.pth` and `plots/`, which is what the submission shows for the best experiment.

On this PC (no NVIDIA GPU) one 400-episode experiment takes roughly 4 hours, because every update runs on the CPU. Emmanuel-NS experiments 1 to 3 used 400 episodes, experiment 4 stopped at 307, experiments 5 and 6 used 250, experiment 7 used 200, and experiments 8 to 10 used 120.

One experiment by itself:

```powershell
uv run python train.py --experiment 1 --episodes 400 --promote
```

Other experiments omit `--promote`. After the 10 runs, copy the winner without training again:

```powershell
uv run python train.py --promote-from 1
```

For these results, experiment 1 is the one copied to `dqn_model.pth`. `--member Emmanuel-NS` stores a run in `models/Emmanuel-NS/`.

A full Pong match on this PC (Intel i5-6300U, no NVIDIA GPU, 8 GB RAM) is slow. The script prints `Device: cpu`. On a machine with an NVIDIA GPU the same script prints `Device: cuda` and uses it. No code change is required. Ctrl+C stops training and still writes the log, the plot, and the best checkpoint reached so far.

Compare runs with the **last 20 average reward** in `summary.json`. Higher is better. About -21 means the agent is still losing every point.

## Gameplay demonstration

After a model has been promoted:

```powershell
uv run python play.py
```

This opens the Pong window and plays 3 games. The agent is greedy: it always takes the action with the highest Q-value. The terminal prints the score after each game.

```powershell
uv run python play.py --games 5 --model dqn_model.pth
```

Best agent playing Pong: https://drive.google.com/file/d/1FCgffpthE-dl_0L3N8E8slXlpYOrQObJ/view?usp=drive_link

The presentation uses the live command above, not the recording.

## Repository layout

```text
train.py                  training loop
play.py                   load the model and show the game
dqn.py                    Q-network, replay buffer, Bellman update
dqn_model.pth             best model, created by --promote or --promote-from
plots/                    reward, episode length, and step plots for that best run
models/<member>/experiment_XX/
    dqn_model.pth
    training_log.csv
    training_performance.png
    summary.json
    config.json
Emmanuel-NSHyperparameter_experiment_results.md
```

`Task_allocation_sheet.pdf` still has to be added by the group. Copy the cohort booking sheet, book a Week 6 slot, export the copy as a PDF, and put it in this folder.

- Cohort 1 booking: https://calendar.app.google/fBUVy7TgAnEC4U6a8
- The Cohort 2 link was not included in the assignment text that was provided. Use the link posted for Cohort 2.

Each group has 20 minutes: 10 to present and 10 for questions. Cameras stay on. Every member answers questions.

## Presentation outline

- 1 minute: ALE/Pong-v5. One paddle, reward +1 or -1, first side to 21.
- 2 minutes: the Q-network, replay buffer, target network, epsilon-greedy actions, and the Bellman loss in the diagram above.
- 6 minutes: each member's own 10 experiments, from that member's results file.
- 2 minutes: `uv run python play.py`.

## Fixed settings

These stay the same across the 10 experiments so the six requested hyperparameters are the only change.

| Setting | Value | Why |
| --- | --- | --- |
| Optimizer | Adam | Updates the online network. Default Adam settings, learning rate from the table. |
| Replay buffer | 5000 | Fits in 8 GB of RAM. |
| Learning starts | 1000 steps | Fill the buffer with experience before the first gradient step. |
| Train every | 4 steps | Same rhythm as the DQN paper. |
| Target update | 1000 steps | The Bellman labels stay fixed between copies. |
| Seed | 1 | Same start for every experiment. |
| Reward clip | [-1, 1] | Keeps Q-values on one scale. |
| Episodes | Written in the results note | The brief does not fix a count. Experiments 1 to 3 used 400. Experiment 4 stopped at 307. Experiments 5 and 6 used 250. Experiment 7 used 200. Experiments 8 to 10 used 120. Compare each row with experiment 1 at the same episode. |
