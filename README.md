# Deep Q-Network for Atari Pong

This project trains a Deep Q-Network to play Atari Pong. The agent is written in PyTorch. The Q-network, the replay buffer, and the Bellman update are in `dqn.py`.

## Group

Formative 3 Group 11.

| Member | Results | Model folder |
| --- | --- | --- |
| Emmanuel Nsabagasani (group leader) | `Emmanuel-NSHyperparameter_experiment_results.md` | `models/Emmanuel-NS/` |
| Emmanuel Mukasa Simiyu | `MukasaHyperparameter_experiment_results.md` | `models/Mukasa/` |
| Henriette Utatsineza | `HenrietteHyperparameter_experiment_results.md` | `models/Henriette/` |

Each member trained 10 experiments and recorded the scores in their own results file. The shared code is `dqn.py`, `train.py`, and `play.py`.

## Environment

**ALE/Pong-v5**, from the Gymnasium Atari collection.

Pong has two paddles. The agent controls one paddle, and the opponent is the built-in Atari player. The reward is **+1** when the ball passes the opponent and **-1** when the ball passes the agent. The match ends when one side reaches 21 points.

`RIGHT` and `LEFT` move the paddle. `NOOP` and `FIRE` change little once a rally has started. The network has one output for every legal action, and it has to learn which outputs matter.

A random agent finishes around **-20 to -21**. A reward closer to **+21** is a stronger match.

## DQN architecture

The input is **4 grayscale frames**, each **84 by 84**. One frame does not show the direction of the ball. Four frames in a row do.

The online network and the target network use the same layers. Adam trains the online network. The target network is a copy, replaced every 1000 steps.

```text
Input                         (4, 84, 84)
  Conv2d 32 filters, 8x8, stride 4, ReLU     -> (32, 20, 20)
  Conv2d 64 filters, 4x4, stride 2, ReLU     -> (64, 9, 9)
  Conv2d 64 filters, 3x3, stride 1, ReLU     -> (64, 7, 7)
  Flatten                                    -> 3136
  Linear 512, ReLU
  Linear n_actions                           -> one Q-value per action
```

`64 * 7 * 7 = 3136`. The last layer has **no activation**, so a Q-value can be negative. A lost point has reward -1.

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
   that was taken                     used to build the label y
```

The class is `QNetwork` in `dqn.py`.

## Training process

One transition is `(state, action, reward, next state, done)`.

1. **Epsilon-greedy action.** With probability epsilon the agent picks a random action. Otherwise it picks the action with the largest online Q-value. Epsilon starts at 1, so the first episode is random. After each episode:

   `epsilon = max(epsilon_end, epsilon * epsilon_decay)`

   The default decay is **0.992**. That brings epsilon from 1.0 down to 0.05 in about 370 episodes:

   `episodes ≈ ln(epsilon_end / epsilon_start) / ln(decay)`

2. **Store the transition** in the replay buffer. The reward stored for learning is clipped to [-1, 1]. Pong rewards are already inside that range. The clip keeps a game with a large score from dominating the loss.

3. **Replay.** After the buffer has enough transitions, every 4th environment step draws a random batch. The DQN paper updates on the same rhythm. A random batch mixes different moments of the game, because consecutive frames are very similar.

4. **Bellman target, from the target network.**

   `y = r + gamma * max_a' Q_target(s', a') * (1 - done)`

   When the match is over, `done` is 1 and `y` is `r`. `gamma` is the weight of a later reward. At 0.99, a point one step ahead counts as 99% of a point scored now.

5. **Loss and update.** The online Q-value for the action that was taken is compared with `y`:

   `loss = mean( (Q_online(s, a) - y)^2 )`

   The loss is mean squared error. Adam updates the online weights. The target network receives a full copy of those weights every 1000 steps, so the labels stay fixed between copies.

6. **Records.** After each episode the script prints the score, the episode length, the total step count, epsilon, and the average loss, and it appends the same numbers to `training_log.csv`. The plot has three panels: reward, episode length, and total steps. The saved checkpoint is the one with the best average reward over 20 episodes.

The replay buffer holds **5000** transitions. The DQN paper used 1,000,000. Five thousand fits in 8 GB of RAM, because each transition stores two stacks of 4 frames.

## Setup

Packages are installed with [uv](https://docs.astral.sh/uv/). From the project folder:

```powershell
$env:Path = "$env:USERPROFILE\.local\bin;" + $env:Path
uv sync
uv run AutoROM --accept-license
```

Skip the `Path` line when `uv` is already on `PATH`. `uv sync` creates `.venv` and installs PyTorch, Gymnasium, and the Atari ROMs. The ROMs stay inside the environment.

## Train

`--promote` copies a finished run to `dqn_model.pth` and `plots/`. That copy is the model used for the video and for `play.py`.

On this PC (Intel i5-6300U, no NVIDIA GPU, 8 GB RAM) the script prints `Device: cpu`. A 400-episode run takes roughly 4 hours. Emmanuel Nsabagasani's experiments 1 to 3 used 400 episodes, experiment 4 stopped at 307, experiments 5 and 6 used 250, experiment 7 used 200, and experiments 8 to 10 used 120. On a machine with an NVIDIA GPU the same script prints `Device: cuda`.

One experiment:

```powershell
uv run python train.py --experiment 1 --episodes 400 --promote
```

The other experiments leave out `--promote`. After the runs, this copies experiment 1 without training again:

```powershell
uv run python train.py --promote-from 1
```

Experiment 1 is the model in `dqn_model.pth`. `--member Emmanuel-NS` writes a run to `models/Emmanuel-NS/`. Ctrl+C stops training and still writes the log, the plot, and the best checkpoint reached so far.

Compare runs with the **last 20 average reward** in `summary.json`. Higher is better. A value near -21 means the agent is still losing almost every point.

## Gameplay

```powershell
uv run python play.py
```

This opens the Pong window and plays 3 games. The agent is greedy: epsilon is 0, so it always takes the action with the highest Q-value. The terminal prints the score after each game.

```powershell
uv run python play.py --games 5 --model dqn_model.pth
```

Best agent playing Pong: https://drive.google.com/file/d/1FCgffpthE-dl_0L3N8E8slXlpYOrQObJ/view?usp=drive_link

The Week 6 presentation plays the game live with `uv run python play.py`. The link above is the recording.

## Repository layout

```text
train.py                  training loop
play.py                   load the model and show the game
dqn.py                    Q-network, replay buffer, Bellman update
dqn_model.pth             best model, from --promote or --promote-from
plots/                    reward, episode length, and step plots for that model
Task_allocation_sheet.pdf group task sheet
models/<member>/experiment_XX/
    dqn_model.pth
    training_log.csv
    training_performance.png
    summary.json
    config.json
Emmanuel-NSHyperparameter_experiment_results.md
MukasaHyperparameter_experiment_results.md
HenrietteHyperparameter_experiment_results.md
```

## Presentation

The task sheet is `Task_allocation_sheet.pdf`. We still have to book a slot for the presentation.

- Cohort 1 booking: https://calendar.app.google/fBUVy7TgAnEC4U6a8

The slot is 20 minutes: 10 to present and 10 for questions. Cameras stay on. Each member answers questions about their own experiments.

- 1 minute: ALE/Pong-v5. One paddle, reward +1 or -1, first side to 21.
- 2 minutes: the Q-network, replay buffer, target network, epsilon-greedy actions, and the Bellman loss.
- 6 minutes: the three results tables, about 2 minutes each.
- 2 minutes: `uv run python play.py`.

## Fixed settings

These stay the same inside one member's 10 experiments. The six hyperparameters in that member's table are the values that change.

| Setting | Value | Why |
| --- | --- | --- |
| Optimizer | Adam | Updates the online network. Default Adam settings, with the learning rate from the table. |
| Replay buffer | 5000 | Fits in 8 GB of RAM. |
| Learning starts | 1000 steps | Collect transitions before the first gradient step. |
| Train every | 4 steps | Same update rhythm as the DQN paper. |
| Target update | 1000 steps | The Bellman labels stay fixed between copies. |
| Seed | 1 | The same start for every experiment. |
| Reward clip | [-1, 1] | Keeps the Q-values on one scale. |
| Episodes | In each results file | Experiments 1 to 3 used 400. Experiment 4 stopped at 307. Experiments 5 and 6 used 250. Experiment 7 used 200. Experiments 8 to 10 used 120. Compare a row with experiment 1 at the same episode. |
