"""Train a Deep Q-Network to play Atari Pong.

Run from the project root:

    uv run python train.py --experiment 1 --episodes 400

Ctrl+C stops the run and still writes the log, the plot, and the best model
saved so far. The learning rule itself is in dqn.py.
"""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from dqn import (
    ENV_ID,
    QNetwork,
    ReplayBuffer,
    copy_weights,
    make_env,
    prepare_state,
    select_action,
    update_q_network,
)

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on Atari Pong.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment", type=int, default=1)
    parser.add_argument("--member", default="Emmanuel-NS")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.992,
        help="Multiplied with epsilon after every episode. 0.992 reaches 0.05 in about 370 episodes.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=5000,
        help="Past transitions kept in memory. 5000 fits an 8GB laptop. The DQN paper used 1,000,000.",
    )
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument(
        "--train-every",
        type=int,
        default=4,
        help="Gradient steps are taken every this many environment steps, as in the DQN paper.",
    )
    parser.add_argument(
        "--target-update",
        type=int,
        default=1000,
        help="Copy online weights onto the target network every this many steps.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Cut each episode off after this many steps. Leave at 0 to play a full game.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the experiment folder. Used for short tests.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Copy this run to dqn_model.pth and plots/ when it finishes.",
    )
    parser.add_argument(
        "--promote-from",
        type=int,
        default=None,
        help="Do not train. Copy this experiment number to dqn_model.pth and plots/.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def experiment_directory(member, experiment, output_dir):
    if output_dir:
        path = Path(output_dir)
        return path if path.is_absolute() else ROOT / path
    return ROOT / "models" / member / f"experiment_{experiment:02d}"


def save_model(path, online, args, n_actions, episode, total_steps):
    torch.save(
        {
            "state_dict": online.state_dict(),
            "n_actions": n_actions,
            "episode": episode,
            "total_steps": total_steps,
            "env": ENV_ID,
            "hyperparameters": {
                "learning_rate": args.learning_rate,
                "gamma": args.gamma,
                "batch_size": args.batch_size,
                "epsilon_start": args.epsilon_start,
                "epsilon_end": args.epsilon_end,
                "epsilon_decay": args.epsilon_decay,
            },
        },
        path,
    )


def moving_average(values, window):
    averages = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
            averages.append(total / window)
        else:
            averages.append(total / (index + 1))
    return averages


def save_plots(history, path, title):
    import os

    # Colab sets a notebook backend that this save cannot use.
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = [row["episode"] for row in history]
    rewards = [row["reward"] for row in history]
    lengths = [row["length"] for row in history]
    steps = [row["total_steps"] for row in history]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(episodes, rewards, label="Episode reward")
    axes[0].plot(episodes, moving_average(rewards, 20), label="Last 20 episodes")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].set_title(title)

    axes[1].plot(episodes, lengths)
    axes[1].set_ylabel("Episode length")

    axes[2].plot(episodes, steps)
    axes[2].set_ylabel("Total steps")
    axes[2].set_xlabel("Episode")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_summary(path, history, best_episode, best_average, total_steps):
    rewards = [row["reward"] for row in history]
    last_20 = rewards[-20:] if rewards else []
    summary = {
        "episodes_completed": len(history),
        "total_steps": total_steps,
        "last_20_average_reward": (sum(last_20) / len(last_20)) if last_20 else None,
        "best_episode_reward": max(rewards) if rewards else None,
        "best_checkpoint_episode": best_episode,
        "best_checkpoint_last_20_average": best_average,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def promote_folder(source, member, experiment):
    """Copy one finished experiment to the files the submission asks for."""
    model = source / "dqn_model.pth"
    if not model.exists():
        raise SystemExit(f"No trained model at {model}")

    plots = ROOT / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, ROOT / "dqn_model.pth")
    print(f"Copied {ROOT / 'dqn_model.pth'}")

    for name in ("training_performance.png", "training_log.csv", "summary.json"):
        source_file = source / name
        if source_file.exists():
            shutil.copy2(source_file, plots / name)
            print(f"Copied {plots / name}")
        else:
            print(f"Missing {source_file}, skipped")

    (plots / "BEST.txt").write_text(
        f"member: {member}\nexperiment: {experiment}\nsource: {source}\n",
        encoding="utf-8",
    )


def train(args):
    if args.buffer_size < args.batch_size:
        raise SystemExit("buffer-size must be at least as large as batch-size")
    if args.train_every < 1 or args.target_update < 1:
        raise SystemExit("train-every and target-update must be at least 1")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = experiment_directory(args.member, args.experiment, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "env": ENV_ID,
        "member": args.member,
        "experiment": args.experiment,
        "episodes": args.episodes,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay": args.epsilon_decay,
        "buffer_size": args.buffer_size,
        "learning_starts": max(args.learning_starts, args.batch_size),
        "train_every": args.train_every,
        "target_update": args.target_update,
        "seed": args.seed,
        "device": str(device),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Environment: {ENV_ID}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(
        f"Learning rate: {args.learning_rate} | gamma: {args.gamma} | "
        f"batch: {args.batch_size}"
    )
    print(
        f"Epsilon: {args.epsilon_start} -> {args.epsilon_end}, "
        f"decay {args.epsilon_decay} per episode"
    )
    print(
        f"Buffer: {args.buffer_size} | learning starts: {config['learning_starts']} | "
        f"train every: {args.train_every} | target update: {args.target_update}"
    )
    print(f"Episodes: {args.episodes}")
    if args.episodes < 20:
        print("Short run: the saved weights will be the final weights.")
    print("Ctrl+C saves the log and the best model so far.")

    env = make_env(render_mode=None)
    n_actions = env.action_space.n
    online = QNetwork(n_actions).to(device)
    target = QNetwork(n_actions).to(device)
    copy_weights(online, target)
    online.train()
    optimizer = torch.optim.Adam(online.parameters(), lr=args.learning_rate)
    buffer = ReplayBuffer(args.buffer_size)

    epsilon = args.epsilon_start
    total_steps = 0
    history = []
    best_average = float("-inf")
    best_episode = None
    model_path = output_dir / "dqn_model.pth"
    reset_seed = args.seed

    try:
        observation, _ = env.reset(seed=reset_seed)
        env.action_space.seed(args.seed)
        state = prepare_state(observation)
        print(f"Observation shape: {tuple(state.shape)} | actions: {n_actions}")

        log_path = output_dir / "training_log.csv"
        with log_path.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(
                [
                    "episode",
                    "episode_length",
                    "total_steps",
                    "reward",
                    "points_for",
                    "points_against",
                    "epsilon",
                    "mean_loss",
                ]
            )
            try:
                for episode in range(1, args.episodes + 1):
                    if episode > 1:
                        observation, _ = env.reset()
                        state = prepare_state(observation)

                    episode_reward = 0.0
                    episode_steps = 0
                    points_for = 0
                    points_against = 0
                    losses = []
                    ended = False

                    while not ended:
                        action = select_action(
                            online, state, epsilon, n_actions, device
                        )
                        next_observation, reward, terminated, truncated, _ = env.step(
                            action
                        )
                        raw_reward = float(reward)
                        # Atari scores can be huge. Store -1, 0, or 1 so one
                        # game cannot dominate the loss. Pong is already in range.
                        stored_reward = float(np.clip(raw_reward, -1.0, 1.0))
                        next_state = prepare_state(next_observation)
                        # `terminated` means the match is over. A time limit
                        # (truncated) still has a future, so the target bootstraps.
                        buffer.push(
                            state,
                            action,
                            stored_reward,
                            next_state,
                            terminated,
                        )

                        state = next_state
                        episode_reward += raw_reward
                        episode_steps += 1
                        total_steps += 1
                        if raw_reward > 0:
                            points_for += int(round(raw_reward))
                        elif raw_reward < 0:
                            points_against += int(round(-raw_reward))

                        ready = (
                            buffer.size >= args.batch_size
                            and total_steps >= config["learning_starts"]
                            and total_steps % args.train_every == 0
                        )
                        if ready:
                            loss = update_q_network(
                                online,
                                target,
                                optimizer,
                                buffer,
                                args.batch_size,
                                args.gamma,
                                device,
                            )
                            losses.append(loss)

                        if total_steps % args.target_update == 0:
                            copy_weights(online, target)

                        hit_limit = (
                            args.max_steps > 0 and episode_steps >= args.max_steps
                        )
                        ended = terminated or truncated or hit_limit

                    mean_loss = (sum(losses) / len(losses)) if losses else None
                    history.append(
                        {
                            "episode": episode,
                            "length": episode_steps,
                            "total_steps": total_steps,
                            "reward": episode_reward,
                            "mean_loss": mean_loss,
                        }
                    )
                    writer.writerow(
                        [
                            episode,
                            episode_steps,
                            total_steps,
                            episode_reward,
                            points_for,
                            points_against,
                            f"{epsilon:.6f}",
                            "" if mean_loss is None else f"{mean_loss:.6f}",
                        ]
                    )
                    log_file.flush()

                    loss_text = "n/a" if mean_loss is None else f"{mean_loss:.4f}"
                    print(
                        f"Episode {episode:4d} | score {points_for}-{points_against} | "
                        f"reward {episode_reward:7.1f} | length {episode_steps:5d} | "
                        f"steps {total_steps:8d} | epsilon {epsilon:.3f} | loss {loss_text}",
                        flush=True,
                    )

                    # Keep the weights from the strongest stretch of 20 games,
                    # not from one lucky game.
                    if len(history) >= 20:
                        recent = [row["reward"] for row in history[-20:]]
                        average = sum(recent) / 20.0
                        if average > best_average:
                            best_average = average
                            best_episode = episode
                            save_model(
                                model_path,
                                online,
                                args,
                                n_actions,
                                episode,
                                total_steps,
                            )

                    epsilon = min(
                        1.0, max(args.epsilon_end, epsilon * args.epsilon_decay)
                    )

                    # Write the plot while the run is still going, so a long
                    # job can be checked without waiting for the last episode.
                    if episode % 20 == 0:
                        save_plots(
                            history,
                            output_dir / "training_performance.png",
                            title=f"{ENV_ID} | {args.member} experiment {args.experiment}",
                        )
                        write_summary(
                            output_dir / "summary.json",
                            history,
                            best_episode,
                            None if best_average == float("-inf") else best_average,
                            total_steps,
                        )
            except KeyboardInterrupt:
                print("\nStopped early. Saving the log and the best model so far.")
    finally:
        env.close()

    if history and not model_path.exists():
        save_model(
            model_path,
            online,
            args,
            n_actions,
            history[-1]["episode"],
            total_steps,
        )
        best_episode = history[-1]["episode"]

    summary = None
    if history:
        save_plots(
            history,
            output_dir / "training_performance.png",
            title=f"{ENV_ID} | {args.member} experiment {args.experiment}",
        )
        summary = write_summary(
            output_dir / "summary.json",
            history,
            best_episode,
            None if best_average == float("-inf") else best_average,
            total_steps,
        )
        window = min(20, summary["episodes_completed"])
        print(
            f"Average reward over the last {window} episodes: "
            f"{summary['last_20_average_reward']:.2f} | "
            f"best episode reward: {summary['best_episode_reward']:.1f}"
        )
        print(f"Model: {model_path}")
        print("Put the last-20 average and a sentence about what you saw into the experiment table.")

    if args.promote and model_path.exists():
        promote_folder(output_dir, args.member, args.experiment)


def main():
    args = parse_args()
    if args.promote_from is not None:
        source = experiment_directory(args.member, args.promote_from, None)
        promote_folder(source, args.member, args.promote_from)
        return
    train(args)


if __name__ == "__main__":
    main()
