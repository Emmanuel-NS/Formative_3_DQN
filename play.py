"""Watch a trained DQN play Atari Pong.

    uv run python play.py

The agent always takes the action with the highest predicted Q-value.
Epsilon is 0 here: this is evaluation, not exploration.
render_mode="human" opens the game window on each step.
"""

import argparse
import time
from pathlib import Path

import torch

from dqn import QNetwork, make_env, prepare_state, select_action

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play Atari Pong with a trained DQN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=str(ROOT / "dqn_model.pth"))
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--render", choices=["human", "none"], default="human")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.03,
        help="Seconds to wait after each action so the game is visible. Use 0 to go as fast as possible.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop each game after this many steps. Leave at 0 to play until the game ends.",
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def load_network(path, device):
    model_path = Path(path)
    if not model_path.exists():
        raise SystemExit(
            f"Could not find {model_path}. Train first, then promote that run:\n"
            "  uv run python train.py --experiment 1\n"
            "  uv run python train.py --promote-from 1"
        )

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    n_actions = int(checkpoint["n_actions"])
    network = QNetwork(n_actions).to(device)
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()
    return network, n_actions, checkpoint


def play(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network, n_actions, checkpoint = load_network(args.model, device)
    render_mode = None if args.render == "none" else "human"
    env = make_env(render_mode=render_mode)

    if env.action_space.n != n_actions:
        env.close()
        raise SystemExit(
            f"The saved model has {n_actions} actions but this game has "
            f"{env.action_space.n}."
        )

    print(f"Loaded {args.model}")
    print(f"Trained env: {checkpoint.get('env', 'unknown')} | device: {device}")
    if hasattr(env.unwrapped, "get_action_meanings"):
        print("Actions:", env.unwrapped.get_action_meanings())
    print("Greedy play: the action is always the one with the largest Q-value.")

    scores = []
    try:
        for game in range(1, args.games + 1):
            observation, _ = env.reset(seed=args.seed + game)
            state = prepare_state(observation)
            points_for = 0
            points_against = 0
            steps = 0
            ended = False

            while not ended:
                # epsilon = 0, so select_action never explores.
                action = select_action(network, state, 0.0, n_actions, device)
                observation, reward, terminated, truncated, _ = env.step(action)
                state = prepare_state(observation)
                reward = float(reward)
                steps += 1
                if reward > 0:
                    points_for += int(round(reward))
                elif reward < 0:
                    points_against += int(round(-reward))

                if args.delay > 0 and render_mode == "human":
                    time.sleep(args.delay)

                hit_limit = args.max_steps > 0 and steps >= args.max_steps
                ended = terminated or truncated or hit_limit

            score = points_for - points_against
            scores.append(score)
            print(
                f"Game {game}: score {points_for}-{points_against} | "
                f"reward {score:.0f} | steps {steps}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()

    if scores:
        average = sum(scores) / len(scores)
        print(f"Average reward over {len(scores)} game(s): {average:.2f}")
        print("A random agent is about -20 or -21. Closer to +21 is better.")


if __name__ == "__main__":
    play(parse_args())
