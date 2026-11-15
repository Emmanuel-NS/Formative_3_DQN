# Hyperparameter experiments

Member name: Emmanuel-NS

Environment: ALE/Pong-v5

Each row changes **one** value from experiment 1. Everything else stays on the baseline, including the seed (1). The episode budget is not the same for every row, so compare each row with experiment 1 at the same episode, not with experiment 1's final score.

The number in each row is the **last 20 average reward** from `models/Emmanuel-NS/experiment_XX/summary.json`. Higher is better. A random player scores about -21.

Every score below comes from that experiment's `summary.json` and training log.

Experiments 5 and 6 used **250 episodes**. Compare them with experiment 1 at episode 250, which was **-17.3**. Experiment 7 used **200 episodes**. Compare it with experiment 1 at episode 200, which was **-19.6**. Experiments 8 to 10 used **120 episodes**. Compare them with experiment 1 at episode 120, which was **-20.25**. Experiment 1 was still about -20.7 at episode 150 and reached -13.1 only at episode 400. A run that is still near -21 at its comparison episode did worse than the baseline. A run of 120 episodes can show a collapse, and it can miss a rally that starts later. Experiment 7's first 10–21 game was at episode 158.

Best experiment: Experiment 1. It is the only full 400-episode run that learned to return the ball. Last-20 average -13.1, best game 17–21. Among the changed settings, experiment 7 (batch 128) was the clearest gain at a matched length: -16.3 against experiment 1's -19.6 at episode 200.

Worst experiment: Experiment 2, learning rate 0.001. Last-20 average -21.0 after 400 episodes. The paddle never learned to return the ball.

Final configuration, and why: Keep experiment 1 for `dqn_model.pth`. Experiment 7 was ahead at episode 200, but that run stopped at 200 episodes and its best game was 12–21. One seed and a shorter run is not enough to replace the 400-episode baseline.

## Results

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.992 | Finished 400 episodes. Last-20 average -13.1. At episode 250 it was -17.3. The paddle returns the ball and reached 17–21 in the best game, but still loses. |
| 2 | 0.001 | 0.99 | 32 | 1.0 | 0.05 | 0.992 | Finished 400 episodes. Last-20 average -21.0. At episode 250 it was -20.9. The paddle never learned to return the ball. The best single game was 4–21, and the final games were 0–21. |
| 3 | 0.00001 | 0.99 | 32 | 1.0 | 0.05 | 0.992 | Finished 400 episodes. Last-20 average -20.95. At episode 250 it was -20.35. The loss fell, but the paddle did not learn to rally. The best game was 4–21, and the final games were 0–21. |
| 4 | 0.0001 | 0.9 | 32 | 1.0 | 0.05 | 0.992 | Stopped at episode 307 of 400 after the process froze. At episode 250 the last-20 average was -17.3, the same as experiment 1 at that point. Best game was 9–21. It later slipped to -18.1 by episode 300. |
| 5 | 0.0001 | 0.999 | 32 | 1.0 | 0.05 | 0.992 | Finished 250 episodes. Last-20 average -16.3, better than experiment 1 at the same point (-17.3). Best 20-game stretch -16.25 at episode 248. Best game was 11–21. The paddle returns the ball often and still loses the match. |
| 6 | 0.0001 | 0.99 | 16 | 1.0 | 0.05 | 0.992 | Finished 250 episodes. Last-20 average -20.75, worse than experiment 1 at the same point (-17.3). Best 20-game stretch was -20.05 at episode 36. Best game was 4–21, and the final games were 0–21. The paddle did not learn to rally. |
| 7 | 0.0001 | 0.99 | 128 | 1.0 | 0.05 | 0.992 | Finished 200 episodes. Last-20 average -16.3, better than experiment 1 at the same point (-19.6). Best 20-game stretch was -15.3 at episode 170. Best game was 12–21. The paddle returns the ball and still loses the match. |
| 8 | 0.0001 | 0.99 | 32 | 0.2 | 0.05 | 0.992 | Finished 120 episodes. Last-20 average -19.15, a little better than experiment 1 at episode 120 (-20.25). Best 20-game stretch was -18.7 at episode 100. Best game was 5–21. The paddle scored a few points and still lost every match. |
| 9 | 0.0001 | 0.99 | 32 | 1.0 | 0.4 | 0.992 | Finished 120 episodes. Last-20 average -20.45, a little worse than experiment 1 at episode 120 (-20.25). Best 20-game stretch was -20.0 at episode 34. Best game was 3–21. Epsilon stayed at 0.4, and the final games were almost all 0–21. |
| 10 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.97 | Finished 120 episodes. Last-20 average -19.6, a little better than experiment 1 at episode 120 (-20.25). Best 20-game stretch was -19.45 at episode 119. Best game was 7–21. Epsilon was already at 0.05 late in the run. The paddle scored a few points and still lost every match. |

## Commands

These are the 10 runs behind the table. Experiment 1 is the model in `dqn_model.pth`.

```powershell
uv run python train.py --member Emmanuel-NS --experiment 1 --episodes 400 --learning-rate 0.0001 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.992
uv run python train.py --member Emmanuel-NS --experiment 2 --episodes 400 --learning-rate 0.001 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.992
uv run python train.py --member Emmanuel-NS --experiment 3 --episodes 400 --learning-rate 0.00001 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.992
uv run python train.py --member Emmanuel-NS --experiment 4 --episodes 400 --learning-rate 0.0001 --gamma 0.9 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.992
uv run python train.py --member Emmanuel-NS --experiment 5 --episodes 250 --learning-rate 0.0001 --gamma 0.999 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.992
uv run python train.py --member Emmanuel-NS --experiment 6 --episodes 250 --learning-rate 0.0001 --gamma 0.99 --batch-size 16 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.992
uv run python train.py --member Emmanuel-NS --experiment 7 --episodes 200 --learning-rate 0.0001 --gamma 0.99 --batch-size 128 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.992
uv run python train.py --member Emmanuel-NS --experiment 8 --episodes 120 --learning-rate 0.0001 --gamma 0.99 --batch-size 32 --epsilon-start 0.2 --epsilon-end 0.05 --epsilon-decay 0.992
uv run python train.py --member Emmanuel-NS --experiment 9 --episodes 120 --learning-rate 0.0001 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.4 --epsilon-decay 0.992
uv run python train.py --member Emmanuel-NS --experiment 10 --episodes 120 --learning-rate 0.0001 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.97
uv run python train.py --member Emmanuel-NS --promote-from 1
```
