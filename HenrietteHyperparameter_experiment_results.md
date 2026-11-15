# Hyperparameter experiments

Member name: Henriette

Environment: ALE/Pong-v5

Each row changes **one** value from experiment 1. The seed stays 1. Every run is **120 episodes**. Compare each row with experiment 1 at episode 120. These values are different from Emmanuel-NS and from Mukasa.

The observed cells come from `models/Henriette/experiment_XX/summary.json`.

Best experiment: Experiment 6, batch size 8. Last-20 average -20.50, compared with experiment 1 at -20.65. The gap is 0.15. The best game in that run was 3–21.

Worst experiment: Experiment 10, epsilon decay 0.95. Last-20 average -20.95. The best game was 2–21.

Final configuration, and why: Keep experiment 1. Every changed setting stayed near -21. A gap of 0.15, from one seed, is not enough to replace the baseline.

## Results

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0003 | 0.995 | 32 | 0.8 | 0.02 | 0.99 | Finished 120 episodes. Last-20 average -20.65. Best game reward -17, which is 4–21. The paddle scored a few points and still lost. |
| 2 | 0.001 | 0.995 | 32 | 0.8 | 0.02 | 0.99 | Finished 120 episodes. Last-20 average -20.80. Best game reward -19, which is 2–21. The higher learning rate did not help. |
| 3 | 0.00003 | 0.995 | 32 | 0.8 | 0.02 | 0.99 | Finished 120 episodes. Last-20 average -20.55. Best game reward -17, which is 4–21. Still losing every match. |
| 4 | 0.0003 | 0.9 | 32 | 0.8 | 0.02 | 0.99 | Finished 120 episodes. Last-20 average -20.80. Best game reward -17, which is 4–21. The shorter horizon did not help. |
| 5 | 0.0003 | 0.97 | 32 | 0.8 | 0.02 | 0.99 | Finished 120 episodes. Last-20 average -20.75. Best game reward -18, which is 3–21. Still losing every match. |
| 6 | 0.0003 | 0.995 | 8 | 0.8 | 0.02 | 0.99 | Finished 120 episodes. Last-20 average -20.50, a little better than experiment 1. Best game reward -18, which is 3–21. Still losing every match. |
| 7 | 0.0003 | 0.995 | 64 | 0.8 | 0.02 | 0.99 | Finished 120 episodes. Last-20 average -20.80. Best game reward -18, which is 3–21. The larger batch did not help. |
| 8 | 0.0003 | 0.995 | 32 | 0.4 | 0.02 | 0.99 | Finished 120 episodes. Last-20 average -20.75. Best game reward -19, which is 2–21. Starting with less exploration did not help. |
| 9 | 0.0003 | 0.995 | 32 | 0.8 | 0.15 | 0.99 | Finished 120 episodes. Last-20 average -20.85. Best game reward -17, which is 4–21. |
| 10 | 0.0003 | 0.995 | 32 | 0.8 | 0.02 | 0.95 | Finished 120 episodes. Last-20 average -20.95, the worst row. Best game reward -19, which is 2–21. Faster decay did not help. |
