# Hyperparameter experiments

Member name: Mukasa

Environment: ALE/Pong-v5

Each row changes **one** value from experiment 1. The seed stays 1. Every run is **120 episodes**. Compare each row with experiment 1 at episode 120. These values are different from Emmanuel-NS and from Henriette.

The observed cells come from `models/Mukasa/experiment_XX/summary.json`.

Best experiment: Experiment 5, gamma 0.99. Last-20 average -20.05, compared with experiment 1 at -20.25. The gap is 0.20. The best game in that run was 3–21. Experiment 3 had the best single game, 5–21, with a last-20 average of -20.10.

Worst experiment: Experiment 8, epsilon start 0.3. Last-20 average -20.90. The best game was 2–21.

Final configuration, and why: Keep experiment 1. Every changed setting stayed near -21. A gap of 0.20, from one seed, is not enough to replace the baseline.

## Results

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.00025 | 0.95 | 64 | 1.0 | 0.1 | 0.995 | Finished 120 episodes. Last-20 average -20.25. Best game reward -17, which is 4–21. The paddle scored a few points and still lost. |
| 2 | 0.0005 | 0.95 | 64 | 1.0 | 0.1 | 0.995 | Finished 120 episodes. Last-20 average -20.50. Best game reward -18, which is 3–21. The higher learning rate did not help. |
| 3 | 0.00005 | 0.95 | 64 | 1.0 | 0.1 | 0.995 | Finished 120 episodes. Last-20 average -20.10. Best game reward -16, which is 5–21. Still losing every match. |
| 4 | 0.00025 | 0.9 | 64 | 1.0 | 0.1 | 0.995 | Finished 120 episodes. Last-20 average -20.55. Best game reward -18, which is 3–21. The shorter horizon did not help. |
| 5 | 0.00025 | 0.99 | 64 | 1.0 | 0.1 | 0.995 | Finished 120 episodes. Last-20 average -20.05, a little better than experiment 1. Best game reward -18, which is 3–21. Still losing every match. |
| 6 | 0.00025 | 0.95 | 16 | 1.0 | 0.1 | 0.995 | Finished 120 episodes. Last-20 average -20.50. Best game reward -17, which is 4–21. The smaller batch did not help. |
| 7 | 0.00025 | 0.95 | 128 | 1.0 | 0.1 | 0.995 | Finished 120 episodes. Last-20 average -20.30. Best game reward -17, which is 4–21. The larger batch did not help. |
| 8 | 0.00025 | 0.95 | 64 | 0.3 | 0.1 | 0.995 | Finished 120 episodes. Last-20 average -20.90, the worst row. Best game reward -19, which is 2–21. Starting with less exploration hurt. |
| 9 | 0.00025 | 0.95 | 64 | 1.0 | 0.01 | 0.995 | Finished 120 episodes. Last-20 average -20.65. Best game reward -18, which is 3–21. |
| 10 | 0.00025 | 0.95 | 64 | 1.0 | 0.1 | 0.98 | Finished 120 episodes. Last-20 average -20.85. Best game reward -18, which is 3–21. Faster decay did not help. |
