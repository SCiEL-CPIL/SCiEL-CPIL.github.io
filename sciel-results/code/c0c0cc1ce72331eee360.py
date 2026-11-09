"""Bermudan put exercise policy via Monte Carlo backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 130.0

    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)

    dt = T / n_exercise
    exercise_times = np.arange(1, n_exercise + 1) * dt
    n_paths = training_paths.shape[0]
    disc = np.exp(-r * dt)

    S_last = training_paths[:, -1]
    V_cont = np.maximum(K - S_last, 0.0)
    thresholds = np.zeros(n_exercise)

    n_bins = 300

    for i in range(n_exercise - 2, -1, -1):
        if time.time() > deadline - 15.0:
            break

        S_i = training_paths[:, i]

        S_min = np.percentile(S_i, 0.1)
        S_max = np.percentile(S_i, 99.9)
        if S_max <= S_min:
            S_max = S_min + 1.0
        bin_edges = np.linspace(S_min, S_max, n_bins + 1)
        bin_indices = np.digitize(S_i, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        disc_V = disc * V_cont
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        np.add.at(bin_sums, bin_indices, disc_V)
        np.add.at(bin_counts, bin_indices, 1)

        bin_counts_safe = np.maximum(bin_counts, 1)
        bin_means = bin_sums / bin_counts_safe

        valid = bin_counts > 0
        if valid.sum() >= 2:
            all_idx = np.arange(n_bins)
            bin_means = np.interp(all_idx, all_idx[valid], bin_means[valid])
        else:
            bin_means[:] = disc_V.mean() if n_paths > 0 else 0.0

        cont_est = bin_means[bin_indices]
        intrinsic = np.maximum(K - S_i, 0.0)

        sort_idx = np.argsort(S_i)
        S_sorted = S_i[sort_idx]
        intr_sorted = intrinsic[sort_idx]
        cont_sorted = cont_est[sort_idx]

        ex_mask = intr_sorted >= cont_sorted
        if np.any(ex_mask):
            thresholds[i] = S_sorted[np.where(ex_mask)[0][-1]]
        else:
            thresholds[i] = 0.0

        V_cont = np.where(intrinsic >= cont_est, intrinsic, cont_est)

    # Fill unfilled thresholds from the end
    for i in range(n_exercise - 2, -1, -1):
        if thresholds[i] == 0.0:
            thresholds[i] = thresholds[i + 1]
    if thresholds[-1] == 0.0:
        thresholds[-1] = K

    # Enforce non-decreasing thresholds (put boundary increases toward maturity)
    for i in range(n_exercise - 2, -1, -1):
        if thresholds[i] > thresholds[i + 1]:
            thresholds[i] = thresholds[i + 1]

    class ExercisePolicy:
        def __init__(self, thresholds, n_exercise, exercise_times):
            self.thresholds = thresholds
            self.n_exercise = n_exercise
            self.exercise_times = exercise_times

        def exercise(self, current_state, time):
            diffs = np.abs(self.exercise_times - time)
            idx = int(np.argmin(diffs))
            if diffs[idx] > 1e-6:
                return np.zeros(len(current_state), dtype=bool)
            return current_state <= self.thresholds[idx]

    return ExercisePolicy(thresholds, n_exercise, exercise_times)
# EVOLVE-BLOCK-END
