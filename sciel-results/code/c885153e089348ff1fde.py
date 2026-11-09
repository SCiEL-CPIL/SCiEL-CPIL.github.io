"""Bermudan put exercise policy: backward induction with refined discretization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction.
    
    Parameters:
    -----------
    training_paths : array-like, shape (n_paths, n_time_steps)
        Simulated spot price paths. n_time_steps = 13 (t=0..12).
    contract : dict
        Contract parameters: K, r, sigma, T, n_steps (12 exercise dates).
        
    Returns:
    --------
    An object with method exercise(current_state, time) -> boolean array.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Extract parameters from contract
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12  # monthly exercise dates, no time-0

    # training_paths shape: (N_paths, n_time_points)
    # n_time_points = n_exercise + 1 = 13 (t=0, 1/12, ..., 1)
    N, n_tp = training_paths.shape
    dt = T / n_exercise
    discount = np.exp(-r * dt)

    # V[i, path] = value at exercise date i for path
    # V[12] is maturity, V[1..11] are intermediate exercise dates
    V = np.zeros((n_exercise + 1, N))

    # Maturity: must exercise if in the money
    V[n_exercise] = np.maximum(K - training_paths[:, n_exercise], 0)

    # Store thresholds for the policy
    # thresholds[i] corresponds to exercise date i+1 (col i+1)
    # thresholds[0] -> col 1, thresholds[10] -> col 11, thresholds[11] -> col 12
    thresholds = np.zeros(n_exercise)

    # Number of bins for piecewise linear regression
    # Using 100 bins for better resolution, but keeping it fast
    n_bins = 100

    for k in range(n_exercise - 1, 0, -1):
        if time.time() > deadline - 15:
            # Use simple heuristic for remaining steps
            for j in range(k, 0, -1):
                thresholds[j-1] = K * 0.8
            break

        S_k = training_paths[:, k]
        y = discount * V[k + 1]

        # Create adaptive bins based on the distribution of S_k
        # Use quantile-based binning to handle non-uniform distributions
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(S_k, quantiles)

        # Ensure unique edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            bin_edges = np.array([S_k.min(), (S_k.min() + S_k.max()) / 2, S_k.max()])

        # Assign each point to a bin
        bin_indices = np.searchsorted(bin_edges, S_k, side='right') - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)

        # Compute mean continuation value for each bin
        n_actual_bins = len(bin_edges) - 1
        cont_per_bin = np.zeros(n_actual_bins)
        counts = np.zeros(n_actual_bins)

        for b in range(n_actual_bins):
            mask = bin_indices == b
            if np.any(mask):
                cont_per_bin[b] = np.mean(y[mask])
                counts[b] = np.sum(mask)

        # Interpolate continuation values for all points
        # Use linear interpolation between bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # For each S_k, find the two nearest bin centers and interpolate
        cont_values = np.zeros(N)
        for i in range(N):
            s = S_k[i]
            # Find the bin
            idx = np.searchsorted(bin_centers, s, side='right') - 1
            idx = np.clip(idx, 0, len(bin_centers) - 2)

            # Linear interpolation
            s0 = bin_centers[idx]
            s1 = bin_centers[idx + 1]
            c0 = cont_per_bin[idx]
            c1 = cont_per_bin[idx + 1]

            if s1 > s0:
                t = (s - s0) / (s1 - s0)
                t = np.clip(t, 0, 1)
                cont_values[i] = c0 + t * (c1 - c0)
            else:
                cont_values[i] = c0

        # Ensure non-negative continuation values
        cont_values = np.maximum(cont_values, 0)

        exercise_payoff = np.maximum(K - S_k, 0)
        V[k] = np.maximum(exercise_payoff, cont_values)

        # Determine threshold: largest S where exercise_payoff >= cont_value
        order = np.argsort(S_k)
        S_sorted = S_k[order]
        ex_sorted = exercise_payoff[order]
        cont_sorted = cont_values[order]

        should_exercise = ex_sorted >= cont_sorted
        if np.any(should_exercise):
            last_true = np.where(should_exercise)[0][-1]
            thresholds[k-1] = S_sorted[last_true]
        else:
            thresholds[k-1] = 0.0

    class Policy:
        def __init__(self, thresholds, n_exercise, K):
            self.thresholds = thresholds
            self.n_exercise = n_exercise
            self.K = K

        def exercise(self, current_state, time):
            if isinstance(time, (int, np.integer)):
                idx = int(time)
                if idx < 1 or idx > self.n_exercise:
                    return np.zeros(len(current_state), dtype=bool)
                idx = idx - 1
            else:
                k = int(np.round(time * 12))
                if k < 1 or k > self.n_exercise:
                    return np.zeros(len(current_state), dtype=bool)
                idx = k - 1

            # At maturity, always exercise if in the money
            if idx == self.n_exercise - 1:
                return current_state < self.K

            thresh = self.thresholds[idx]
            return current_state <= thresh

    return Policy(thresholds, n_exercise, K)
# EVOLVE-BLOCK-END
