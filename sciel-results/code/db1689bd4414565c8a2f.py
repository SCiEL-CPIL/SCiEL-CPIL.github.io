import numpy as np
import time
from scipy.special import roots_hermite

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12  # monthly exercise dates, no time-0

    training_paths = np.asarray(training_paths, dtype=float)
    N, n_tp = training_paths.shape
    dt = T / n_exercise
    discount = np.exp(-r * dt)

    # ---- Build a dense log-space price grid ----
    # Use a wide range to capture tail contributions for early exercise dates
    s_min = max(training_paths.min() * 0.5, 1e-4)
    s_max = max(training_paths.max() * 1.5, 2.0 * K)

    # 4096 points for high resolution
    n_grid = 4096
    log_edges = np.linspace(np.log(s_min), np.log(s_max), n_grid + 1)
    log_centers = 0.5 * (log_edges[:-1] + log_edges[1:])
    grid = np.exp(log_centers)  # (n_grid,)

    # ---- Gauss-Hermite quadrature for E[ f(S_{t+dt}) | S_t = s ] ----
    # Risk-neutral GBM: log(S_{t+dt}/s) ~ N(mu, s2), mu = (r - 0.5*sigma^2)*dt
    mu = (r - 0.5 * sigma * sigma) * dt
    s2 = sigma * sigma * dt

    # 64 points for quadrature
    n_q = 64
    x, w = roots_hermite(n_q)
    # Transform standard normal z -> log-return u = mu + sqrt(s2)*z ; z = sqrt(2)*x
    u = mu + np.sqrt(2.0 * s2) * x
    qw = w / np.sqrt(np.pi)

    # Precompute exp(u) for speed
    exp_u = np.exp(u)
    qw_col = qw.reshape(1, -1)

    # ---- Backward induction on the grid ----
    V = np.zeros((n_exercise + 1, n_grid))
    # At maturity (index n_exercise), value is intrinsic
    V[n_exercise] = np.maximum(K - grid, 0.0)

    def continuation_on_grid(V_next):
        # S_next = grid[:, None] * exp_u[None, :]
        # Shape: (n_grid, n_q)
        S_next = np.outer(grid, exp_u)
        log_S_next = np.log(S_next)

        # Interpolate V_next onto S_next (log-space)
        # log_centers is sorted
        idx = np.searchsorted(log_centers, log_S_next, side='right') - 1
        idx = np.clip(idx, 0, n_grid - 2)

        c0 = V_next[idx]
        c1 = V_next[idx + 1]
        lc0 = log_centers[idx]
        lc1 = log_centers[idx + 1]

        # Linear interpolation in log-space
        t = (log_S_next - lc0) / (lc1 - lc0)
        t = np.clip(t, 0.0, 1.0)

        interp = c0 + t * (c1 - c0)

        # Weighted sum over quadrature
        return np.sum(interp * qw_col, axis=1)

    # Iterate backwards from n_exercise-1 to 1
    # Note: Exercise dates are 1..12. Index 0 is t=0 (no exercise).
    # V[k] is value at date k.
    for k in range(n_exercise - 1, 0, -1):
        if time.time() > deadline:
            # Emergency break
            break

        # Continuation value: discount * E[V[k+1](S_{t+dt}) | S_t]
        cont_grid = discount * continuation_on_grid(V[k + 1])

        # Exercise value
        ex_grid = np.maximum(K - grid, 0.0)

        # Optimal value is max(Exercise, Continuation)
        V[k] = np.maximum(ex_grid, cont_grid)

    # ---- Refinement: Use training paths to adjust thresholds via path-based Monte Carlo ----
    # For each exercise date k (1..n_exercise-1), estimate the optimal threshold
    # by looking at where exercise is optimal on the training paths
    thresholds = np.zeros(n_exercise)

    for k in range(1, n_exercise):
        if time.time() > deadline - 5:
            # Not enough time, use grid-based threshold
            # Find threshold from grid: largest S where exercise >= continuation
            ex_grid = np.maximum(K - grid, 0.0)
            cont_grid = discount * continuation_on_grid(V[k + 1])
            should_exercise = ex_grid >= cont_grid
            if np.any(should_exercise):
                last_idx = np.where(should_exercise)[0][-1]
                thresholds[k - 1] = grid[last_idx]
            else:
                thresholds[k - 1] = 0.0
            continue

        # Get stock prices at time k for training paths
        S_k = training_paths[:, k]

        # Interpolate continuation value onto training paths
        log_S_k = np.log(S_k)
        idx_k = np.searchsorted(log_centers, log_S_k, side='right') - 1
        idx_k = np.clip(idx_k, 0, n_grid - 2)

        c0 = V[k + 1][idx_k]
        c1 = V[k + 1][idx_k + 1]
        lc0 = log_centers[idx_k]
        lc1 = log_centers[idx_k + 1]
        t_k = (log_S_k - lc0) / (lc1 - lc0)
        t_k = np.clip(t_k, 0.0, 1.0)
        cont_k = c0 + t_k * (c1 - c0)
        cont_k = discount * cont_k

        # Exercise payoff
        ex_k = np.maximum(K - S_k, 0.0)

        # Determine where exercise is optimal
        should_exercise = ex_k >= cont_k

        if np.any(should_exercise):
            # Threshold is the maximum S where exercise is optimal
            thresholds[k - 1] = S_k[should_exercise].max()
        else:
            thresholds[k - 1] = 0.0

    # Set threshold for the last exercise date (maturity)
    thresholds[n_exercise - 1] = K

    # Create policy object
    class BermudanPutPolicy:
        def __init__(self, grid, log_centers, V, n_grid, K, n_exercise, thresholds):
            self.grid = grid
            self.log_centers = log_centers
            self.V = V
            self.n_grid = n_grid
            self.K = K
            self.n_exercise = n_exercise
            self.thresholds = thresholds

        def exercise(self, current_state, time_step):
            current_state = np.asarray(current_state, dtype=float)
            try:
                if isinstance(time_step, (int, np.integer)):
                    idx = int(time_step)
                else:
                    idx = int(np.round(time_step * 12))

                if idx < 1 or idx > self.n_exercise:
                    return np.zeros(len(current_state), dtype=bool)

                # Maturity: exercise if in the money
                if idx == self.n_exercise:
                    return current_state < self.K

                # Use threshold-based decision
                thresh = self.thresholds[idx - 1]
                return current_state <= thresh

            except Exception:
                return np.zeros(len(current_state), dtype=bool)

    return BermudanPutPolicy(grid, log_centers, V, n_grid, K, n_exercise, thresholds)
