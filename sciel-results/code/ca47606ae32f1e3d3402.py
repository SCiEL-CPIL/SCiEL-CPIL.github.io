"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator

class ExercisePolicy:
    def __init__(self, boundaries, n_exercises):
        self.boundaries = boundaries  # list of boundary values for each exercise date
        self.n_exercises = n_exercises

    def exercise(self, current_state, time):
        """
        Returns boolean vector: True if should exercise at current_state at given time.
        current_state: array of stock prices
        time: array of times (or scalar)
        """
        current_state = np.asarray(current_state, dtype=float)
        time = np.asarray(time, dtype=float)

        if np.ndim(time) == 0:
            time = np.full_like(current_state, time)

        n = len(current_state)
        result = np.zeros(n, dtype=bool)

        for i in range(n):
            t = time[i]
            s = current_state[i]
            if s <= 0:
                continue
            # Exercise dates are at T/12, 2T/12, ..., 12T/12
            idx = int(np.floor(t * self.n_exercises))
            if idx < 0:
                idx = 0
            if idx >= self.n_exercises:
                idx = self.n_exercises - 1

            boundary = self.boundaries[idx]
            if s <= boundary:
                result[i] = True

        return result

def _compute_boundaries(K, r, sigma, T, n_exercises, S_grid, deadline, n_quad, log_mult, w_quad_norm, discount, mu, sd, V_next):
    """
    Run backward induction on S_grid to produce boundaries (list of length n_exercises,
    indexed by step 0..n_exercises-1, where step i corresponds to exercise date (i+1)*dt).
    Also returns the final V_next (value at time 0+ if needed).
    """
    S_min = S_grid[0]
    S_max = S_grid[-1]
    n_grid = len(S_grid)
    boundaries = [0.0] * n_exercises

    # Precompute the polynomial interpolant of V_next in log-space for fast evaluation.
    # We interpolate V as a function of log(S) using PCHIP on the grid.
    log_S_grid = np.log(S_grid)

    # Work backwards from the last exercise date (step = n_exercises-1) down to 0
    for step in range(n_exercises - 1, -1, -1):
        if time.time() > deadline:
            # Not enough time; fill remaining boundaries with current best guess (0)
            break

        # Build interpolant for V_next in log-space
        # PCHIP is C1 continuous and shape-preserving; good for smooth-ish V
        # For speed, use a coarser subset for the interpolant if grid is large
        interp_stride = max(1, n_grid // 2000)
        idx_sub = np.arange(0, n_grid, interp_stride)
        if idx_sub[-1] != n_grid - 1:
            idx_sub = np.concatenate([idx_sub, [n_grid - 1]])
        pchip = PchipInterpolator(log_S_grid[idx_sub], V_next[idx_sub])

        # Compute continuation value: discount * E[V_next(S_next)]
        # S_next = S * exp(mu + sd * Z)
        # log(S_next) = log(S) + mu + sd * Z
        cont_values = np.zeros(n_grid)
        chunk_size = 1000
        for start_idx in range(0, n_grid, chunk_size):
            end_idx = min(start_idx + chunk_size, n_grid)
            log_S_chunk = log_S_grid[start_idx:end_idx]  # (chunk,)
            # log_S_next = log_S_chunk[:,None] + (mu + sd * z_quad)[None,:]
            log_S_next = log_S_chunk[:, np.newaxis] + (mu + sd * (np.sqrt(2) * np.polynomial.hermite.hermgauss(n_quad)[0]))[np.newaxis, :]
            # Clip log_S_next to [log(S_min), log(S_max)]
            log_S_next = np.clip(log_S_next, np.log(S_min), np.log(S_max))
            # Evaluate V_next at log_S_next using PCHIP
            V_at_Snext = pchip(log_S_next)  # (chunk, n_quad)
            cont_values[start_idx:end_idx] = np.sum(w_quad_norm[np.newaxis, :] * V_at_Snext, axis=1)

        cont_value = discount * cont_values
        exercise_value = np.maximum(K - S_grid, 0.0)
        V_current = np.maximum(exercise_value, cont_value)

        # Find the boundary: largest S where exercise is optimal
        diff = exercise_value - cont_value
        # The boundary is where diff crosses zero from positive to negative
        # Since exercise_value is decreasing in S and cont_value is also decreasing but slower,
        # diff should be positive for small S and negative for large S.
        sign_change = np.where(np.diff(np.sign(diff)) < 0)[0]
        if len(sign_change) > 0:
            idx = sign_change[0]
            S1 = S_grid[idx]
            S2 = S_grid[idx + 1]
            d1_val = diff[idx]
            d2_val = diff[idx + 1]
            if d2_val != d1_val:
                boundary = S1 + (S2 - S1) * d1_val / (d1_val - d2_val)
            else:
                boundary = S1
        elif np.all(diff >= 0):
            boundary = S_grid[-1]
        else:
            boundary = S_grid[0]

        boundaries[step] = boundary
        V_next = V_current

    return boundaries, V_next

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option using backward induction
    on a fine grid to compute optimal exercise boundaries.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin within 120s limit

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    # Monthly exercise dates
    dt = T / n_exercises

    # Precompute log-normal transition parameters
    mu = (r - sigma**2 / 2) * dt
    sd = sigma * np.sqrt(dt)

    # Use Gauss-Hermite quadrature for the expectation
    n_quad = 100
    x_quad, w_quad = np.polynomial.hermite.hermgauss(n_quad)
    z_quad = np.sqrt(2) * x_quad
    w_quad_norm = w_quad / np.sqrt(np.pi)

    # Precompute discount factor
    discount = np.exp(-r * dt)

    # Build a fine grid for the stock price
    # Use logarithmic spacing for better resolution
    S_min = 0.01
    S_max = 500.0
    n_grid = 8000
    S_grid = np.exp(np.linspace(np.log(S_min), np.log(S_max), n_grid))

    # At the last exercise date (T), the value is max(K - S, 0)
    V_next = np.maximum(K - S_grid, 0.0)

    # Run backward induction
    boundaries, _ = _compute_boundaries(K, r, sigma, T, n_exercises, S_grid, deadline, n_quad, None, w_quad_norm, discount, mu, sd, V_next)

    # If we ran out of time, boundaries may have zeros; that's acceptable but suboptimal.
    # For now, return the policy.
    return ExercisePolicy(boundaries, n_exercises)
# EVOLVE-BLOCK-END
