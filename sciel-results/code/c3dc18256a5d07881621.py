import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Parameters from contract
    K = float(contract.get('K', 100.0))
    r = float(contract.get('r', 0.05))
    sigma = float(contract.get('sigma', 0.2))
    T = float(contract.get('T', 1.0))
    n_ex = int(contract.get('n_exercise_dates', 12))

    # Exercise times: monthly, no time-0
    dt = T / n_ex
    exercise_times = np.arange(1, n_ex + 1) * dt

    # Risk-neutral parameters
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p_up = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Fine log-spaced grid: high resolution near the critical boundary (~K),
    # lower resolution far from it where the payoff is flat.
    S_min = max(1e-3, K * 0.005)
    S_max = K * 10.0
    n_grid = 20000
    S_grid = np.geomspace(S_min, S_max, n_grid)

    # Terminal value: European put at T
    V = np.maximum(K - S_grid, 0.0)

    # Backward induction over the 11 early exercise dates (k = 10..0).
    # The terminal date (k=11) is European (no early exercise) so V is already correct.
    # We store the optimal exercise boundary (critical price) at each early date.
    boundaries = np.full(n_ex, np.nan)

    for k in range(n_ex - 2, -1, -1):  # exercise dates 1..11 (0-indexed 0..10)
        if time.time() > deadline - 8.0:
            break

        S_up = S_grid * u
        S_down = S_grid * d

        # Interpolate continuation value from the previous (later) step
        V_up = np.interp(S_up, S_grid, V, left=0.0, right=0.0)
        V_down = np.interp(S_down, S_grid, V, left=0.0, right=0.0)

        cont = disc * (p_up * V_up + (1.0 - p_up) * V_down)

        # Intrinsic (exercise) value on the grid
        ex_val = np.maximum(K - S_grid, 0.0)

        # Optimal value: max(exercise, continue)
        V_new = np.maximum(ex_val, cont)

        # Determine the critical boundary: largest S where exercise is (weakly) optimal.
        # For a put, the exercise region is S <= boundary.
        # Find where ex_val >= cont (exercise region) and take the max S in that region.
        ex_region = ex_val >= cont - 1e-12
        if np.any(ex_region):
            S_ex = S_grid[ex_region]
            boundaries[k] = float(np.max(S_ex))
        else:
            boundaries[k] = 0.0

        V = V_new

    # High-precision polish pass: refine each boundary locally using a denser
    # local grid around the coarse estimate. This tightens the boundary to
    # sub-grid resolution, which is the dominant source of error.
    if time.time() < deadline - 5.0:
        # Recompute V from terminal for the polish (we need consistent continuation values).
        # We redo backward induction but only refine boundaries at the end.
        # Instead, we refine each boundary by re-evaluating the local region on a fine grid
        # using the already-computed value functions. Since we overwrote V, we redo the
        # backward pass storing V at each step for local refinement.
        V_steps = np.empty((n_ex - 1, n_grid))  # V after processing date k (k=0..n_ex-2)
        V = np.maximum(K - S_grid, 0.0)
        for k in range(n_ex - 2, -1, -1):
            if time.time() > deadline - 5.0:
                break
            S_up = S_grid * u
            S_down = S_grid * d
            V_up = np.interp(S_up, S_grid, V, left=0.0, right=0.0)
            V_down = np.interp(S_down, S_grid, V, left=0.0, right=0.0)
            cont = disc * (p_up * V_up + (1.0 - p_up) * V_down)
            ex_val = np.maximum(K - S_grid, 0.0)
            V_new = np.maximum(ex_val, cont)
            V_steps[k] = V_new
            V = V_new

        # Refine each boundary using a local fine grid
        for k in range(n_ex - 1):
            if time.time() > deadline - 3.0:
                break
            b0 = boundaries[k]
            if not np.isfinite(b0) or b0 <= 0:
                continue
            # Local grid around b0
            local_half = b0 * 0.05
            local_min = max(S_min, b0 - local_half)
            local_max = min(S_max, b0 + local_half)
            n_local = 2000
            S_local = np.geomspace(local_min, local_max, n_local)

            # We need cont at S_local for step k. cont depends on V from step k+1.
            # V_steps[k] is V_new after processing k. V_steps[k+1] (or terminal) is the "next" V.
            # cont at step k uses V from the later date.
            if k < n_ex - 2:
                V_later = V_steps[k + 1]
            else:
                V_later = np.maximum(K - S_grid, 0.0)  # terminal

            S_up_l = S_local * u
            S_down_l = S_local * d
            V_up_l = np.interp(S_up_l, S_grid, V_later, left=0.0, right=0.0)
            V_down_l = np.interp(S_down_l, S_grid, V_later, left=0.0, right=0.0)
            cont_l = disc * (p_up * V_up_l + (1.0 - p_up) * V_down_l)
            ex_l = np.maximum(K - S_local, 0.0)

            ex_region_l = ex_l >= cont_l - 1e-12
            if np.any(ex_region_l):
                boundaries[k] = float(np.max(S_local[ex_region_l]))

    # Fill any uncomputed boundaries (in case of early break) with a reasonable default
    for k in range(n_ex):
        if not np.isfinite(boundaries[k]):
            # Default: use a heuristic based on the grid
            boundaries[k] = K * 0.7

    # Create policy object
    class BermudanPolicy:
        def __init__(self, boundaries, exercise_times, K, r, sigma, T, n_ex):
            self.boundaries = boundaries
            self.exercise_times = exercise_times
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_ex = n_ex

        def exercise(self, current_state, time_val):
            """
            current_state: array of stock prices
            time_val: current time (scalar)
            returns: boolean array, True if exercise
            """
            current_state = np.asarray(current_state, dtype=float)

            # Find which exercise date we're at
            for k in range(self.n_ex):
                if abs(time_val - self.exercise_times[k]) < 1e-3:
                    return current_state <= self.boundaries[k]

            # Not at an exercise date, cannot exercise
            return np.zeros(len(current_state), dtype=bool)

        def __call__(self, current_state, time_val):
            return self.exercise(current_state, time_val)

    policy = BermudanPolicy(boundaries, exercise_times, K, r, sigma, T, n_ex)
    return policy
