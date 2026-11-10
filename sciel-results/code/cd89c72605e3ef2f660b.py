"""Bermudan exercise-policy research: structural re-design using quantile-binning policy."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize_scalar

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using quantile-binning on training paths.
    Returns an object with exercise(current_state, time) -> boolean vector.
    """
    start_time = time.time()
    deadline = 150.0  # seconds wall-clock limit from evaluator

    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    num_steps = 12  # 12 monthly exercise dates

    # Handle training_paths shape
    if training_paths.ndim == 1:
        training_paths = training_paths.reshape(1, -1)

    N, total_cols = training_paths.shape

    # Determine which columns are exercise dates
    # If total_cols == num_steps + 1, index 0 is t=0, indices 1..12 are exercise dates
    # If total_cols == num_steps, indices 0..11 are exercise dates t=1..12
    if total_cols == num_steps + 1:
        s0 = training_paths[:, 0]  # initial spot (not used for exercise)
        exercise_paths = training_paths[:, 1:]  # shape (N, 12)
    elif total_cols == num_steps:
        exercise_paths = training_paths[:, :]  # shape (N, 12)
    else:
        # Fallback: take last num_steps columns
        exercise_paths = training_paths[:, -num_steps:]

    # Time grid: t_i = i * T / num_steps for i = 1, ..., 12
    dt = T / num_steps
    discount_factors = [np.exp(-r * i * dt) for i in range(1, num_steps + 1)]

    # Intrinsic value at each exercise date
    intrinsic = np.maximum(K - exercise_paths, 0.0)  # shape (N, 12)

    # For each exercise date, we'll learn a threshold on the current spot price.
    # Policy: exercise if S_t < threshold_t
    # We'll use quantile-based binning to find good thresholds.

    # Strategy: For each exercise date i, find a threshold that maximizes
    # the estimated continuation value vs exercise value.
    # We approximate the continuation value using the training paths.

    # For a Bermudan put, the optimal exercise policy is:
    # Exercise at time t_i if S_{t_i} < threshold_i
    # The thresholds can be found via dynamic programming on the training paths.

    # Approach: Use the training paths to estimate the continuation value.
    # For each path and each time step, compute the "exercise value" and
    # an estimate of the "continuation value".

    # Continuation value estimate: discount the average future intrinsic value
    # from the remaining exercise dates, but this is a crude approximation.
    # Better: use the fact that for a put, the continuation value is higher
    # when the spot is lower (more time value) but also depends on volatility.

    # Let's use a simpler approach: for each time step, find the threshold
    # by comparing exercise payoff to a Monte Carlo estimate of continuation.

    # For each exercise date i (0-indexed in our arrays, corresponding to t_{i+1}):
    # - Exercise value: intrinsic[:, i]
    # - Continuation value: estimate using paths that don't exercise at i
    #   This is tricky because we don't know which paths would exercise.

    # Alternative: Use the binomial tree reference as a guide, but adapt
    # thresholds based on training path distribution.

    # Let's use a quantile-based approach:
    # For each time step, divide the spot prices into quantiles and find
    # the optimal exercise region.

    # Actually, let's try a direct approach:
    # For each time step i, we want to find threshold_i such that
    # we exercise if S <= threshold_i.
    # The optimal threshold can be found by dynamic programming backward.

    # Backward induction on training paths:
    # V_N(S) = max(K - S, 0)  # at final exercise date
    # V_i(S) = max(K - S, discount * E[V_{i+1}(S_{i+1}) | S_i = S])

    # To estimate E[V_{i+1}(S_{i+1}) | S_i = S], we can use nearest-neighbor
    # or kernel regression on the training paths.

    # For simplicity and speed, let's use a grid-based approach:
    # 1. For each time step, create a grid of spot prices.
    # 2. Estimate the continuation value on this grid using training paths.
    # 3. Find the threshold where exercise value = continuation value.

    # Given time constraints, let's use a simpler heuristic:
    # The optimal exercise threshold for a Bermudan put is typically between
    # 0 and K. We'll search for thresholds that maximize the estimated value.

    # Let's use a Monte Carlo approach with the training paths:
    # For a candidate set of thresholds, simulate the exercise policy
    # on the training paths and compute the total discounted payoff.

    # To speed up, we'll use a coarse-to-fine search.

    # First, let's establish a baseline: exercise at the last date only.
    # This gives value = mean(discount_factor[-1] * intrinsic[:, -1])

    # Now, let's try to improve by adding exercise opportunities.

    # For each time step from last to first, find the optimal threshold.

    # We'll use a simple grid search for each threshold.

    # Time check
    if time.time() - start_time > deadline - 10:
        # Return a simple policy: exercise only at the last date
        class SimplePolicy:
            def exercise(self, current_state, time_idx):
                if time_idx >= num_steps - 1:
                    return current_state < K
                return np.zeros(len(current_state), dtype=bool)
        return SimplePolicy()

    # For each time step, find the optimal threshold
    # We'll search over a grid of thresholds from 0 to K

    # To be efficient, we'll use the training paths to estimate the value
    # for each candidate threshold.

    # For a given set of thresholds [th_1, ..., th_12], the exercise policy is:
    # At time i, exercise if S_i <= th_i
    # The total payoff is sum over paths of sum over i of discount_i * intrinsic_i * exercised_i

    # To optimize, we'll do backward induction with estimated continuation values.

    # Estimate continuation value using training paths:
    # For time step i, given S_i, estimate E[discount * V_{i+1}(S_{i+1})]
    # where V_{i+1}(S) = max(K - S, discount * E[...])

    # Let's discretize the state space and do dynamic programming.

    # Create a grid of spot prices
    all_spots = exercise_paths.flatten()
    min_s = max(0.0, np.percentile(all_spots, 1))
    max_s = min(2 * K, np.percentile(all_spots, 99))

    # Use a moderate grid size
    n_grid = 200
    grid = np.linspace(min_s, max_s, n_grid)

    # Initialize value function at the last exercise date
    # V[last](S) = max(K - S, 0)
    V = np.maximum(K - grid, 0.0)  # shape (n_grid,)

    # Thresholds from last to first
    thresholds = [None] * num_steps

    # Backward induction
    for i in range(num_steps - 1, -1, -1):
        if time.time() - start_time > deadline - 5:
            break

        # Estimate the continuation value at time i for each grid point
        # Continuation = discount * E[V_{i+1}(S_{i+1}) | S_i = s]

        # For each grid point s at time i, find the expected V_{i+1} at time i+1
        # given that S_i = s.

        # Use the training paths: for paths where S_i is close to s,
        # compute the average V_{i+1}(S_{i+1}).

        # To be efficient, we'll use a kernel density approach.
        # For each grid point, compute the weighted average of V_{i+1}(S_{i+1})
        # using weights based on proximity of S_i to the grid point.

        # This is O(n_grid * N) which might be slow for large N.
        # Let's use a simpler approach: bin the training paths by S_i.

        # Bin the S_i values into n_grid bins
        s_i = exercise_paths[:, i]  # shape (N,)

        # For each grid point, find the average V_{i+1}(S_{i+1}) for paths
        # whose S_i is in the corresponding bin.

        # Create bins
        bin_edges = np.linspace(min_s, max_s, n_grid + 1)
        bin_indices = np.digitize(s_i, bin_edges) - 1  # shape (N,)
        bin_indices = np.clip(bin_indices, 0, n_grid - 1)

        # For each bin, compute the average V_{i+1}(S_{i+1})
        s_next = exercise_paths[:, i + 1] if i + 1 < num_steps else None

        if i + 1 < num_steps:
            # Compute V_{i+1}(S_{i+1}) for each path
            # Interpolate S_{i+1} onto the grid
            v_next = np.interp(s_next, grid, V)  # shape (N,)

            # For each bin, compute the average
            continuation_bin = np.zeros(n_grid)
            counts = np.zeros(n_grid)
            for b in range(n_grid):
                mask = bin_indices == b
                if np.any(mask):
                    continuation_bin[b] = np.mean(v_next[mask])
                    counts[b] = np.sum(mask)

            # Discount
            disc = np.exp(-r * dt)
            continuation_value = disc * continuation_bin

            # For bins with no data, use 0 or interpolate
            # Let's interpolate to fill gaps
            valid = counts > 0
            if np.any(~valid):
                # Interpolate
                continuation_value[~valid] = np.interp(
                    grid[~valid], grid[valid], continuation_value[valid]
                )
        else:
            # This shouldn't happen since we start from i = num_steps - 1
            continuation_value = np.zeros(n_grid)

        # Now, the value at time i for each grid point is:
        # V_i(s) = max(K - s, continuation_value)
        intrinsic_grid = np.maximum(K - grid, 0.0)
        V = np.maximum(intrinsic_grid, continuation_value)

        # Find the threshold: the point where intrinsic = continuation
        # Exercise if intrinsic >= continuation, i.e., K - s >= continuation_value
        # s <= K - continuation_value
        # So threshold_i = max(0, min(K, K - continuation_value)) at the crossing point

        # Find where intrinsic_grid >= continuation_value
        exercise_mask = intrinsic_grid >= continuation_value
        if np.any(exercise_mask):
            # The threshold is the largest s where we exercise
            # i.e., the rightmost point in the exercise region
            # Since intrinsic is decreasing in s and continuation is roughly flat or increasing,
            # the exercise region should be [0, threshold]
            # Find the largest index where exercise_mask is True
            # But we need to be careful: the exercise region might not be contiguous
            # For a put, it should be contiguous from 0

            # Find the boundary
            # Look for the last transition from True to False
            diffs = np.diff(exercise_mask.astype(int))
            transitions = np.where(diffs == -1)[0]
            if len(transitions) > 0:
                # The threshold is at grid[transitions[0] + 1]
                # But we want the largest s where we still exercise
                # So threshold = grid[transitions[0]]
                thresholds[i] = float(grid[transitions[0]])
            else:
                # All exercise or none
                if exercise_mask[0]:
                    thresholds[i] = float(grid[-1])  # exercise always
                else:
                    thresholds[i] = 0.0  # never exercise
        else:
            thresholds[i] = 0.0  # never exercise

    # If we didn't compute all thresholds due to time, fill in with reasonable defaults
    for i in range(num_steps):
        if thresholds[i] is None:
            # Use a simple heuristic: threshold decreases as we get closer to expiration
            # At the last date, threshold = K (exercise if in the money)
            # At earlier dates, threshold is lower (more time value)
            # A rough approximation: threshold_i = K * (1 - i / num_steps)
            thresholds[i] = K * (1.0 - i / num_steps)

    # Time check
    if time.time() - start_time > deadline - 2:
        # Return the policy with the thresholds we have
        pass

    # Create the policy object
    class BermudanPolicy:
        def __init__(self, thresholds, num_steps):
            self.thresholds = thresholds
            self.num_steps = num_steps

        def exercise(self, current_state, time_idx):
            """
            current_state: array of spot prices at the current time
            time_idx: integer index of the exercise date (0-based, 0 = first exercise date)
            Returns: boolean array, True if exercise
            """
            if time_idx < 0 or time_idx >= self.num_steps:
                return np.zeros(len(current_state), dtype=bool)
            threshold = self.thresholds[time_idx]
            return current_state <= threshold

    return BermudanPolicy(thresholds, num_steps)

# EVOLVE-BLOCK-END
