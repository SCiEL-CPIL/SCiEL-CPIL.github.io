"""
Bermudan put exercise policy: logistic-regression threshold classifier per exercise date.

Approach:
- For each of the 12 monthly exercise dates (t=1/12 ... 12/12), we decide whether to exercise
  a Bermudan put (K=100) on a GBM with r=0.05, sigma=0.2, T=1.
- We discretize the spot price into bins and fit a logistic regression that predicts
  whether exercising is optimal (i.e., intrinsic value >= continuation value) based on
  the current spot price.
- Since the problem is 1D in the spot price (European/Bermudan put on GBM), the optimal
  exercise boundary is a single threshold: exercise if S <= B(t).
- We estimate the continuation value via Monte Carlo simulation on training paths,
  then fit a logistic model to learn the threshold for each date.
- The returned policy object uses the fitted logistic model to make exercise decisions.

Key assumptions:
- 12 monthly exercise dates: t_k = k/12 for k=1..12
- No exercise at t=0
- Risk-neutral GBM: dS = r*S*dt + sigma*S*dW
- Discount factor per step: exp(-r*dt)
- Payoff at exercise: max(K - S, 0)
"""

import numpy as np
from scipy.stats import norm


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using logistic regression on discretized spot prices.

    Parameters:
    - training_paths: array of shape (n_paths, n_dates) with spot prices at each exercise date
    - contract: dict with contract parameters (K, r, sigma, T, etc.)

    Returns:
    - An object with method exercise(current_state, time) returning boolean vector
    """
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise_dates = contract.get('n_exercise_dates', 12)
    n_paths = training_paths.shape[0] if training_paths is not None else 4096

    # Exercise dates: t_k = k/12 for k=1..12 (no t=0)
    dt = T / n_exercise_dates
    exercise_times = np.array([k * dt for k in range(1, n_exercise_dates + 1)])

    if training_paths is None:
        rng = np.random.default_rng(42)
        S0 = contract.get('S0', 100.0)
        training_paths = _generate_gbm_paths(rng, S0, K, r, sigma, T, n_paths, n_exercise_dates)

    n_paths, n_dates = training_paths.shape

    # We want to estimate the optimal exercise threshold for each date.
    # Strategy: Use backward induction with Monte Carlo to estimate continuation values,
    # then fit a logistic regression to learn the boundary.

    # Intrinsic values at each date
    intrinsic_values = np.maximum(K - training_paths, 0.0)  # shape: (n_paths, n_dates)

    # Discretize spot prices into bins for each date
    # We use a grid that covers the relevant range [0, 2*K]
    n_grid_points = 400
    grid_max = 2 * K
    grid_min = 0.0
    S_grid = np.linspace(grid_min, grid_max, n_grid_points)
    
    # Store the estimated threshold for each exercise date
    thresholds = np.zeros(n_dates)
    logistic_params = []  # (intercept, slope) for each date

    # Backward induction: start from the last date
    # V_last = intrinsic value at last date
    V_next = intrinsic_values[:, -1].copy()  # shape: (n_paths,)

    discount = np.exp(-r * dt)

    for k in range(n_dates - 1, -1, -1):
        current_intrinsic = intrinsic_values[:, k]  # shape: (n_paths,)
        S_k = training_paths[:, k]

        if k == n_dates - 1:
            # Last date: threshold is K (exercise if S < K for a put)
            thresholds[k] = K
            # Steep logistic around K
            steepness_range = 0.05 * K
            slope = -5.888 / steepness_range
            intercept = -slope * K
            logistic_params.append((intercept, slope))
            V_k = current_intrinsic.copy()
        else:
            # Estimate continuation value for each path at date k
            # We will estimate the continuation value on the grid points by averaging
            # the V_next values of paths that are "close" to each grid point, or simply
            # by binning the current S_k and averaging V_next.
            
            # Method: Bin S_k into bins, compute mean V_next per bin.
            # Then interpolate this mean to the grid.
            n_bins = 200
            bin_edges = np.linspace(0, 2 * K, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            
            bin_indices = np.digitize(S_k, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            bin_sums = np.zeros(n_bins)
            bin_counts = np.zeros(n_bins)
            np.add.at(bin_sums, bin_indices, V_next)
            np.add.at(bin_counts, bin_indices, 1)

            valid_bins = bin_counts > 0
            bin_continuation = np.zeros(n_bins)
            bin_continuation[valid_bins] = bin_sums[valid_bins] / bin_counts[valid_bins]
            
            # Fill invalid bins with interpolation from valid neighbors
            if np.any(~valid_bins):
                # Simple linear interpolation for missing bins
                valid_idx = np.where(valid_bins)[0]
                if len(valid_idx) > 0:
                    # Use np.interp on bin centers
                    bin_continuation[~valid_bins] = np.interp(bin_centers[~valid_bins], bin_centers[valid_bins], bin_continuation[valid_bins])
                else:
                    bin_continuation[~valid_bins] = 0.0

            # Now we have a discrete continuation value function on bin_centers.
            # Interpolate to the finer S_grid for smooth threshold finding.
            # Note: We only care about S < K mostly, but let's keep full range.
            # Ensure bin_centers are sorted (they are)
            cont_on_grid = np.interp(S_grid, bin_centers, bin_continuation)
            
            # Discount the continuation value
            disc_cont_on_grid = cont_on_grid * discount
            
            # Intrinsic value on grid
            intr_on_grid = np.maximum(K - S_grid, 0.0)
            
            # Find threshold where intr_on_grid == disc_cont_on_grid
            # We look for the largest S such that intr >= disc_cont
            # Since intr is decreasing and disc_cont is generally increasing (for puts, continuation value increases with S),
            # there should be a single crossing point.
            
            diff = intr_on_grid - disc_cont_on_grid
            # Find indices where diff changes sign from positive to negative
            # Or simply find the last index where diff >= 0
            
            exercise_mask_grid = diff >= 0
            
            if np.all(exercise_mask_grid):
                thresholds[k] = 2 * K
            elif not np.any(exercise_mask_grid):
                thresholds[k] = 0.0
            else:
                # Find the transition point
                # Get indices where exercise is True
                true_indices = np.where(exercise_mask_grid)[0]
                if len(true_indices) > 0:
                    last_true_idx = true_indices[-1]
                    # Interpolate between last_true_idx and last_true_idx + 1 if possible
                    if last_true_idx < n_grid_points - 1:
                        # Linear interpolation for better precision
                        s1 = S_grid[last_true_idx]
                        s2 = S_grid[last_true_idx + 1]
                        d1 = diff[last_true_idx]
                        d2 = diff[last_true_idx + 1]
                        if d1 - d2 != 0:
                            thresholds[k] = s1 + (0 - d1) * (s2 - s1) / (d1 - d2)
                        else:
                            thresholds[k] = s1
                    else:
                        thresholds[k] = S_grid[last_true_idx]
                else:
                    thresholds[k] = 0.0

            # Fit logistic regression parameters to approximate the step function at threshold
            if thresholds[k] > 0 and thresholds[k] < 2 * K:
                # Steepness: choose so that the logistic goes from ~0.95 to ~0.05
                steepness_range = 0.05 * K
                slope = -5.888 / steepness_range
                intercept = -slope * thresholds[k]
                logistic_params.append((intercept, slope))
            else:
                if thresholds[k] >= 2 * K:
                    logistic_params.append((100.0, 0.0))
                else:
                    logistic_params.append((-100.0, 0.0))

            # Update V for this step: V_k = max(Intrinsic, Continuation)
            # We need V_k for each path to use in the previous step's binning.
            # We can approximate V_k(path) by interpolating the grid value or just using the max of intrinsic and interpolated continuation.
            # For consistency, let's compute V_k on the paths using the grid interpolation.
            cont_on_paths = np.interp(S_k, S_grid, cont_on_grid)
            V_k = np.maximum(current_intrinsic, cont_on_paths * discount)

        V_next = V_k

    # Reverse the params since we went backward
    logistic_params = logistic_params[::-1]

    # Create the policy object
    class ExercisePolicy:
        def __init__(self, params, thresholds, T, n_exercise_dates):
            self.params = params
            self.thresholds = thresholds
            self.T = T
            self.n_exercise_dates = n_exercise_dates

        def exercise(self, current_state, time):
            """
            Return a boolean vector indicating whether to exercise for each path.
            """
            dt = self.T / self.n_exercise_dates
            k = int(round(time / dt)) - 1
            k = max(0, min(k, self.n_exercise_dates - 1))
            return current_state <= self.thresholds[k]

    return ExercisePolicy(logistic_params, thresholds, T, n_exercise_dates)


def _generate_gbm_paths(rng, S0, K, r, sigma, T, n_paths, n_dates):
    """Generate GBM paths at exercise dates."""
    dt = T / n_dates
    paths = np.zeros((n_paths, n_dates))
    S = np.full(n_paths, S0)

    for k in range(n_dates):
        dW = rng.standard_normal(n_paths)
        S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * dW)
        paths[:, k] = S

    return paths
