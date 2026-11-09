"""Rewritten program for Bermudan exercise-policy research."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)

    # Time grid: 12 monthly exercise dates (no time-0 exercise)
    dt = T / n_steps

    # Convert to numpy array if needed
    training_paths = np.asarray(training_paths, dtype=np.float64)
    n_paths = training_paths.shape[0]

    # Use a fine grid of spot prices for the value function
    # Focus on the relevant range for a put: from near 0 to slightly above K
    # Use a logarithmic grid for better resolution at low prices
    s_min = 1.0
    s_max = 1.5 * K
    n_grid = 4096

    # Create grid: use log-spaced grid for better resolution at low S
    log_s_min = np.log(s_min)
    log_s_max = np.log(s_max)
    grid = np.exp(np.linspace(log_s_min, log_s_max, n_grid))

    # Backward induction
    # V[t](S) = value at time t for spot S
    # At exercise time t_i, V[i](S) = max(K - S, discount * E[V[i+1](S_{i+1}) | S_i = S])

    # Initialize terminal value (at t_12)
    # V[12](S) = max(K - S, 0)  # European put payoff
    V_next = np.maximum(K - grid, 0.0)

    # Store exercise decisions for each time step
    # exercise_decision[t_idx, grid_idx] = True if exercise at that grid point
    exercise_decisions = {}

    # Discount factor
    discount = np.exp(-r * dt)

    # GBM parameters for conditional expectation
    # S_{t+1} | S_t = S ~ S * exp((r - sigma^2/2) * dt + sigma * sqrt(dt) * Z)
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    # Use Gauss-Hermite quadrature for the conditional expectation
    # E[f(S * exp(drift + vol * Z))] where Z ~ N(0,1)
    # = E[f(S * exp(drift + vol * sqrt(2) * Y))] where Y ~ N(0, 0.5) ... no
    # Actually, use standard Gauss-Hermite: E[g(Z)] for Z ~ N(0,1)
    # We can use scipy's Gauss-Hermite nodes and weights
    # But for speed, use a fixed set of nodes

    # Gauss-Hermite quadrature with 64 points
    # Using the standard Gauss-Hermite quadrature
    # E[h(Z)] = sum w_i * h(x_i) / sqrt(pi) where x_i, w_i are GH nodes/weights
    # But scipy doesn't directly provide GH in the way we need easily.
    # Let's use a simple Monte Carlo approach with many points, or
    # use a deterministic quadrature.

    # Actually, for speed, let's use a fixed set of quantiles of the normal distribution
    # and use trapezoidal rule or similar.

    # Better: use the fact that we can precompute the transition kernel
    # For each grid point S_i, compute E[V_next(S_{i+1}) | S_i]
    # S_{i+1} = S_i * exp(drift + vol * Z)

    # Use Gauss-Hermite quadrature with 64 points
    # The GH quadrature approximates E[f(Z)] = (1/sqrt(pi)) * sum w_i * f(x_i)
    # where x_i and w_i are the GH nodes and weights

    # Precompute GH nodes and weights
    # For n=64, we can use numpy.polynomial.hermite.hermgauss
    n_gh = 64
    gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(n_gh)
    # Scale: Z = sqrt(2) * Y where Y ~ N(0, 1/2) ... no
    # Standard GH: E[f(Z)] for Z~N(0,1) is approximated by
    # sum w_i * f(sqrt(2) * x_i) / sqrt(pi)
    # Wait, the standard GH quadrature is for E[f(Y)] where Y~N(0, 1/2)
    # E[f(Y)] = sum w_i * f(x_i)
    # If Z = sqrt(2) * Y, then Y = Z / sqrt(2)
    # E[f(Z)] = E[f(sqrt(2) * Y)] = sum w_i * f(sqrt(2) * x_i)

    # So: E[f(Z)] ≈ sum_i w_i * f(sqrt(2) * x_i)
    # But this is for E[f(Z)] where the weight is 1/sqrt(2*pi) * exp(-z^2/2)
    # The standard GH is for weight exp(-y^2)
    # So we need to be careful.

    # Let's use a simpler approach: fixed quantile-based quadrature
    # Use 1000 equally spaced quantiles of the standard normal
    n_quad = 256
    # Use symmetric quantiles for better accuracy
    # For Z ~ N(0,1), use quantiles from -4 to 4
    z_nodes = np.linspace(-4.0, 4.0, n_quad)
    # Weights: use the PDF values normalized
    z_weights = norm.pdf(z_nodes)
    z_weights /= z_weights.sum()

    # Precompute the exponentials for the quadrature
    # S_{i+1} = S_i * exp(drift + vol * z)
    exp_factors = np.exp(drift + vol * z_nodes)  # shape: (n_quad,)

    # Backward induction from t_11 down to t_1
    for t_idx in range(n_steps - 1, 0, -1):
        if time.time() > deadline - 10.0:
            break

        # Compute continuation value at each grid point
        # For each grid point S_i, compute E[V_next(S_i * exp(drift + vol * Z))]
        # = sum_j z_weights[j] * V_next(S_i * exp_factors[j])

        # Vectorize: for each grid point, compute the weighted sum
        # grid has shape (n_grid,), exp_factors has shape (n_quad,)
        # We want: cont[j, i] = V_next(grid[i] * exp_factors[j])
        # This is a (n_quad, n_grid) matrix, which might be too large
        # n_quad=256, n_grid=4096 -> 1M elements, which is fine

        # Compute the product grid[i] * exp_factors[j]
        # Use broadcasting: grid[:, None] * exp_factors[None, :]
        # Shape: (n_grid, n_quad)
        next_spots = grid[:, None] * exp_factors[None, :]  # (n_grid, n_quad)

        # Interpolate V_next at next_spots
        # V_next is defined on 'grid', so we need to interpolate
        # Use np.interp for each row, but that's slow
        # Instead, use a vectorized approach

        # Clip next_spots to the grid range for interpolation
        next_spots_clipped = np.clip(next_spots, grid[0], grid[-1])

        # Interpolate V_next at next_spots_clipped
        # For each row i, we need to interpolate V_next at next_spots[i, :]
        # This is tricky to vectorize directly with np.interp

        # Alternative: use the fact that V_next is piecewise linear
        # We can compute the interpolation indices and weights

        # For each next_spot, find the two surrounding grid points
        # idx = searchsorted(grid, next_spot)
        # weight = (next_spot - grid[idx-1]) / (grid[idx] - grid[idx-1])
        # V = (1-weight) * V_next[idx-1] + weight * V_next[idx]

        # Vectorized approach:
        # For the (n_grid, n_quad) array of next_spots
        idx = np.searchsorted(grid, next_spots_clipped)
        idx = np.clip(idx, 1, n_grid - 1)
        # idx is the upper index, idx-1 is the lower index
        s_lower = grid[idx - 1]  # (n_grid, n_quad)
        s_upper = grid[idx]  # (n_grid, n_quad)
        # Weights
        denom = s_upper - s_lower
        denom = np.where(denom == 0, 1.0, denom)  # Avoid division by zero
        w = (next_spots_clipped - s_lower) / denom
        w = np.clip(w, 0.0, 1.0)

        # Interpolated values
        V_lower = V_next[idx - 1]  # (n_grid, n_quad)
        V_upper = V_next[idx]  # (n_grid, n_quad)
        V_interp = (1.0 - w) * V_lower + w * V_upper  # (n_grid, n_quad)

        # Weighted sum over quadrature points
        # cont[i] = sum_j z_weights[j] * V_interp[i, j]
        cont_values = V_interp @ z_weights  # (n_grid,)

        # Discount
        cont_val = discount * cont_values

        # Exercise value at each grid point
        exercise_val = np.maximum(K - grid, 0.0)

        # Optimal decision: exercise if exercise_val >= cont_val
        exercise_at_grid = exercise_val >= cont_val

        # Store the decision
        exercise_decisions[t_idx] = exercise_at_grid

        # Current value function: max of exercise and continuation
        V_current = np.maximum(exercise_val, cont_val)

        # Update for next iteration
        V_next = V_current

    # Now we have exercise_decisions for each time step
    # Create the exercise policy object

    class ExercisePolicy:
        def __init__(self, grid, exercise_decisions, n_steps, K):
            self.grid = grid
            self.exercise_decisions = exercise_decisions
            self.n_steps = n_steps
            self.K = K

        def exercise(self, current_state, time):
            """
            current_state: array of spot prices for each path
            time: integer time index (1-based, corresponding to exercise date)
            Returns: boolean array indicating whether to exercise
            """
            if time < 1 or time > self.n_steps:
                return np.zeros(len(current_state), dtype=bool)

            if time not in self.exercise_decisions:
                # If no decision stored, use simple rule: exercise if S < K
                return current_state < self.K

            exercise_at_grid = self.exercise_decisions[time]

            # For each spot, find if we should exercise based on the grid-based decision
            # Use the grid indices
            current_state = np.asarray(current_state, dtype=np.float64)
            grid_indices = np.searchsorted(self.grid, current_state)
            grid_indices = np.clip(grid_indices, 0, len(self.grid) - 1)

            # Get the decision for each grid point
            decisions = exercise_at_grid[grid_indices]

            return decisions

    policy = ExercisePolicy(grid, exercise_decisions, n_steps, K)
    return policy
# EVOLVE-BLOCK-END
