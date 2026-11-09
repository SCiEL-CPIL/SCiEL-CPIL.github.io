"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    
    n_paths = training_paths.shape[0]
    n_time_points = training_paths.shape[1]
    
    # dt for each step
    dt = T / n_steps
    
    # Discount factor per step
    df = np.exp(-r * dt)
    
    # Create a grid of spot values
    # Use a wider range and more points for better accuracy
    s_min = 1.0
    s_max = 300.0
    n_grid = 500
    grid = np.linspace(s_min, s_max, n_grid)
    
    # V[i, k] = value of the option at exercise date i when spot is grid[k]
    # Start with terminal condition
    V = np.zeros((n_steps, n_grid))
    V[-1, :] = np.maximum(K - grid, 0.0)
    
    # Precompute quadrature points using Gauss-Hermite for better accuracy
    n_quad = 40
    # Gauss-Hermite nodes and weights
    z_vals, z_weights = np.polynomial.hermite.hermgauss(n_quad)
    # Scale to standard normal: x = z/sqrt(2), dx = dz/sqrt(2)
    # E[f(X)] where X ~ N(0,1) = 1/sqrt(pi) * sum w_i * f(z_i/sqrt(2))
    z_vals = z_vals / np.sqrt(2)
    z_weights = z_weights / np.sqrt(np.pi)
    
    # Work backwards
    for i in range(n_steps - 2, -1, -1):
        # At each exercise date i, for each grid point, we need:
        # continuation = df * E[V[i+1](S_{i+1}) | S_i = grid[k]]
        
        # Vectorized computation for all grid points at once
        # log_s_next[k, q] = log(grid[k]) + (r - sigma^2/2)*dt + sigma*sqrt(dt)*z_vals[q]
        log_grid = np.log(grid)  # (n_grid,)
        drift = (r - sigma**2 / 2) * dt
        vol = sigma * np.sqrt(dt)
        
        # Shape: (n_grid, n_quad)
        log_s_next = log_grid[:, np.newaxis] + drift + vol * z_vals[np.newaxis, :]
        s_next = np.exp(log_s_next)
        
        # Clip to grid range
        s_next_clipped = np.clip(s_next, s_min, s_max)
        
        # Interpolate V[i+1] at s_next for each grid point and quadrature point
        # v_next[k, q] = V[i+1](s_next[k, q])
        v_next = np.zeros_like(s_next_clipped)
        for q in range(n_quad):
            v_next[:, q] = np.interp(s_next_clipped[:, q], grid, V[i + 1, :])
        
        # Weighted sum for expected continuation
        continuation = df * np.sum(v_next * z_weights[np.newaxis, :], axis=1)
        
        # Exercise value
        exercise_val = np.maximum(K - grid, 0.0)
        
        # Optimal value
        V[i, :] = np.maximum(exercise_val, continuation)
    
    # Check time
    if time.time() > deadline - 5.0:
        # Return a simple policy
        class SimplePolicy:
            def __init__(self, K):
                self.K = K
            def exercise(self, current_state, time_idx):
                return np.array(current_state) < self.K * 0.95
        return SimplePolicy(K)
    
    # Precompute continuation values on a fine grid for each time step
    # to speed up evaluation
    # For each time step i (0 to n_steps-2), precompute continuation value on the grid
    continuation_grid = np.zeros((n_steps, n_grid))
    for i in range(n_steps - 1):
        log_grid = np.log(grid)
        drift = (r - sigma**2 / 2) * dt
        vol = sigma * np.sqrt(dt)
        log_s_next = log_grid[:, np.newaxis] + drift + vol * z_vals[np.newaxis, :]
        s_next = np.exp(log_s_next)
        s_next_clipped = np.clip(s_next, s_min, s_max)
        v_next = np.zeros_like(s_next_clipped)
        for q in range(n_quad):
            v_next[:, q] = np.interp(s_next_clipped[:, q], grid, V[i + 1, :])
        continuation_grid[i, :] = df * np.sum(v_next * z_weights[np.newaxis, :], axis=1)
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, K, r, sigma, T, n_steps, grid, V, continuation_grid, df, dt):
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_steps = n_steps
            self.grid = grid
            self.V = V
            self.continuation_grid = continuation_grid
            self.df = df
            self.dt = dt
            
        def exercise(self, current_state, time_idx):
            # time_idx: which exercise date (0-indexed, 0 to n_steps-1)
            # current_state: array of spot prices
            
            # Determine the exercise date index
            if isinstance(time_idx, (int, np.integer)):
                idx = int(time_idx)
            else:
                # time_idx is a float in [0, T]
                idx = int(time_idx / self.dt)
                idx = min(idx, self.n_steps - 1)
                idx = max(idx, 0)
            
            # At the last exercise date, always exercise if in the money
            if idx == self.n_steps - 1:
                return np.array(current_state) < self.K
            
            # For earlier dates, compare exercise value to continuation value
            s_arr = np.array(current_state, dtype=float)
            
            # Exercise value
            exercise_val = np.maximum(self.K - s_arr, 0.0)
            
            # Continuation value: interpolate from precomputed grid
            # Clip s_arr to grid range
            s_clipped = np.clip(s_arr, self.grid[0], self.grid[-1])
            continuation = np.interp(s_clipped, self.grid, self.continuation_grid[idx, :])
            
            return exercise_val >= continuation
    
    return ExercisePolicy(K, r, sigma, T, n_steps, grid, V, continuation_grid, df, dt)
# EVOLVE-BLOCK-END
