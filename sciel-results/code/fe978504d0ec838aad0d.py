"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator, interp1d

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
        current_state = np.asarray(current_state)
        time = np.asarray(time)
        
        if np.ndim(time) == 0:
            time = np.full_like(current_state, time)
        
        n = len(current_state)
        result = np.zeros(n, dtype=bool)
        
        # Vectorized version
        idx = np.floor(time * self.n_exercises).astype(int)
        idx = np.clip(idx, 0, self.n_exercises - 1)
        
        for i in range(n):
            s = current_state[i]
            if s <= 0:
                continue
            boundary = self.boundaries[idx[i]]
            if s <= boundary:
                result[i] = True
        
        return result

def _compute_boundary(exercise_value, cont_value, S_grid):
    """Find the optimal exercise boundary with smooth interpolation."""
    diff = exercise_value - cont_value
    
    # Find sign changes from positive to negative
    sign_change = np.where(np.diff(np.sign(diff)) < 0)[0]
    
    if len(sign_change) == 0:
        if np.all(diff >= 0):
            return S_grid[-1]
        else:
            return S_grid[0]
    
    idx = sign_change[0]
    S1 = S_grid[idx]
    S2 = S_grid[idx + 1]
    d1 = diff[idx]
    d2 = diff[idx + 1]
    
    if d2 != d1:
        boundary = S1 + (S2 - S1) * d1 / (d1 - d2)
    else:
        boundary = S1
    
    return boundary

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
    
    # Build a fine grid for the stock price
    # Use logarithmic spacing for better resolution
    S_min = 0.01
    S_max = 500.0
    n_grid = 10000
    S_grid = np.exp(np.linspace(np.log(S_min), np.log(S_max), n_grid))
    
    # Precompute discount factor for one step
    discount = np.exp(-r * dt)
    
    # Backward induction: start from last exercise date
    # At the last exercise date (T), the value is max(K - S, 0) since no further exercise
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Store boundaries for each exercise date
    boundaries = [0.0] * n_exercises
    
    # Precompute log-normal transition parameters
    mu = (r - sigma**2 / 2) * dt
    sd = sigma * np.sqrt(dt)
    
    # Use higher-order Gauss-Hermite quadrature for more accurate expectations
    n_quad = 120
    x_quad, w_quad = np.polynomial.hermite.hermgauss(n_quad)
    z_quad = np.sqrt(2) * x_quad
    w_quad_norm = w_quad / np.sqrt(np.pi)
    log_mult = np.exp(mu + sd * z_quad)  # shape (n_quad,)
    
    # Create smooth interpolator for the value function
    V_interp = PchipInterpolator(S_grid, V_next)
    
    # Work backwards from the last exercise date
    for step in range(n_exercises - 1, -1, -1):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Compute continuation value: discount * E[V_next(S_next)]
        # S_next = S * exp(mu + sd * Z)
        cont_values = np.zeros(n_grid)
        chunk_size = 400
        for start_idx in range(0, n_grid, chunk_size):
            end_idx = min(start_idx + chunk_size, n_grid)
            S_chunk = S_grid[start_idx:end_idx]  # (chunk,)
            S_next_chunk = S_chunk[:, np.newaxis] * log_mult[np.newaxis, :]  # (chunk, n_quad)
            S_next_chunk = np.clip(S_next_chunk, S_min, S_max)
            # Use smooth interpolation
            V_at_Snext = V_interp(S_next_chunk)  # (chunk, n_quad)
            cont_values[start_idx:end_idx] = np.sum(w_quad_norm[np.newaxis, :] * V_at_Snext, axis=1)
        
        # Continuation value with discount
        cont_value = discount * cont_values
        
        # Optimal exercise: max(immediate exercise, continuation)
        exercise_value = np.maximum(K - S_grid, 0.0)
        V_current = np.maximum(exercise_value, cont_value)
        
        # Find the boundary
        boundary = _compute_boundary(exercise_value, cont_value, S_grid)
        
        boundaries[step] = boundary
        V_next = V_current
        
        # Update interpolator for next iteration
        V_interp = PchipInterpolator(S_grid, V_next)
    
    # High-precision polish pass: refine each boundary using a local fine grid
    if time.time() < deadline - 15.0:
        for step in range(n_exercises - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            
            # Get the coarse boundary estimate
            coarse_boundary = boundaries[step]
            
            # Create a local fine grid around the boundary
            local_width = max(0.5, 0.02 * coarse_boundary)
            local_min = max(0.01, coarse_boundary - local_width)
            local_max = min(500.0, coarse_boundary + local_width)
            n_local = 2000
            S_local = np.exp(np.linspace(np.log(local_min), np.log(local_max), n_local))
            
            # Recompute continuation value on local grid using the V_next from this step
            # We need to recompute V_next for this step
            if step == n_exercises - 1:
                V_next_local = np.maximum(K - S_local, 0.0)
            else:
                # Recompute continuation on local grid
                cont_local = np.zeros(n_local)
                chunk_size = 500
                for start_idx in range(0, n_local, chunk_size):
                    end_idx = min(start_idx + chunk_size, n_local)
                    S_chunk = S_local[start_idx:end_idx]
                    S_next_chunk = S_chunk[:, np.newaxis] * log_mult[np.newaxis, :]
                    S_next_chunk = np.clip(S_next_chunk, S_min, S_max)
                    V_at_Snext = V_interp(S_next_chunk)
                    cont_local[start_idx:end_idx] = np.sum(w_quad_norm[np.newaxis, :] * V_at_Snext, axis=1)
                V_next_local = discount * cont_local
            
            exercise_value_local = np.maximum(K - S_local, 0.0)
            refined_boundary = _compute_boundary(exercise_value_local, V_next_local, S_local)
            
            if refined_boundary > 0:
                boundaries[step] = refined_boundary
    
    return ExercisePolicy(boundaries, n_exercises)
# EVOLVE-BLOCK-END
