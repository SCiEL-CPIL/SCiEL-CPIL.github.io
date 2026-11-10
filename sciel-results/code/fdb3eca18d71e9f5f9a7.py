"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class ExercisePolicy:
    def __init__(self, boundaries, n_exercises, T):
        self.boundaries = boundaries  # list of boundary values for each exercise date
        self.n_exercises = n_exercises
        self.T = T
        self.dt = T / n_exercises
    
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
        
        for i in range(n):
            t = time[i]
            s = current_state[i]
            if s <= 0:
                continue
            # Find which exercise date we're at
            # Exercise dates are at dt, 2*dt, ..., n_exercises*dt
            # t/dt gives the exercise index (1-based for first exercise)
            idx = int(np.round(t / self.dt)) - 1
            if idx < 0:
                idx = 0
            if idx >= self.n_exercises:
                idx = self.n_exercises - 1
            
            boundary = self.boundaries[idx]
            if s <= boundary:
                result[i] = True
        
        return result

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option using backward induction
    on a fine grid to compute optimal exercise boundaries.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin within 120s limit
    
    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    
    # Monthly exercise dates
    dt = T / n_exercises
    
    # Build a fine grid for the stock price
    # Use logarithmic spacing for better resolution near the critical region
    S_min = 0.01
    S_max = 500.0
    n_grid = 8000
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
    
    # Use Gauss-Hermite quadrature for the expectation
    n_quad = 200
    x_quad, w_quad = np.polynomial.hermite.hermgauss(n_quad)
    z_quad = np.sqrt(2) * x_quad
    w_quad_norm = w_quad / np.sqrt(np.pi)
    log_mult = np.exp(mu + sd * z_quad)  # shape (n_quad,)
    
    # Work backwards from the last exercise date
    for step in range(n_exercises - 1, -1, -1):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Compute continuation value: discount * E[V_next(S_next)]
        # S_next = S * exp(mu + sd * Z)
        cont_values = np.zeros(n_grid)
        chunk_size = 800
        for start_idx in range(0, n_grid, chunk_size):
            end_idx = min(start_idx + chunk_size, n_grid)
            S_chunk = S_grid[start_idx:end_idx]  # (chunk,)
            S_next_chunk = S_chunk[:, np.newaxis] * log_mult[np.newaxis, :]  # (chunk, n_quad)
            S_next_chunk = np.clip(S_next_chunk, S_min, S_max)
            V_at_Snext = np.interp(S_next_chunk, S_grid, V_next)  # (chunk, n_quad)
            cont_values[start_idx:end_idx] = np.sum(w_quad_norm[np.newaxis, :] * V_at_Snext, axis=1)
        
        # Continuation value with discount
        cont_value = discount * cont_values
        
        # Optimal exercise: max(immediate exercise, continuation)
        exercise_value = np.maximum(K - S_grid, 0.0)
        V_current = np.maximum(exercise_value, cont_value)
        
        # Find the boundary: largest S where exercise is optimal
        diff = exercise_value - cont_value
        # Find sign change from positive to negative
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
        else:
            # No sign change: either always exercise or never exercise
            if np.all(diff >= 0):
                boundary = S_grid[-1]
            else:
                boundary = S_grid[0]
        
        boundaries[step] = boundary
        V_next = V_current
    
    return ExercisePolicy(boundaries, n_exercises, T)
# EVOLVE-BLOCK-END
