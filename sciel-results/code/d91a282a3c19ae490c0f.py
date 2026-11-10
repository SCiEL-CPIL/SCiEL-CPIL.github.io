"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

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
        
        for i in range(n):
            t = time[i]
            s = current_state[i]
            if s <= 0:
                continue
            # Find which exercise period we're in
            # Exercise dates are at T/12, 2T/12, ..., 12T/12
            # We need to find the most recent exercise date <= t
            idx = int(np.floor(t * self.n_exercises))
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
    deadline = start_time + 100.0  # Safety margin within 120s limit
    
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
    S_min = 1.0
    S_max = 300.0
    n_grid = 4000
    S_grid = np.exp(np.linspace(np.log(S_min), np.log(S_max), n_grid))
    
    # Precompute discount factors
    discount = np.exp(-r * dt)
    
    # Backward induction: start from last exercise date
    # V[i] = continuation value at exercise date i
    # We store the value function on the grid
    
    # At the last exercise date (T), the value is max(K - S, 0) since no further exercise
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Store boundaries for each exercise date
    boundaries = [0.0] * n_exercises
    
    # Work backwards from the last exercise date
    for step in range(n_exercises - 1, -1, -1):
        # Current time is (step+1)*dt
        # Continuation value: discount expected future value
        # For GBM, we can use the risk-neutral distribution
        
        # Compute expected continuation value using quadrature or direct computation
        # For a put, we need E[max(K - S_next, 0) | S_current]
        
        # Use the fact that S_next = S_current * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
        # where Z ~ N(0,1)
        
        # For each S in S_grid, compute E[max(K - S_next, 0)]
        # This is equivalent to pricing a European put with one period to expiry
        
        # Using Black-Scholes formula for the continuation value
        # European put price with time to expiry = dt
        S = S_grid[:, np.newaxis]  # (n_grid, 1)
        
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * dt) / (sigma * np.sqrt(dt))
        d2 = d1 - sigma * np.sqrt(dt)
        
        # European put price
        from scipy.stats import norm
        cdf = norm.cdf
        pdf = norm.pdf
        
        # Put price = K*exp(-r*dt)*N(-d2) - S*N(-d1)
        cont_value = K * np.exp(-r * dt) * cdf(-d2) - S * cdf(-d1)
        cont_value = cont_value.flatten()
        
        # Optimal exercise: max(immediate exercise, continuation)
        exercise_value = np.maximum(K - S_grid, 0.0)
        V_current = np.maximum(exercise_value, discount * cont_value)
        
        # Find the boundary: largest S where exercise is optimal
        # Exercise when S <= boundary
        exercise_mask = (exercise_value >= discount * cont_value)
        
        if np.any(exercise_mask):
            # Find the largest S where exercise is optimal
            # The boundary is where exercise_value = discount * cont_value
            # Find the transition point
            diff = exercise_value - discount * cont_value
            # Find where diff changes sign from positive to negative
            sign_change = np.where(np.diff(np.sign(diff)) < 0)[0]
            if len(sign_change) > 0:
                idx = sign_change[0]
                # Linear interpolation for better accuracy
                S1 = S_grid[idx]
                S2 = S_grid[idx + 1]
                d1_val = diff[idx]
                d2_val = diff[idx + 1]
                if d2_val != d1_val:
                    boundary = S1 + (S2 - S1) * d1_val / (d1_val - d2_val)
                else:
                    boundary = S1
            else:
                # All exercise or no exercise
                if np.all(exercise_mask):
                    boundary = S_grid[-1]
                else:
                    boundary = S_grid[0]
        else:
            boundary = S_grid[0]
        
        boundaries[step] = boundary
        
        # Update V_next for next iteration
        V_next = V_current
    
    # Check deadline
    if time.time() > deadline:
        # Return what we have
        pass
    
    return ExercisePolicy(boundaries, n_exercises)
# EVOLVE-BLOCK-END
