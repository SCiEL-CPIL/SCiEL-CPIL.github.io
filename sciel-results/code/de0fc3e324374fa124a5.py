import numpy as np
import time
import math

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    # Determine number of steps
    # "12 monthly exercise dates (no time-0 exercise)"
    # training_paths shape is likely (N, 13) or (N, 12).
    # The contract implies 12 exercise opportunities.
    # Let's assume the grid covers 12 steps.
    n_exercises = 12
    dt = T / n_exercises
    
    # Construct a high-resolution binomial tree for backward induction.
    # We need a grid that covers the range of possible stock prices.
    # For a put with K=100, S0 in [90, 110], sigma=0.2, T=1.
    # Min price ~ S0 * exp(-sigma*sqrt(T) * 4) ~ 100 * exp(-0.8) ~ 45.
    # Max price ~ S0 * exp(sigma*sqrt(T) * 4) ~ 100 * exp(0.8) ~ 220.
    # To be safe, cover [0, 300] or similar.
    
    # Number of grid points. 
    # 1200 steps was mentioned as a reference diagnostic in the prompt.
    # A grid of 2000-5000 points should be fast enough in numpy.
    # Let's use 4096 points for good resolution.
    n_grid = 4096
    
    # Define grid boundaries
    S_min = 0.0
    S_max = 300.0
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # Up and down factors for binomial tree
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p_up = (math.exp(r * dt) - d) / (u - d)
    p_down = 1.0 - p_up
    
    # Pre-compute discount factor
    disc = math.exp(-r * dt)
    
    # V_next will store the continuation value at the next time step for each grid point.
    # At t=12 (maturity), the value is the intrinsic value.
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Store exercise boundaries (thresholds) for each time step.
    # exercise_boundary[t] = S_threshold such that if S <= S_threshold, exercise.
    # We store these to be used in the policy object.
    boundaries = [0.0] * n_exercises
    
    # Backward induction loop
    # t goes from 12 down to 1.
    # V_current at step t is max(Intrinsic, disc * E[V_next])
    # Note: The "next" value in the tree corresponds to the values computed in the previous iteration (t+1).
    
    for t in range(n_exercises, 0, -1):
        if time.time() > deadline:
            break
            
        # Compute expected next value
        # V_next is defined on S_grid.
        # To get E[V_next(S_{t+1}) | S_t], we need to interpolate V_next at S_t*u and S_t*d.
        
        # Linear interpolation indices
        # S_up = S_grid * u
        # S_down = S_grid * d
        
        S_up = S_grid * u
        S_down = S_grid * d
        
        # Clip to grid bounds to avoid out-of-bounds errors in searchsorted
        # For a put, if S is very high, V is 0. If S is 0, V is K*disc^...
        # Our grid covers [0, 300]. 
        # S_up might exceed 300. S_down might be < 0.
        
        # Helper function to interpolate values from grid
        # Since S_grid is linear, we can use searchsorted.
        
        # Find indices for interpolation
        # We want value at x.
        # idx = floor((x - S_min) / dx)
        dx = (S_max - S_min) / (n_grid - 1)
        
        # Vectorized interpolation
        def interpolate_values(x):
            # x is array
            # Clip x to [S_min, S_max]
            x_clipped = np.clip(x, S_min, S_max)
            
            # Calculate position
            pos = (x_clipped - S_min) / dx
            idx0 = np.floor(pos).astype(int)
            idx1 = idx0 + 1
            
            # Handle boundaries
            idx0 = np.clip(idx0, 0, n_grid - 1)
            idx1 = np.clip(idx1, 0, n_grid - 1)
            
            # Fraction
            frac = pos - np.floor(pos)
            
            # Linear interpolation
            v0 = V_next[idx0]
            v1 = V_next[idx1]
            return v0 * (1 - frac) + v1 * frac
            
        V_up = interpolate_values(S_up)
        V_down = interpolate_values(S_down)
        
        # Expected value
        V_expected = p_up * V_up + p_down * V_down
        
        # Continuation value
        V_cont = disc * V_expected
        
        # Intrinsic value
        V_intrinsic = np.maximum(K - S_grid, 0.0)
        
        # Optimal value
        V_curr = np.maximum(V_intrinsic, V_cont)
        
        # Determine exercise boundary for this time step t
        # Exercise if V_intrinsic > V_cont
        # For a put, this is S <= Threshold.
        # Find the largest S in S_grid where V_intrinsic >= V_cont.
        
        exercise_mask = V_intrinsic >= V_cont
        
        if np.any(exercise_mask):
            # Find the max index where exercise is optimal
            idx_exercise = np.where(exercise_mask)[0]
            max_idx = idx_exercise[-1]
            boundary = S_grid[max_idx]
        else:
            boundary = 0.0
            
        boundaries[t-1] = boundary
        
        # Prepare for next iteration
        V_next = V_curr
        
    # Create policy object
    class Policy:
        def __init__(self, boundaries, K):
            self.boundaries = boundaries
            self.K = K
            
        def exercise(self, current_state, time):
            S = np.asarray(current_state)
            
            if time < 1 or time > 12:
                # Fallback: exercise if ITM
                return S < self.K
            
            # Get boundary for this time step
            # boundaries[0] is for t=1, boundaries[11] is for t=12
            threshold = self.boundaries[time - 1]
            
            # Exercise if S <= threshold
            # Note: At t=12, threshold should be K (or close to it).
            # Intrinsic > Continuation (0) => S < K.
            return S <= threshold

    return Policy(boundaries, K)
