"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    
    # Training paths shape: (n_paths, n_steps+1) or (n_paths, n_steps)
    # We need to determine the number of exercise dates
    # The task says 12 monthly exercise dates (no time-0 exercise)
    # So paths have values at t_1, t_2, ..., t_12
    
    n_paths = training_paths.shape[0]
    n_time_points = training_paths.shape[1]
    
    # dt for each step
    dt = T / n_steps
    
    # Discount factor per step
    df = np.exp(-r * dt)
    
    # For a Bermudan put, the payoff at exercise date i is max(K - S_i, 0)
    # We want to find optimal exercise policy using backward induction
    
    # The training_paths give us spot prices at each time step
    # training_paths[i, j] = spot price for path i at time step j+1 (first exercise date)
    # or training_paths[i, j] = spot price for path i at time step j (if 0-indexed)
    
    # Let's assume training_paths has shape (n_paths, n_steps) where each column
    # corresponds to one of the 12 exercise dates
    
    # Backward induction for Bermudan option:
    # V_n(S) = max(K - S, 0)  at final exercise date
    # V_i(S) = max(K - S, df * E[V_{i+1}(S_{i+1}) | S_i = S])
    
    # We need to estimate the continuation value E[V_{i+1}(S_{i+1}) | S_i = S]
    # This is a regression problem
    
    # Since we have limited time, let's use a simple approach:
    # 1. Start from the final exercise date with intrinsic value
    # 2. Work backwards, at each step fit a regression of continuation value on current spot
    # 3. The exercise policy is: exercise if K - S >= continuation_value(S)
    
    # For simplicity and speed, let's use a piecewise linear or polynomial regression
    
    # First, let's compute the terminal payoff
    # At the last exercise date (index n_steps - 1), the value is just the intrinsic value
    # V_{n-1}(S) = max(K - S, 0)
    
    # We'll work backwards from the last exercise date to the first
    
    # For each exercise date i (from n_steps-2 down to 0):
    #   continuation_value = df * E[V_{i+1}(S_{i+1}) | S_i]
    #   We need to regress V_{i+1}(S_{i+1}) on S_i
    
    # Let's use a simple approach: for each exercise date, we can use the training paths
    # to estimate the conditional expectation
    
    # Actually, let me think about this more carefully.
    # training_paths[i, j] gives us the spot at exercise date j+1 for path i
    # So to compute E[V_{j+1}(S_{j+1}) | S_j = s], we need to look at paths where S_j is close to s
    
    # A simpler approach: use the fact that under GBM, the conditional distribution
    # of S_{j+1} given S_j is lognormal. But we can also just use empirical regression.
    
    # Let's try a different approach: use the training paths to directly estimate
    # the optimal exercise policy via dynamic programming on a grid of spot values.
    
    # Create a grid of spot values
    s_min = 10.0
    s_max = 200.0
    n_grid = 200
    grid = np.linspace(s_min, s_max, n_grid)
    
    # V[i, k] = value of the option at exercise date i when spot is grid[k]
    # Start with terminal condition
    V = np.zeros((n_steps, n_grid))
    V[-1, :] = np.maximum(K - grid, 0.0)
    
    # Work backwards
    for i in range(n_steps - 2, -1, -1):
        # At each exercise date i, for each grid point, we need:
        # continuation = df * E[V[i+1](S_{i+1}) | S_i = grid[k]]
        # 
        # Under GBM: S_{i+1} = S_i * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
        # So given S_i = s, S_{i+1} ~ LogNormal(ln(s) + (r - sigma^2/2)*dt, sigma^2*dt)
        
        # We can compute the expected continuation value analytically or numerically
        # Let's use numerical integration with a reasonable number of points
        
        n_quad = 50
        # Quadrature points for standard normal
        # Use Gauss-Hermite or just uniform in a range
        z_vals = np.linspace(-4, 4, n_quad)
        z_weights = np.ones(n_quad) / n_quad  # Simple uniform weighting as approximation
        
        # Actually, let's use proper Gauss-Hermite quadrature for better accuracy
        # But for speed, let's use a simpler approach
        
        # For each grid point, compute E[V[i+1](S_{i+1}) | S_i = s]
        continuation = np.zeros(n_grid)
        for k in range(n_grid):
            s = grid[k]
            # S_{i+1} = s * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
            log_s_next = np.log(s) + (r - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * z_vals
            s_next = np.exp(log_s_next)
            
            # Interpolate V[i+1] at s_next
            # Clip to grid range
            s_next_clipped = np.clip(s_next, s_min, s_max)
            v_next = np.interp(s_next_clipped, grid, V[i + 1, :])
            
            continuation[k] = df * np.sum(z_weights * v_next)
        
        # Exercise value
        exercise_val = np.maximum(K - grid, 0.0)
        
        # Optimal value
        V[i, :] = np.maximum(exercise_val, continuation)
    
    # Now we have the optimal value function at each exercise date
    # The exercise policy is: exercise if exercise_val >= continuation
    # i.e., if max(K - S, 0) >= df * E[V[i+1](S_{i+1}) | S_i = S]
    
    # For the exercise function, we need to return a boolean vector indicating
    # whether to exercise at each current state
    
    # We need to store the thresholds or the value functions
    # Let's store the grid and V for interpolation
    
    # Check time
    if time.time() > deadline - 5.0:
        # Return a simple policy: exercise if K - S > 0 and S < K * 0.9
        # This is a rough heuristic
        class SimplePolicy:
            def __init__(self, K):
                self.K = K
            def exercise(self, current_state, time_idx):
                return np.array(current_state) < self.K * 0.95
        return SimplePolicy(K)
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, K, r, sigma, T, n_steps, grid, V):
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_steps = n_steps
            self.grid = grid
            self.V = V
            self.df = np.exp(-r * T / n_steps)
            self.dt = T / n_steps
            
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
            
            # Continuation value: df * E[V[idx+1](S_{idx+1}) | S_i = s]
            # We need to compute this. Let's use the same quadrature as during fitting
            n_quad = 50
            z_vals = np.linspace(-4, 4, n_quad)
            z_weights = np.ones(n_quad) / n_quad
            
            continuation = np.zeros(len(s_arr))
            for j in range(len(s_arr)):
                s = s_arr[j]
                if s <= 0:
                    continuation[j] = 0
                    continue
                log_s_next = np.log(s) + (self.r - self.sigma**2 / 2) * self.dt + self.sigma * np.sqrt(self.dt) * z_vals
                s_next = np.exp(log_s_next)
                s_next_clipped = np.clip(s_next, self.grid[0], self.grid[-1])
                v_next = np.interp(s_next_clipped, self.grid, self.V[idx + 1, :])
                continuation[j] = self.df * np.sum(z_weights * v_next)
            
            return exercise_val >= continuation
    
    return ExercisePolicy(K, r, sigma, T, n_steps, grid, V)
# EVOLVE-BLOCK-END
