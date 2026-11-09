import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12  # monthly exercise dates, no time-0
    
    # training_paths shape: (N_paths, n_time_points)
    # n_time_points = n_exercise + 1 = 13 (t=0, 1/12, ..., 1)
    N, n_tp = training_paths.shape
    dt = T / n_exercise
    discount = np.exp(-r * dt)
    
    # V[i, path] = value at exercise date i for path
    # Indices: 0..n_exercise
    # V[n_exercise] is maturity (col n_exercise)
    # V[i] is value at col i (for i=1..n_exercise)
    # We store V as (n_exercise+1, N)
    V = np.zeros((n_exercise + 1, N))
    
    # Maturity: must exercise if in the money
    V[n_exercise] = np.maximum(K - training_paths[:, n_exercise], 0)
    
    # Store thresholds for the policy
    # thresholds[i] corresponds to exercise date i+1 (col i+1)
    # thresholds[0] -> col 1, thresholds[10] -> col 11, thresholds[11] -> col 12
    thresholds = np.zeros(n_exercise)
    
    # Use a log-space grid for better resolution of put payoffs
    # Finer grid for more precise threshold determination
    n_grid = 500
    # Grid from small positive value to well above K
    S_min_grid = max(0.5, K * 0.005)
    S_max_grid = K * 6.0
    log_S_min = np.log(S_min_grid)
    log_S_max = np.log(S_max_grid)
    log_grid = np.linspace(log_S_min, log_S_max, n_grid)
    S_grid = np.exp(log_grid)
    
    # Precompute exercise payoffs on the grid
    exercise_payoff_grid = np.maximum(K - S_grid, 0)
    
    # Helper: smooth continuation values onto query points using Gaussian kernel
    def smooth_to_grid(S_data, y_data, S_query, bandwidth_factor=0.06):
        """Estimate E[y|S] on grid using local Gaussian weighting"""
        n_data = len(S_data)
        if n_data == 0:
            return np.zeros_like(S_query)
        
        # Adaptive bandwidth based on data spread
        S_range = S_data.max() - S_data.min()
        if S_range < 1e-10:
            return np.full_like(S_query, y_data.mean())
        bandwidth = bandwidth_factor * S_range
        
        cont_grid = np.zeros_like(S_query)
        
        # Process in chunks to avoid memory issues
        chunk_size = 256
        for i in range(0, len(S_query), chunk_size):
            chunk_end = min(i + chunk_size, len(S_query))
            S_chunk = S_query[i:chunk_end]
            
            # Compute Gaussian weights
            diff = (S_chunk[:, None] - S_data[None, :]) / bandwidth
            weights = np.exp(-0.5 * diff ** 2)
            
            # Weighted mean
            weight_sums = weights.sum(axis=1)
            weighted_vals = (weights * y_data[None, :]).sum(axis=1)
            
            valid = weight_sums > 1e-15
            cont_grid[i:chunk_end][valid] = weighted_vals[valid] / weight_sums[valid]
            cont_grid[i:chunk_end][~valid] = y_data.mean()
        
        return cont_grid
    
    for k in range(n_exercise - 1, 0, -1):
        if time.time() > deadline - 20:
            # Use simple heuristic for remaining steps
            for j in range(k, 0, -1):
                thresholds[j-1] = K * 0.8
            break
        
        S_k = training_paths[:, k]
        y = discount * V[k + 1]
        
        # Estimate continuation values on grid using Gaussian smoothing
        cont_on_grid = smooth_to_grid(S_k, y, S_grid, bandwidth_factor=0.06)
        cont_on_grid = np.maximum(cont_on_grid, 0)
        
        # Compute V[k] on training paths for next iteration
        cont_values = smooth_to_grid(S_k, y, S_k, bandwidth_factor=0.06)
        cont_values = np.maximum(cont_values, 0)
        exercise_payoff = np.maximum(K - S_k, 0)
        V[k] = np.maximum(exercise_payoff, cont_values)
        
        # Determine threshold from the grid: largest S where exercise >= continuation
        # Use a small tolerance to avoid numerical noise
        diff = exercise_payoff_grid - cont_on_grid
        should_exercise_grid = diff >= -1e-10
        if np.any(should_exercise_grid):
            last_true_idx = np.where(should_exercise_grid)[0][-1]
            thresholds[k-1] = S_grid[last_true_idx]
        else:
            thresholds[k-1] = 0.0
    
    # Final polish: refine thresholds using local interpolation around exercise boundary
    for k in range(n_exercise - 1, 0, -1):
        if time.time() > deadline - 5:
            break
        
        S_k = training_paths[:, k]
        y = discount * V[k + 1]
        
        # Get continuation and exercise payoff on a finer local grid around the threshold
        thresh_current = thresholds[k-1]
        if thresh_current <= 0:
            continue
        
        # Create a fine local grid around the current threshold estimate
        local_width = 0.05 * K  # 5% of K
        S_local_min = max(0.1, thresh_current - local_width)
        S_local_max = thresh_current + local_width
        n_local = 100
        S_local = np.linspace(S_local_min, S_local_max, n_local)
        
        # Estimate continuation on local grid
        cont_local = smooth_to_grid(S_k, y, S_local, bandwidth_factor=0.03)
        cont_local = np.maximum(cont_local, 0)
        exercise_local = np.maximum(K - S_local, 0)
        
        # Find crossing point: where exercise - cont changes sign
        diff_local = exercise_local - cont_local
        sign_changes = np.where(np.diff(np.sign(diff_local)))[0]
        
        if len(sign_changes) > 0:
            # Use the last sign change (highest S where we switch to not exercising)
            idx = sign_changes[-1]
            # Linear interpolation for more precise threshold
            s1, s2 = S_local[idx], S_local[idx+1]
            d1, d2 = diff_local[idx], diff_local[idx+1]
            if abs(d2 - d1) > 1e-15:
                threshold_refined = s1 - d1 * (s2 - s1) / (d2 - d1)
                # Clamp to valid range
                threshold_refined = np.clip(threshold_refined, S_local_min, S_local_max)
                thresholds[k-1] = threshold_refined
    
    # Set threshold for the last exercise date (maturity)
    thresholds[n_exercise - 1] = K
    
    # Create policy object
    class BermudanPutPolicy:
        def __init__(self, thresholds, n_exercise, K):
            self.thresholds = thresholds
            self.n_exercise = n_exercise
            self.K = K
        
        def exercise(self, current_state, time_step):
            # current_state: array of stock prices at current time
            # time_step: index into the time grid (integer)
            # Return boolean vector: True if exercise
            try:
                if isinstance(time_step, (int, np.integer)):
                    idx = int(time_step)
                else:
                    # If time_step is a float time, convert to index
                    idx = int(np.round(time_step * 12))
                
                # Exercise dates are at indices 1..n_exercise
                if idx < 1 or idx > self.n_exercise:
                    return np.zeros(len(current_state), dtype=bool)
                
                # Map to thresholds index
                thresh = self.thresholds[idx - 1]
                
                # Exercise if S <= threshold
                return current_state <= thresh
            except:
                return np.zeros(len(current_state), dtype=bool)
    
    return BermudanPutPolicy(thresholds, n_exercise, K)
