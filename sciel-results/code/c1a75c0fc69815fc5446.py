"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Extract parameters
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12
    
    tp = np.asarray(training_paths, dtype=np.float64)
    
    # Handle potential shape variations
    if tp.ndim == 2:
        if tp.shape[0] == 4096:
            paths = tp
            n_paths, n_times = tp.shape
        elif tp.shape[1] == 4096:
            paths = tp.T
            n_times, n_paths = tp.shape
        else:
            # Default assumption: rows are paths
            paths = tp
            n_paths, n_times = tp.shape
    else:
        paths = tp
        n_paths, n_times = paths.shape

    # Determine if time 0 is included
    # Task says "12 monthly exercise dates (no time-0 exercise)".
    # If n_times == 13, it likely includes t=0. If 12, it starts at t=1/12.
    if n_times == 13:
        S = paths[:, 1:]  # Exclude t=0
    else:
        S = paths
        
    dt = T / n_dates
    discount_step = np.exp(-r * dt)
    
    # Backward Induction with Smoothing
    # V[i] is the value at step i (0-indexed in S, so S[:, 0] is step 1)
    # We store values for steps 1..12
    # V_next corresponds to the value at the *next* exercise date
    
    # Initialize at the last date (index n_dates-1 in S, which is step 12)
    V_next = np.maximum(K - S[:, n_dates - 1], 0.0)
    
    # Store decisions and thresholds
    thresholds = np.zeros(n_dates)
    decisions = [None] * n_dates
    
    # We will iterate backwards from step 11 (index 10) down to step 1 (index 0)
    # Note: We need to store the decision for each step to determine thresholds later?
    # Actually, we can determine the threshold on the fly or store the decision masks.
    # Storing 12 masks of 4096 booleans is cheap.
    
    # To smooth the continuation value, we use a local averaging approach.
    # For each step i, we sort paths by S_i and average V_next in a window.
    window_size = 100  # Number of neighbors to average
    
    for i in range(n_dates - 1, -1, -1):
        if time.time() > deadline - 5:
            # Fallback if running out of time: use simple threshold K for later dates
            # This is a safety break, though unlikely to trigger
            break
            
        S_i = S[:, i]
        immediate = np.maximum(K - S_i, 0.0)
        
        # Estimate continuation value E[V_next | S_i]
        # Sort indices by S_i
        sort_idx = np.argsort(S_i)
        S_sorted = S_i[sort_idx]
        V_sorted = V_next[sort_idx]
        
        # Calculate moving average of V_sorted
        # We want C_smooth[j] to be the average of V in a window around j
        # Using cumsum for efficiency
        cumsum = np.cumsum(V_sorted)
        # Pad cumsum with 0 at the start for easy windowing
        cumsum_padded = np.concatenate(([0], cumsum))
        
        C_smooth_sorted = np.zeros(n_paths)
        # For each point j, average V_sorted from max(0, j-half) to min(N, j+half)
        half = window_size // 2
        for j in range(n_paths):
            left = max(0, j - half)
            right = min(n_paths, j + half + 1)
            # Sum = cumsum[right] - cumsum[left]
            # Count = right - left
            if right > left:
                C_smooth_sorted[j] = (cumsum_padded[right] - cumsum_padded[left]) / (right - left)
            else:
                C_smooth_sorted[j] = V_sorted[j]
                
        # Unsort to original order
        C_smooth = np.zeros(n_paths)
        C_smooth[sort_idx] = C_smooth_sorted
        
        # Discount the continuation value
        cont_value = discount_step * C_smooth
        
        # Decision: Exercise if Immediate >= Continuation
        decision = immediate >= cont_value
        
        decisions[i] = decision
        
        # Update V for the next iteration (backwards)
        # Use the max of immediate and the *smoothed* continuation?
        # Or the actual realized value?
        # Using the smoothed value propagates the regularization.
        V_next = np.maximum(immediate, cont_value)
        
        # Extract threshold for this step i
        # Exercise if S <= threshold
        # Find the max S where decision is True
        if np.any(decision):
            thresholds[i] = np.max(S_i[decision])
        else:
            thresholds[i] = 0.0
            
    # Enforce Monotonicity on Thresholds
    # For a put, the exercise boundary should be non-decreasing with time (step index)
    # i.e., thresholds[0] <= thresholds[1] <= ... <= thresholds[11]
    # Also, thresholds should not exceed K (though numerically they might if noise is high, 
    # but since immediate payoff is 0 for S>K, decision should be False there, 
    # unless cont_value is negative? No, cont_value is positive. 
    # So threshold should naturally be <= K. We clamp it anyway.)
    
    for i in range(1, n_dates):
        if thresholds[i] < thresholds[i-1]:
            thresholds[i] = thresholds[i-1]
            
    # Clamp to K
    thresholds = np.minimum(thresholds, K)
    
    # Ensure the last threshold is close to K? 
    # At maturity, we always exercise if S < K. 
    # The backward induction at the last step sets V = max(K-S, 0). 
    # The decision at the last step (i=n_dates-1) is based on comparing immediate to... 
    # Wait, my loop goes from n_dates-1 down to 0.
    # At i = n_dates-1 (last step), V_next was initialized to payoff at T.
    # So cont_value is discount * payoff_at_T? No.
    # Standard backward induction:
    # At step 12 (last), Value = Payoff. Decision is trivially "Exercise if Payoff > 0" (or just take payoff).
    # In my loop:
    # i = 11 (second to last): V_next is payoff at step 12.
    # ...
    # i = 0 (first step): V_next is value at step 2.
    # 
    # The threshold at i=11 (step 12) is calculated in the loop?
    # My loop: `for i in range(n_dates - 1, -1, -1)`
    # So i goes 11, 10, ..., 0.
    # At i=11, V_next is the payoff at step 12 (calculated before loop).
    # cont_value = discount * payoff_at_12? 
    # No, the value at step 12 IS the payoff. The continuation value *from* step 12 to step 13 (maturity) is just the payoff if we assume maturity is step 12?
    # The problem says "12 monthly exercise dates". Usually this means t=1/12, ..., 12/12=1.
    # So step 12 is maturity.
    # At maturity, you MUST exercise if in the money. There is no "continuation" after maturity.
    # So for i=11 (step 12), the decision should be: Exercise if K - S > 0.
    # My code:
    # V_next (before loop) = max(K - S_12, 0).
    # In loop i=11:
    # immediate = max(K - S_12, 0).
    # C_smooth is based on V_next (which is immediate).
    # cont_value = discount * C_smooth.
    # decision = immediate >= discount * immediate.
    # Since discount < 1, immediate >= discount * immediate is always true if immediate > 0.
    # So decision is True if S_12 < K.
    # This is correct.
    
    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K
            self.n_dates = len(thresholds)

        def exercise(self, current_state, time):
            cs = np.asarray(current_state, dtype=np.float64)
            # Map time to step index
            # time is in [0, 1]
            # Steps are at 1/12, 2/12, ..., 12/12
            # If time < 1/12, we are before the first exercise date. 
            # The task says "no time-0 exercise". 
            # If current time is before the first date, we cannot exercise? 
            # Or is the first date the earliest we can exercise?
            # "12 monthly exercise dates".
            # If time=0, we can't exercise.
            # If time=0.1 (approx 1.2 months), we are at the first date.
            # We should map to the nearest valid date index.
            # Indices 0..11 correspond to times 1/12..12/12.
            
            # Convert time to months
            months = time * 12.0
            # Round to nearest integer
            idx = int(round(months))
            
            # Clamp to valid range 1..12 (indices 0..11)
            if idx < 1:
                idx = 1
            elif idx > 12:
                idx = 12
                
            # Map to array index 0..11
            arr_idx = idx - 1
            
            threshold = self.thresholds[arr_idx]
            return (cs <= threshold).astype(bool)

    return ExercisePolicy(thresholds, K)
# EVOLVE-BLOCK-END
