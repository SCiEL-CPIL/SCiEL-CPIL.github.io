import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    # training_paths shape: (N, M) where M includes time 0 through T
    # 12 monthly exercise dates, no time-0 exercise
    # So exercise dates are at t=1/12, 2/12, ..., 12/12
    N, M = training_paths.shape
    
    # Determine exercise date indices
    # Assuming paths are sampled at equal intervals including t=0
    # M points from t=0 to t=T, so dt = T/(M-1)
    # Exercise dates at t_k = k*T/12 for k=1,...,12
    # Index for t_k: k*(M-1)/12
    exercise_indices = []
    for k in range(1, 13):
        idx = int(round(k * (M - 1) / 12.0))
        idx = min(idx, M - 1)
        exercise_indices.append(idx)
    
    # Time points
    t_vals = np.linspace(0, T, M)
    
    # For each exercise date, estimate continuation value as function of S
    # Using polynomial regression on training paths
    # Continuation value at time t for state S: E[discounted future payoff | S_t = S, no exercise yet]
    # We approximate by assuming no early exercise (European-style)
    
    thresholds = np.zeros(12)  # Critical stock price for each exercise date
    
    for i, idx in enumerate(exercise_indices):
        if time.time() > deadline - 10:
            break
            
        S_now = training_paths[:, idx]
        
        # Future payoff if not exercised: discounted max(K - S_future, 0)
        # Take the max over future exercise dates (European-style approximation)
        future_indices = exercise_indices[i+1:]
        if len(future_indices) == 0:
            # Last exercise date: continuation = 0, always exercise if K > S
            thresholds[i] = K
            continue
        
        # For each future exercise date, compute discounted payoff
        # Discount factor from t_idx to t_future
        t_now = t_vals[idx]
        
        # Use the last future date for simplicity (or max over all)
        # Actually, for a put, the optimal European-style continuation is the max over future dates
        # But that's complex. Let's use a simpler heuristic:
        # Estimate E[discounted (K - S_future)^+ | S_now] using regression
        
        # Take the nearest future exercise date
        next_idx = future_indices[0]
        t_next = t_vals[next_idx]
        dt = t_next - t_now
        discount = np.exp(-r * dt)
        
        S_future = training_paths[:, next_idx]
        payoff_future = np.maximum(K - S_future, 0)
        cont_value = discount * payoff_future
        
        # Fit polynomial: cont_value ~ f(S_now)
        # Use degree 3 polynomial
        degree = 3
        # Normalize S_now for better conditioning
        S_min, S_max = S_now.min(), S_now.max()
        S_range = S_max - S_min + 1e-10
        S_norm = (S_now - S_min) / S_range
        
        # Fit
        try:
            coeffs = np.polyfit(S_norm, cont_value, degree)
            # Find threshold where K - S = poly(S)
            # K - S = c0*S^3 + c1*S^2 + c2*S + c3
            # c0*S^3 + c1*S^2 + (c2+1)*S + (c3-K) = 0
            poly_coeffs = np.array([coeffs[0], coeffs[1], coeffs[2] + 1.0, coeffs[3] - K])
            roots = np.roots(poly_coeffs)
            # Find real roots in [S_min, S_max]
            real_roots = roots[np.isreal(roots)].real
            valid_roots = real_roots[(real_roots >= S_min) & (real_roots <= S_max)]
            
            if len(valid_roots) > 0:
                # The threshold is the largest root below which we exercise
                # Actually, we want the boundary where immediate = continuation
                # For a put, we exercise when S is low
                # The relevant root is the one where S < threshold means exercise
                thresholds[i] = np.max(valid_roots)
            else:
                # No root in range, use heuristic
                thresholds[i] = K * 0.7
        except:
            thresholds[i] = K * 0.7
    
    # Create policy object
    class BermudanPutPolicy:
        def __init__(self, thresholds, exercise_indices, t_vals, K):
            self.thresholds = thresholds
            self.exercise_indices = exercise_indices
            self.t_vals = t_vals
            self.K = K
        
        def exercise(self, current_state, time_step):
            # current_state: array of stock prices at current time
            # time_step: index into the time grid
            # Return boolean vector: True if exercise
            try:
                # Map time_step to exercise index
                # time_step is an integer index
                # Find which exercise date this corresponds to
                # exercise_indices are the indices in the path
                if time_step not in self.exercise_indices:
                    return np.zeros(len(current_state), dtype=bool)
                
                ex_idx = self.exercise_indices.index(time_step)
                threshold = self.thresholds[ex_idx]
                
                # Exercise if S < threshold
                return current_state < threshold
            except:
                return np.zeros(len(current_state), dtype=bool)
    
    return BermudanPutPolicy(thresholds, exercise_indices, t_vals, K)
# EVOLVE-BLOCK-END
