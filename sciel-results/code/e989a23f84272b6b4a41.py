"""Bermudan exercise-policy: binomial-tree boundary-based policy."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Parse contract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # Exercise times: monthly, no time-0, so at T/12, 2T/12, ..., 12T/12
    exercise_times = np.linspace(T / n_dates, T, n_dates)
    
    # Build binomial tree with fine steps for accurate boundaries
    # Use enough steps to get smooth boundaries
    n_steps = 1200
    dt = T / n_steps
    
    # Binomial parameters
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # We need to find the exercise boundary at each of the n_dates exercise dates
    # The exercise dates correspond to specific tree steps
    # Exercise date i is at time t_i = (i+1)*T/n_dates
    # Corresponding tree step: step_i = round(t_i / dt) = round(n_steps * (i+1)/n_dates)
    
    # Build the tree backward to find optimal exercise boundaries
    # At each node, value = max(immediate_payoff, discounted_expected_continuation)
    
    # We'll store the continuation value and exercise value at each step
    # For a put: immediate payoff = max(K - S, 0)
    
    # Start at maturity (step n_steps): value = max(K - S, 0) for all S
    # We need to track the boundary: the highest S where exercise is optimal
    
    # Instead of storing full tree, we can compute boundaries via backward induction
    # on a sufficiently fine grid
    
    # Use a log-spaced grid for stock prices to cover the relevant range
    # For a put with K=100, relevant S range is roughly [K*0.1, K*3]
    S_min = K * 0.05
    S_max = K * 5.0
    n_grid = 2000  # grid points for stock price
    
    # Log-spaced grid
    S_grid = np.exp(np.linspace(np.log(S_min), np.log(S_max), n_grid))
    
    # Value at maturity (step n_steps)
    V = np.maximum(K - S_grid, 0.0)
    
    # Boundary at maturity: exercise if S < K (put payoff positive)
    # boundary = K at maturity
    
    # We need boundaries at exercise dates. Let's do backward induction step by step
    # and record the boundary at each exercise date step.
    
    # Map exercise dates to tree steps
    exercise_steps = []
    for i in range(n_dates):
        t_i = exercise_times[i]
        step_i = int(round(t_i / dt))
        # Ensure step is within [1, n_steps]
        step_i = max(1, min(n_steps, step_i))
        exercise_steps.append(step_i)
    
    # We'll do backward induction from n_steps down to 1
    # At each step, update V and record boundary if this step is an exercise date
    
    # For efficiency, we only need to record at exercise steps
    # Create a set of exercise steps for quick lookup
    exercise_step_set = set(exercise_steps)
    
    # Boundaries indexed by exercise date index
    boundaries = [None] * n_dates
    step_to_date = {step: i for i, step in enumerate(exercise_steps)}
    
    # Backward induction
    # V is the continuation value at the current step (before deciding to exercise)
    # Actually, V should be the optimal value at each step
    
    # Let's be more careful:
    # At step j (time t_j = j*dt), the optimal value is:
    # V_j(S) = max( (K-S)^+ , exp(-r*dt) * [p * V_{j+1}(S*u) + (1-p) * V_{j+1}(S*d)] )
    # if step j is an exercise date, otherwise:
    # V_j(S) = exp(-r*dt) * [p * V_{j+1}(S*u) + (1-p) * V_{j+1}(S*d)]
    
    # We start with V_{n_steps}(S) = (K-S)^+
    
    # For each step from n_steps-1 down to 1:
    #   Compute continuation = exp(-r*dt) * [p * V_{j+1}(S*u) + (1-p) * V_{j+1}(S*d)]
    #   If step j is an exercise date:
    #     V_j(S) = max((K-S)^+, continuation)
    #     boundary = max S where (K-S)^+ >= continuation
    #   Else:
    #     V_j(S) = continuation
    
    # To interpolate V_{j+1} at S*u and S*d, we need to interpolate on the log grid
    
    # Precompute log grid
    log_S_grid = np.log(S_grid)
    
    # For each step, we need to evaluate V_{j+1} at S_grid * u and S_grid * d
    # This requires interpolation
    
    # Let's use numpy's interp for this
    
    for j in range(n_steps - 1, 0, -1):
        # Check time
        if time.time() > deadline - 5.0:
            # If we're running out of time, fill remaining boundaries with K (conservative)
            for i in range(n_dates):
                if boundaries[i] is None:
                    boundaries[i] = K
            break
        
        # Interpolate V at S_grid * u and S_grid * d
        S_u = S_grid * u
        S_d = S_grid * d
        
        # Interpolate using log-space
        log_S_u = np.log(S_u)
        log_S_d = np.log(S_d)
        
        # Clip to grid range for interpolation
        log_S_u_clipped = np.clip(log_S_u, log_S_grid[0], log_S_grid[-1])
        log_S_d_clipped = np.clip(log_S_d, log_S_grid[0], log_S_grid[-1])
        
        V_u = np.interp(log_S_u_clipped, log_S_grid, V)
        V_d = np.interp(log_S_d_clipped, log_S_grid, V)
        
        # Continuation value
        cont = np.exp(-r * dt) * (p * V_u + (1 - p) * V_d)
        
        # Check if this step is an exercise date
        if j in exercise_step_set:
            date_idx = step_to_date[j]
            immediate = np.maximum(K - S_grid, 0.0)
            V = np.maximum(immediate, cont)
            
            # Find boundary: highest S where immediate >= cont
            # i.e., where (K-S)^+ >= cont
            # For S < K: K - S >= cont => S <= K - cont
            # For S >= K: 0 >= cont (only if cont <= 0, which it isn't for puts typically)
            # So boundary is where K - S = cont, i.e., S = K - cont
            # But we need the highest S where exercise is optimal
            
            # Find the crossing point
            # exercise_mask = immediate >= cont
            # boundary = max S where exercise_mask is True
            # If no S exercises, boundary = 0
            # If all S exercise, boundary = S_max
            
            diff = immediate - cont
            # Find where diff changes from positive to negative
            # The boundary is at the crossing point
            
            # Find indices where diff >= 0
            pos_mask = diff >= 0
            if not np.any(pos_mask):
                boundaries[date_idx] = 0.0
            elif np.all(pos_mask):
                boundaries[date_idx] = S_grid[-1]
            else:
                # Find the last index where diff >= 0
                last_pos = np.where(pos_mask)[0][-1]
                # Interpolate between last_pos and last_pos+1
                if last_pos + 1 < n_grid:
                    # Linear interpolation in S-space
                    S1 = S_grid[last_pos]
                    S2 = S_grid[last_pos + 1]
                    d1 = diff[last_pos]
                    d2 = diff[last_pos + 1]
                    if d1 != d2:
                        t_frac = d1 / (d1 - d2)
                        boundaries[date_idx] = S1 + t_frac * (S2 - S1)
                    else:
                        boundaries[date_idx] = S1
                else:
                    boundaries[date_idx] = S_grid[last_pos]
        else:
            V = cont
    
    # Ensure all boundaries are set
    for i in range(n_dates):
        if boundaries[i] is None:
            boundaries[i] = K  # fallback
    
    boundaries = np.array(boundaries)
    
    # Create the policy object
    class BermudanPutPolicy:
        def __init__(self, boundaries, exercise_times, K, r, T, n_dates):
            self.boundaries = boundaries
            self.exercise_times = exercise_times
            self.K = K
            self.r = r
            self.T = T
            self.n_dates = n_dates
        
        def exercise(self, current_state, time_val):
            """
            current_state: array of stock prices
            time_val: scalar or array of current time(s)
            Returns: boolean array, True if exercise
            """
            # Convert to arrays
            current_state = np.asarray(current_state, dtype=float)
            time_val = np.asarray(time_val, dtype=float)
            
            # Handle scalar time
            if time_val.ndim == 0:
                time_val = np.full_like(current_state, time_val)
            
            # Determine which exercise date each time corresponds to
            # Exercise dates are at exercise_times[i]
            # For each time, find the closest exercise date within a small tolerance
            
            # Create output array
            result = np.zeros_like(current_state, dtype=bool)
            
            # For each unique time value, determine if it's an exercise date
            unique_times = np.unique(time_val)
            for t in unique_times:
                # Find if t is close to any exercise date
                for i in range(self.n_dates):
                    if abs(t - self.exercise_times[i]) < 1e-6:
                        # Exercise if current_state < boundary[i]
                        mask = (time_val == t) & (current_state < self.boundaries[i])
                        result |= mask
                        break
            
            return result
    
    policy = BermudanPutPolicy(boundaries, exercise_times, K, r, T, n_dates)
    return policy
# EVOLVE-BLOCK-END
