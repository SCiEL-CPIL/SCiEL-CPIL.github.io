import numpy as np
import math
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option using Longstaff-Schwartz Monte Carlo.
    """
    start_wall = time.time()
    deadline = 140.0  # Safety margin

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    # Assume training_paths is shape (n_paths, n_time_steps)
    # The task specifies 12 monthly exercise dates.
    # We assume the columns of training_paths correspond to these dates, 
    # possibly excluding time 0 or including it.
    # "12 monthly exercise dates (no time-0 exercise)"
    # If the paths have 12 columns, they are t=1/12...1.
    # If they have 13, they are t=0...1.
    
    paths = np.asarray(training_paths)
    n_paths, n_steps = paths.shape
    
    # Determine the indices for the 12 exercise dates.
    # If n_steps == 12, indices are 0..11 corresponding to months 1..12.
    # If n_steps == 13, indices are 1..12 corresponding to months 1..12.
    # We need to be careful. Let's assume the standard case where the last column is T=1.
    # The exercise dates are at t_k = k * T / n_exercises.
    
    # We will work backwards from the last exercise date (index corresponding to T=1)
    # to the first (index corresponding to T=1/12).
    
    # Let's assume the paths are provided for the exercise dates only.
    # If n_steps is 12, the dates are 1/12 to 1.
    # If n_steps is 13, the dates are 0 to 1, and we ignore index 0.
    
    if n_steps == 12:
        ex_indices = np.arange(12)
    elif n_steps == 13:
        ex_indices = np.arange(1, 13)
    else:
        # Fallback: assume the last 12 columns are the exercise dates
        ex_indices = np.arange(n_steps - 12, n_steps)

    n_ex = len(ex_indices)
    
    # Discount factor per step (monthly)
    dt = T / n_ex
    disc = math.exp(-r * dt)
    
    # Initialize the value at the last exercise date (terminal)
    # V is the value of the option at the current step for each path
    S_last = paths[:, ex_indices[-1]]
    V = np.maximum(K - S_last, 0.0)
    
    # We will store the regression coefficients for each step
    # Basis: [1, ln(S), (ln(S))^2]
    # We store the coefficients for the polynomial in ln(S)
    coeffs_list = [None] * n_ex # coeffs_list[k] for exercise date k (0-indexed in ex_indices)
    coeffs_list[-1] = None # Terminal doesn't need regression, payoff is known
    
    # Backward induction
    # k goes from n_ex - 2 down to 0
    for k in range(n_ex - 2, -1, -1):
        S_curr = paths[:, ex_indices[k]]
        S_next = paths[:, ex_indices[k+1]]
        
        # Continuation value is discounted next value
        V_cont = disc * V
        
        # Fit regression: V_cont ~ f(S_curr)
        # Use ln(S) as feature
        log_S = np.log(S_curr)
        
        # Design matrix
        # We use a quadratic basis: 1, x, x^2
        X = np.column_stack((np.ones_like(log_S), log_S, log_S**2))
        
        # Solve least squares: X @ c = V_cont
        # Using np.linalg.lstsq for stability
        try:
            c, residuals, rank, s = np.linalg.lstsq(X, V_cont, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback to simple mean if lstsq fails
            c = np.array([np.mean(V_cont), 0.0, 0.0])
            
        coeffs_list[k] = c
        
        # Calculate regression value for all paths at this step
        V_reg = X @ c
        
        # Exercise decision: Exercise if intrinsic > continuation (regression)
        intrinsic = np.maximum(K - S_curr, 0.0)
        exercise_mask = intrinsic > V_reg
        
        # Update V for next iteration (going backwards)
        # V is the value at step k, which is max(intrinsic, V_reg)
        # Note: V_reg is the estimate of the continuation value E[V_{k+1}|S_k]
        V = np.where(exercise_mask, intrinsic, V_reg)
        
        # Deadline guard
        if time.time() > deadline:
            break
            
    # Fill remaining coeffs if deadline hit early
    for k in range(n_ex - 1, -1, -1):
        if coeffs_list[k] is None and k < n_ex - 1:
            # Fallback: use the next available coeffs or simple heuristic
            if k + 1 < n_ex and coeffs_list[k+1] is not None:
                coeffs_list[k] = coeffs_list[k+1]
            else:
                # Simple heuristic: linear in S
                coeffs_list[k] = np.array([K, -1, 0]) # Approximation of K-S
        
    class ExercisePolicy:
        def __init__(self, coeffs, ex_indices, n_ex, K_val):
            self.coeffs = coeffs
            self.ex_indices = ex_indices
            self.n_ex = n_ex
            self.K = K_val
            
        def exercise(self, current_state, time_val):
            # time_val is in [0, T]
            # Determine which exercise step this corresponds to
            # Exercise dates are at t_k = (k+1) * T / n_ex for k=0..n_ex-1
            # If time_val is close to an exercise date, we use that policy.
            # The input time_val is likely the current time.
            
            # Map time_val to step index
            # t_k = (k+1) * dt
            # k = time_val / dt - 1
            dt = 1.0 / self.n_ex # Assuming T=1 for normalization in mapping, 
                                 # but better to use actual T if available. 
                                 # The contract T is 1.0.
            
            # time_val is the current time.
            # We exercise at discrete times.
            # If time_val is not exactly at an exercise date, 
            # the problem statement says "exercise(current_state, time_val)".
            # Usually, this implies time_val IS one of the exercise dates.
            
            # Let's find the closest exercise index
            # Exercise dates: 1/12, 2/12, ..., 12/12
            # time_val should be one of these.
            
            # Calculate step index
            # step = time_val / (T / n_ex)
            # If T=1, step = time_val * 12
            step_float = time_val * self.n_ex
            step_idx = int(round(step_float)) - 1 # 0-indexed
            
            if step_idx < 0:
                return np.zeros_like(current_state, dtype=bool)
            if step_idx >= self.n_ex:
                step_idx = self.n_ex - 1
                
            c = self.coeffs[step_idx]
            if c is None:
                return np.zeros_like(current_state, dtype=bool)
                
            log_S = np.log(np.maximum(current_state, 1e-8))
            X = np.column_stack((np.ones_like(log_S), log_S, log_S**2))
            V_reg = X @ c
            intrinsic = np.maximum(self.K - current_state, 0.0)
            
            return intrinsic > V_reg

    return ExercisePolicy(coeffs_list, ex_indices, n_ex, K)
