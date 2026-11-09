"""Bermudan put exercise policy via Longstaff-Schwartz Monte Carlo with polynomial regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    # Extract parameters from contract
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Exercise dates: 12 monthly, no time-0
    dt = T / n_exercise
    exercise_times = np.arange(1, n_exercise + 1) * dt
    
    # Get training paths - assume shape (n_paths, n_exercise + 1) or (n_paths, n_exercise)
    # training_paths[i, j] is the state at exercise date j (0-indexed for exercise dates 1..n)
    # Or it could be the full path including time 0
    n_paths = training_paths.shape[0] if hasattr(training_paths, 'shape') else len(training_paths)
    
    # Ensure we have the right number of exercise dates
    if hasattr(training_paths, 'shape') and training_paths.ndim == 2:
        if training_paths.shape[1] == n_exercise + 1:
            # Includes time 0, so exercise states are columns 1..n
            exercise_states = training_paths[:, 1:]
        elif training_paths.shape[1] == n_exercise:
            # Only exercise states
            exercise_states = training_paths
        else:
            # Try to figure out
            exercise_states = training_paths[:, -n_exercise:]
    else:
        # Fallback
        exercise_states = np.array(training_paths)
        if exercise_states.ndim == 1:
            exercise_states = exercise_states.reshape(-1, n_exercise)
        elif exercise_states.shape[1] > n_exercise:
            exercise_states = exercise_states[:, -n_exercise:]
    
    # Put payoff: max(K - S, 0)
    def put_payoff(S):
        return np.maximum(K - S, 0.0)
    
    # Discount factor per step
    disc = np.exp(-r * dt)
    
    # Longstaff-Schwartz algorithm
    # V[j] = estimated continuation value at exercise date j (0-indexed)
    # We work backwards from the last exercise date
    
    # For the last exercise date, continuation value = 0 (must exercise or expire)
    # Actually, for Bermudan, at the last date you either exercise or the option expires
    # So continuation value at last date = 0, and value = payoff
    
    n = n_exercise  # number of exercise dates
    
    # Store continuation values for each exercise date
    # cont_val[j] is the vector of continuation values at exercise date j
    # We'll store these to compute the policy
    
    # Initialize: at the last exercise date, continuation = 0
    # Value at last date = payoff
    
    # We need to estimate continuation values at each date
    # cont[j] = E[disc * V[j+1] | S_j]
    # where V[j+1] = max(payoff_{j+1}, cont[j+1])
    
    # Start from the end
    # V[n-1] = payoff[n-1] (last exercise date, no continuation)
    # cont[n-2] = E[disc * V[n-1] | S_{n-2}]
    # V[n-2] = max(payoff[n-2], cont[n-2])
    # ...
    
    # We'll store the continuation value estimates as functions of S
    # For each date j (0 to n-2), we fit a regression of disc * V[j+1] on S_j
    
    # First, compute V from the end
    # V[n-1] = payoff at last date
    V = np.zeros((n_paths, n))
    V[:, n-1] = put_payoff(exercise_states[:, n-1])
    
    # For j = n-2 down to 0:
    # Fit regression: disc * V[j+1] ~ f(S_j)
    # cont[j] = f(S_j)
    # V[j] = max(payoff[j], cont[j])
    
    # Store the fitted models for policy evaluation
    # We'll store the coefficients of the polynomial regression for each date
    
    # Use polynomial features of degree 3 for S
    def make_features(S):
        """Create polynomial features of degree 3."""
        S = np.asarray(S, dtype=np.float64)
        return np.column_stack([
            np.ones_like(S),
            S,
            S**2,
            S**3
        ])
    
    # Store models: list of (coef, intercept) or just coef vectors for each date
    # We'll use simple linear regression with polynomial features
    
    # For numerical stability, we'll standardize S or just use raw features
    # Let's use raw features but be careful with scaling
    
    # Actually, let's use a simpler approach: fit using np.linalg.lstsq
    
    # Store the continuation value functions
    # cont_funcs[j] will be a function that takes S and returns continuation value
    # We'll store the regression coefficients
    
    reg_coefs = [None] * n  # reg_coefs[j] is the coefficient vector for date j
    reg_intercepts = [None] * n  # not needed if we include ones in features
    
    # Work backwards
    for j in range(n-2, -1, -1):
        # Target: disc * V[j+1]
        target = disc * V[:, j+1]
        
        # Features: polynomial in S_j
        S_j = exercise_states[:, j]
        X = make_features(S_j)
        
        # Fit regression: target = X @ coef
        # Use lstsq for least squares
        coef, residuals, rank, sv = np.linalg.lstsq(X, target, rcond=None)
        
        reg_coefs[j] = coef
        
        # Continuation value at date j
        cont_j = X @ coef
        V[:, j] = np.maximum(put_payoff(S_j), cont_j)
        
        # Check time
        if time.time() > deadline - 5.0:
            # Not enough time, break and use what we have
            break
    
    # Now we have the policy:
    # At exercise date j, exercise if payoff[j] >= cont[j]
    # i.e., payoff[j] >= X_j @ reg_coefs[j]
    
    # For the last date (n-1), always exercise (or rather, the value is the payoff)
    # Actually, at the last date, there's no continuation, so you exercise if payoff > 0
    
    # Create the exercise policy object
    class ExercisePolicy:
        def __init__(self, reg_coefs, K, n_exercise, S0):
            self.reg_coefs = reg_coefs
            self.K = K
            self.n_exercise = n_exercise
            self.S0 = S0
            
        def exercise(self, current_state, time_idx):
            """
            Return a boolean vector indicating whether to exercise at each state.
            current_state: array of current spot prices
            time_idx: the exercise date index (0-based, 0 to n_exercise-1)
            """
            S = np.asarray(current_state, dtype=np.float64)
            if S.ndim == 0:
                S = S.reshape(1)
            
            if time_idx is None or time_idx < 0:
                # Before first exercise date, can't exercise
                return np.zeros(len(S), dtype=bool)
            
            if time_idx >= self.n_exercise:
                # After last exercise date, can't exercise
                return np.zeros(len(S), dtype=bool)
            
            # At the last exercise date, exercise if payoff > 0
            if time_idx == self.n_exercise - 1:
                payoff = np.maximum(self.K - S, 0.0)
                return payoff > 0
            
            # Otherwise, compare payoff to continuation value
            if self.reg_coefs[time_idx] is None:
                # Fallback: exercise if in the money
                payoff = np.maximum(self.K - S, 0.0)
                return payoff > 0
            
            # Compute continuation value
            X = make_features(S)
            cont = X @ self.reg_coefs[time_idx]
            
            # Exercise if payoff >= continuation
            payoff = np.maximum(self.K - S, 0.0)
            return payoff >= cont
    
    return ExercisePolicy(reg_coefs, K, n_exercise, S0)
# EVOLVE-BLOCK-END
