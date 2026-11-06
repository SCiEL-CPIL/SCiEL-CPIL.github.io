"""Bermudan exercise-policy research with explicit candidate API."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit the exercise policy using Longstaff-Schwartz Monte Carlo regression.
    
    Parameters:
    -----------
    training_paths : np.ndarray
        Array of shape (n_paths, n_timesteps + 1).
        n_timesteps is 12 (monthly).
    contract : dict
        Contains keys: 'S0', 'K', 'r', 'sigma', 'T', 'n_paths', 'n_eval_paths'.
        Note: S0 might vary, but the policy is usually state-dependent, so it shouldn't depend on S0 explicitly in the fit, 
        but the regression is on S_t.
        
    Returns:
    --------
    object with exercise(current_state, time) method.
    """
    
    # Extract parameters
    K = contract['K']
    r = contract['r']
    sigma = contract['sigma']
    T = contract['T']
    
    # Determine number of steps
    # Contract says 12 monthly exercise dates.
    # training_paths shape is likely (N, 13) -> t0 to t12.
    # Exercise dates are t1...t12.
    n_paths, n_steps_plus_1 = training_paths.shape
    n_steps = n_steps_plus_1 - 1
    
    # Time step
    dt = T / n_steps
    
    # Discount factor per step
    disc = np.exp(-r * dt)
    
    # Initialize value array for the last step
    # V[i, k] is the value of the option at step k for path i
    V = np.zeros((n_paths, n_steps + 1))
    
    # Last step: t12
    # Payoff is intrinsic value
    S_last = training_paths[:, n_steps]
    V[:, n_steps] = np.maximum(K - S_last, 0)
    
    # Store regression models for each step to use in the exercise function
    # We will store the coefficients and the normalization params
    models = []
    
    # Backward induction
    # We need to fit from k = n_steps - 1 down to 1
    # Exercise dates are 1 to n_steps.
    # At step k, we decide whether to exercise.
    # Continuation value is E[disc * V_{k+1} | S_k]
    
    for k in range(n_steps - 1, 0, -1):
        S_k = training_paths[:, k]
        V_next = V[:, k + 1]
        
        # Target for regression: Discounted future value
        Y = disc * V_next
        
        # Normalize S_k for better polynomial conditioning
        # Map S_k to [-1, 1] or [0, 1]
        # Let's use [0, 1] scaling based on observed min/max or fixed range
        # Fixed range might be better for generalization if eval paths go out of range,
        # but observed min/max is standard for LS on training data.
        # Let's use a robust range: [0.5*K, 1.5*K] or similar? 
        # Actually, just scaling to [0,1] based on training min/max is standard.
        
        S_min = np.min(S_k)
        S_max = np.max(S_k)
        if S_max == S_min:
            S_norm = np.zeros_like(S_k)
            scale = 1.0
            offset = 0.0
        else:
            scale = S_max - S_min
            offset = S_min
            S_norm = (S_k - offset) / scale
        
        # Build design matrix
        # Degree 5 or 6 polynomial
        degree = 6
        # Use np.vander
        # np.vander(S_norm, N=degree+1, increasing=True)
        X = np.vander(S_norm, N=degree + 1, increasing=True)
        
        # Ridge Regression
        # Minimize ||X w - Y||^2 + alpha * ||w||^2
        alpha = 1.0
        try:
            # Solve (X^T X + alpha I) w = X^T Y
            XtX = X.T @ X
            XtY = X.T @ Y
            # Add alpha to diagonal
            XtX += alpha * np.eye(degree + 1)
            w = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            # Fallback to linear if fails
            degree = 1
            X = np.vander(S_norm, N=2, increasing=True)
            XtX = X.T @ X
            XtY = X.T @ Y
            XtX += alpha * np.eye(2)
            w = np.linalg.solve(XtX, XtY)
            
        # Store model parameters
        models.append({
            'w': w,
            'offset': offset,
            'scale': scale,
            'degree': degree
        })
        
        # Calculate continuation value for this step
        C_k = X @ w
        
        # Update V for this step
        # Exercise if Intrinsic > Continuation
        Intrinsic = np.maximum(K - S_k, 0)
        V[:, k] = np.maximum(Intrinsic, C_k)
        
    # Note: V[:, 0] is not used for exercise decision as there is no exercise at t0.
    # But we might need it if the interface asks for value? No, only exercise().
    
    # Create the policy object
    class BermudanPolicy:
        def __init__(self, models, K, r, T, n_steps):
            self.models = models
            self.K = K
            self.r = r
            self.T = T
            self.n_steps = n_steps
            
        def exercise(self, current_state, time_index):
            """
            current_state: np.ndarray of shape (n_eval_paths,)
            time_index: int, 1-based index of exercise date (1 to 12)
            Returns: boolean vector of shape (n_eval_paths,)
            """
            if time_index < 1 or time_index > self.n_steps:
                # No exercise at t0 or beyond
                return np.zeros(len(current_state), dtype=bool)
            
            # Get the model for this time index
            # models list is indexed by k-1? 
            # In the loop, k went from n_steps-1 down to 1.
            # models[0] corresponds to k = n_steps - 1
            # models[i] corresponds to k = n_steps - 1 - i
            # So for time_index k, index is n_steps - 1 - k
            idx = self.n_steps - 1 - time_index
            if idx < 0 or idx >= len(self.models):
                return np.zeros(len(current_state), dtype=bool)
            
            model = self.models[idx]
            w = model['w']
            offset = model['offset']
            scale = model['scale']
            degree = model['degree']
            
            # Normalize current state
            if scale == 0:
                S_norm = np.zeros_like(current_state)
            else:
                S_norm = (current_state - offset) / scale
            
            # Build design matrix
            X = np.vander(S_norm, N=degree + 1, increasing=True)
            
            # Calculate continuation value
            C_k = X @ w
            
            # Intrinsic value
            Intrinsic = np.maximum(self.K - current_state, 0)
            
            # Exercise if Intrinsic > Continuation
            return Intrinsic > C_k
            
    return BermudanPolicy(models, K, r, T, n_steps)

# EVOLVE-BLOCK-END
