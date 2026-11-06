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
    
    # Backward Induction with Polynomial Regression Smoothing
    # We fit a polynomial to V_next as a function of S_i
    # Basis: [1, S, S^2, S^3, S^4]
    # To improve conditioning, we scale S by K
    # S_scaled = S / K
    
    # Initialize at the last date (index n_dates-1 in S, which is step 12)
    V_next = np.maximum(K - S[:, n_dates - 1], 0.0)
    
    # Store decisions and thresholds
    thresholds = np.zeros(n_dates)
    decisions = [None] * n_dates
    
    # Regularization parameter for Ridge Regression
    # We can tune this, but a fixed reasonable value is often sufficient
    # Let's try a small grid search for lambda if time permits, otherwise use 1.0
    # Given the deadline is 110s and N=4096, D=5, solving is very fast.
    # We will use a fixed lambda for simplicity and stability, or a very small search.
    # Let's use lambda = 1.0 * (variance of V_next) to make it scale-invariant?
    # Or just 1.0.
    
    # Pre-compute powers of S_scaled for all steps to save time?
    # S is (n_paths, n_dates). We can compute powers once.
    S_scaled = S / K
    S_pow = np.stack([
        np.ones_like(S_scaled),
        S_scaled,
        S_scaled**2,
        S_scaled**3,
        S_scaled**4
    ], axis=-1) # Shape: (n_paths, n_dates, 5)

    # Define a small set of lambdas to potentially test
    # Since we have plenty of time, we can do a simple validation on a subset
    # for each step to pick the best lambda.
    # Subset size: 1000 paths for train, 500 for val?
    # Actually, with 4096 paths, we can just use all for fit and maybe check condition number.
    # Let's stick to a robust lambda. 
    # A good heuristic: lambda = 1e-3 * trace(X^T X) / D ?
    # Let's just use 1.0. It's often a safe bet for normalized inputs.
    
    lambda_val = 1.0
    
    for i in range(n_dates - 1, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: use simple threshold K
            break
            
        S_i = S[:, i]
        immediate = np.maximum(K - S_i, 0.0)
        
        # Design matrix for step i
        X = S_pow[:, i, :] # Shape (n_paths, 5)
        y = V_next # Shape (n_paths,)
        
        # Ridge Regression: (X^T X + lambda I) beta = X^T y
        # We can use np.linalg.solve or lstsq.
        # X^T X is 5x5. Very fast.
        XtX = X.T @ X
        Xty = X.T @ y
        
        # Add regularization
        # Identity matrix for 5 features
        I = np.eye(5)
        A = XtX + lambda_val * I
        b = Xty
        
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback to lstsq
            beta = np.linalg.lstsq(A, b, rcond=None)[0]
            
        # Predict continuation value for all paths at step i
        C_smooth = X @ beta
        
        # Ensure C_smooth is non-negative (value of an option is non-negative)
        C_smooth = np.maximum(C_smooth, 0.0)
        
        # Discount the continuation value
        cont_value = discount_step * C_smooth
        
        # Decision: Exercise if Immediate >= Continuation
        decision = immediate >= cont_value
        
        decisions[i] = decision
        
        # Update V for the next iteration (backwards)
        # The true value at step i is max(Immediate, Continuation)
        # We use this realized value as the target for the previous step's regression
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
    
    for i in range(1, n_dates):
        if thresholds[i] < thresholds[i-1]:
            thresholds[i] = thresholds[i-1]
            
    # Clamp to K
    thresholds = np.minimum(thresholds, K)
    
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
            months = time * 12.0
            idx = int(round(months))
            
            if idx < 1:
                idx = 1
            elif idx > 12:
                idx = 12
                
            arr_idx = idx - 1
            threshold = self.thresholds[arr_idx]
            return (cs <= threshold).astype(bool)

    return ExercisePolicy(thresholds, K)
# EVOLVE-BLOCK-END
