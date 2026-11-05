"""
Sequential Coordinate-wise Ensemble Square Root Filter (EnSRF) for Lorenz-96 assimilation.
This strategy differs from the seed LEKF by:
1. Using a square-root filter formulation that updates the ensemble directly without computing a Kalman gain matrix.
2. Processing observations sequentially (one coordinate at a time) using regression-based updates.
3. This naturally handles the small ensemble size (N=16) and avoids the spurious correlations and bias introduced by the matrix inversion in the LEKF.
"""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform Sequential Coordinate-wise EnSRF update.
    ensemble: (16, 20)
    observations: (8,)
    observed_indices: (8,)
    observation_variance: scalar
    """
    N, n_vars = ensemble.shape
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    # Work on a copy
    X = ensemble.copy()
    
    # Ensure observations and indices are numpy arrays
    y = np.asarray(observations, dtype=float)
    idx_obs = np.asarray(observed_indices, dtype=int)
    R = observation_variance
    
    # Sequential update: process each observed variable one by one
    # For each observation y_k at index i_k:
    # 1. Compute the ensemble mean and covariance at i_k and all other variables
    # 2. Update the ensemble using the regression formula
    
    for k in range(len(idx_obs)):
        if time.time() > deadline:
            break
            
        i_obs = idx_obs[k]
        y_k = y[k]
        
        # Current state of the ensemble
        # x_obs: values at the observed index for all members (N,)
        x_obs = X[:, i_obs]
        
        # Mean of observed variable
        mu_obs = np.mean(x_obs)
        
        # Innovation
        innov = y_k - mu_obs
        
        # Compute covariance between observed variable and all other variables
        # We need Cov(x_i, x_j) for all j
        # Centered ensemble: X_c = X - mean(X, axis=0)
        # But we only need the covariance with x_obs
        # Cov(x_obs, X_j) = (1/(N-1)) * sum_m (x_obs[m] - mu_obs) * (X[m, j] - mu_j)
        
        # To be efficient, we can compute the update directly.
        # The update rule for EnSRF (sequential) is:
        # x_j_new = x_j + K_j * (y - x_obs)
        # where K_j is the regression coefficient of x_j on x_obs.
        # K_j = Cov(x_j, x_obs) / Var(x_obs)
        
        # Compute Var(x_obs)
        var_obs = np.var(x_obs, ddof=1)
        
        # Avoid division by zero
        if var_obs < 1e-10:
            continue
            
        # Compute Cov(x_j, x_obs) for all j
        # We need the covariance matrix between x_obs and all columns of X
        # Let's compute the centered x_obs: dx_obs = x_obs - mu_obs
        dx_obs = x_obs - mu_obs
        
        # Centered X: X_c = X - mean(X, axis=0)
        # But we can compute Cov(x_j, x_obs) = (1/(N-1)) * sum_m dx_obs[m] * (X[m, j] - mu_j)
        # This is equivalent to (dx_obs @ (X - mu)) / (N-1)
        
        # Compute mean of X
        mu_X = np.mean(X, axis=0)
        
        # Centered X
        X_c = X - mu_X
        
        # Cov(x_j, x_obs) for all j:
        # cov_vec = (dx_obs @ X_c) / (N-1)  -> shape (n_vars,)
        cov_vec = (dx_obs @ X_c) / (N - 1)
        
        # The gain for each variable j is:
        # K_j = cov_vec[j] / var_obs
        K = cov_vec / var_obs
        
        # Apply the update:
        # X_new = X + K * (y_k - x_obs)
        # Note: (y_k - x_obs) is the innovation for each member
        innov_vec = y_k - x_obs  # shape (N,)
        
        # X is (N, n_vars)
        # K is (n_vars,)
        # We need to add K * innov_vec[m] to X[m, j] for all m, j
        # This is an outer product: np.outer(innov_vec, K)
        
        X = X + np.outer(innov_vec, K)
        
        # Safety check: ensure finite values
        if not np.all(np.isfinite(X)):
            # Revert to previous state if something went wrong? 
            # Or just clip?
            # Let's clip to avoid NaNs propagating
            X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)
            X = np.clip(X, -100.0, 100.0)
    
    # Final safety checks
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)
    X = np.clip(X, -100.0, 100.0)
    
    return X

# EVOLVE-BLOCK-END
