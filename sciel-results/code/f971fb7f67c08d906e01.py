import numpy as np
from scipy.stats import norm
import time

def _ei_score(mu_val, sigma_val, f_best):
    """Compute EI for a single point."""
    if sigma_val < 1e-8:
        sigma_val = 1e-8
    z = (f_best - mu_val) / sigma_val
    z = np.clip(z, -10, 10)
    return (f_best - mu_val) * norm.cdf(z) + sigma_val * norm.pdf(z)

def _local_optimize_ei(mu, sigma, f_best, start_idx, max_iter=50):
    """
    Local optimization of EI around the discrete candidate at start_idx.
    Uses coordinate descent on the 2D candidate space.
    Returns the best EI value found in the neighborhood.
    """
    # Get the starting point
    x = np.array([mu[start_idx], sigma[start_idx]], dtype=float)
    best_ei = _ei_score(x[0], x[1], f_best)
    
    # Coordinate descent with adaptive step size
    step_size = 0.05
    for _ in range(max_iter):
        improved = False
        for dim in range(2):
            for direction in [-1, 1]:
                trial = x.copy()
                trial[dim] += direction * step_size
                # Clip to valid range
                trial[0] = np.clip(trial[0], -10, 10)
                trial[1] = np.clip(trial[1], 1e-8, 10)
                
                trial_ei = _ei_score(trial[0], trial[1], f_best)
                if trial_ei > best_ei:
                    best_ei = trial_ei
                    x = trial
                    improved = True
        
        if not improved:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-4:
                break
    
    return best_ei

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with local optimization polish.
    
    For each of the 2000 discrete candidates, we compute the EI score.
    Then, for the top-K candidates, we perform a local optimization in the
    (mu, sigma) space to find a better EI value in the neighborhood.
    This allows us to find sub-grid optima that the discrete grid might miss.
    """
    # Track time for deadline guard
    t_start = time.time()
    deadline = t_start + 290.0  # 300s limit, 10s safety margin
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute Z-score for Expected Improvement
    Z = (f_best - mu) / sigma
    
    # Clip Z to avoid numerical issues in CDF/PDF
    Z = np.clip(Z, -10, 10)
    
    # Phi(Z) = standard normal CDF, phi(Z) = standard normal PDF
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    EI = (f_best - mu) * Phi + sigma * phi
    
    # Ensure finite scores
    EI = np.where(np.isfinite(EI), EI, 0.0)
    
    # Check if we have time for local optimization polish
    elapsed = time.time() - t_start
    if elapsed < deadline - 5.0:
        # Find top-K candidates for local optimization
        top_k = min(20, len(mu))
        top_indices = np.argsort(EI)[-top_k:]
        
        # For each top candidate, try local optimization
        for idx in top_indices:
            if time.time() > deadline:
                break
            
            # Local optimization around this candidate
            polished_ei = _local_optimize_ei(mu, sigma, f_best, idx, max_iter=30)
            
            # Update the EI score for this candidate with the polished value
            if polished_ei > EI[idx]:
                EI[idx] = polished_ei
    
    # Final check for finite scores
    EI = np.where(np.isfinite(EI), EI, 0.0)
    
    return EI
