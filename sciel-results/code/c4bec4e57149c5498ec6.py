"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Hartmann 3D.
    
    Structural Change: Uses a Scale-Invariant Additive Cost Penalty on Expected Improvement.
    
    Instead of dividing EI by cost (Cost-Weighted EI), which can overly suppress 
    expensive regions even if they hold the global minimum, this method subtracts 
    a normalized cost penalty from the normalized EI.
    
    Score = EI_norm - alpha * Cost_norm
    
    This allows the algorithm to tolerate higher costs if the expected improvement 
    is sufficiently large, while still preferring cheaper evaluations when gains 
    are comparable. The normalization ensures the penalty is relative to the 
    current candidate distribution, making the hyperparameter alpha scale-invariant.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill
    
    # Ensure inputs are valid numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    n = len(mu)
    
    # Basic safety: if anything is wrong, return a simple fallback
    if n == 0 or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        return np.zeros(n, dtype=np.float64)
    
    # Clamp sigma to positive values for numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in cdf/pdf for extreme values
    z = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI should be non-negative. Clip small negative values due to floating point errors.
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Check time
    if time.time() > deadline - 5.0:
        # Fallback to simple EI if running out of time
        return ei
    
    # Normalize EI to [0, 1] range
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros(n, dtype=np.float64)
    
    # Normalize Cost to [0, 1] range based on current candidate set
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    
    if cost_range > 1e-10:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros(n, dtype=np.float64)
    
    # Define the penalty weight alpha
    # A smaller alpha means we are less sensitive to cost.
    # A larger alpha means we strongly prefer cheap points.
    # 0.15 suggests we are willing to pay up to 15% of the max possible gain in extra cost.
    alpha = 0.15
    
    # Compute the final score
    score = ei_norm - alpha * cost_norm
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
