import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score using Expected Improvement with 
    a logarithmic cost penalty (equivalent to dividing by sqrt(cost)).
    
    Args:
        mu: Array of GP means (shape 2000,)
        sigma: Array of GP standard deviations (shape 2000,)
        f_best: Current best observed value (scalar)
        cost: Array of evaluation costs (shape 2000,)
        
    Returns:
        Array of acquisition scores (shape 2000,), higher is better.
    """
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s limit

    # Convert to numpy arrays and ensure float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Sanitize inputs
    # If mu is not finite, replace with f_best (neutral)
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    # If sigma is not finite or too small, replace with a small positive value
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-6)
    # If cost is not finite or too small, replace with 1.0 (minimum cost)
    cost_safe = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    
    # Calculate Expected Improvement (EI)
    # For minimization:
    # improvement = f_best - mu
    # Z = improvement / sigma
    # EI = improvement * Phi(-Z) + sigma * phi(Z)
    
    improvement = f_best - mu_safe
    # Avoid division by zero
    Z = improvement / sigma_safe
    
    # Calculate CDF and PDF of standard normal distribution
    # Phi(-Z) = 1 - Phi(Z)
    # We use scipy.stats.norm for accuracy
    phi_Z = norm.pdf(Z)
    Phi_neg_Z = norm.cdf(-Z)
    
    EI = improvement * Phi_neg_Z + sigma_safe * phi_Z
    
    # Ensure EI is non-negative (numerical stability)
    EI = np.maximum(EI, 0.0)
    
    # Apply cost penalty
    # The directive suggests a log-cost penalty: log(S) = log(EI) - 0.5 * log(cost)
    # This is equivalent to S = EI / sqrt(cost)
    # This penalizes high-cost points less aggressively than linear cost division,
    # preserving exploration in high-cost regions if the EI is sufficiently high.
    
    cost_penalty = np.sqrt(cost_safe)
    
    # Final score
    score = EI / cost_penalty
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() > deadline:
        return score

    return score
