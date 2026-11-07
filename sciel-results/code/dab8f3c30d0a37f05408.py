import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization.
    Uses Expected Improvement per unit cost (EI/cost) combined with
    a small UCB component, all normalized by cost to balance
    exploration, exploitation, and budget efficiency.
    """
    t0 = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB for minimization
    # Use a moderate exploration parameter
    xi = 0.1
    ucb = -mu + xi * sigma_safe
    
    # Cost-aware combination:
    # Primary: EI / cost (expected improvement per unit cost)
    # Secondary: small UCB / cost component for exploration
    # This directly optimizes marginal return on budget
    
    # Normalize EI to be in a reasonable range relative to f_best
    # Scale EI by 1/(f_best + epsilon) to make it dimensionless-ish
    scale = 1.0 / (abs(f_best) + 1e-6)
    
    ei_normalized = ei * scale
    ucb_normalized = ucb * scale
    
    # Combine: weight EI more heavily, add small UCB for exploration
    w_ei = 1.0
    w_ucb = 0.15
    
    combined = w_ei * ei_normalized + w_ucb * ucb_normalized
    
    # Apply cost penalty: divide by cost to get "improvement per unit cost"
    # Use a soft penalty: cost^1.0 (linear) rather than cost^1.5
    # This is less aggressive but more principled for BOB
    score = combined / cost_safe
    
    # Add a tiny regularization to avoid extreme values
    score = score + 1e-10
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety: if computation took too long, we could simplify, but 
    # for 2000 points this should be fast
    elapsed = time.time() - t0
    if elapsed > 280:
        # Return a simpler fallback if we're running out of time
        return np.maximum(ei / cost_safe, 0.0)
    
    return score
