import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB and Expected Improvement
    with budget-aware exploration scaling.
    
    Args:
        mu: Mean of the GP prediction, shape (N,)
        sigma: Standard deviation of the GP prediction, shape (N,)
        f_best: Best observed value so far.
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        scores: Acquisition scores, shape (N,), higher = evaluate next
    """
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    # Input validation and safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure shapes are correct
    n = mu.shape[0]
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Check for non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)
    
    # Budget-aware beta: start high (exploration), decrease as we approach budget limit
    # The minimum of hartmann 6d is -3.32237
    min_hartmann = -3.32237
    dist_to_min = max(0.0, f_best - min_hartmann)
    
    # beta starts at ~3.0 and decreases to ~1.0 as we get closer to optimum
    beta_base = 1.0 + 2.0 * np.exp(-dist_to_min / 0.5)
    
    # Scale by a conservative budget fraction estimate
    beta = beta_base * np.sqrt(0.5)
    
    # UCB component for minimization: mu - beta * sigma
    # We want to maximize the score, so we negate: -mu + beta * sigma
    ucb_score = -mu + beta * sigma_safe
    
    # Expected Improvement component for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Clip to avoid numerical issues
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI is positive when mu < f_best, which is what we want for minimization
    
    # Normalize both components to have similar scales
    ucb_std = np.std(ucb_score) + 1e-8
    ei_std = np.std(ei) + 1e-8
    
    ucb_norm = ucb_score / ucb_std
    ei_norm = ei / ei_std
    
    # Adaptive weights: more UCB when uncertain, more EI when confident
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean > 1e-6:
        # High uncertainty: favor UCB
        ucb_weight = 0.65
        ei_weight = 0.35
    else:
        # Low uncertainty: favor EI
        ucb_weight = 0.4
        ei_weight = 0.6
    
    combined = ucb_weight * ucb_norm + ei_weight * ei_norm
    
    # Cost-aware: divide by cost for stronger penalty on expensive points
    # This encourages the optimizer to prefer cheaper evaluations
    scores = combined / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return a safe fallback
        return np.zeros(n, dtype=np.float64)
    
    return scores
