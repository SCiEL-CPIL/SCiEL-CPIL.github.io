import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB and Expected Improvement.
    
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
    
    # UCB component for minimization: mu - beta * sigma
    # We want to maximize the score, so we negate: -mu + beta * sigma
    beta = 2.0
    ucb_score = -mu + beta * sigma_safe
    
    # Expected Improvement component for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF of standard normal, phi is PDF
    from scipy.stats import norm
    
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Clip to avoid numerical issues
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI is positive when mu < f_best, which is what we want for minimization
    
    # Combine UCB and EI with adaptive weighting
    # Weight based on uncertainty: when sigma is high, rely more on UCB
    # When sigma is low, rely more on EI
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean > 1e-6:
        # Normalize sigma to get relative uncertainty
        rel_sigma = sigma_safe / sigma_mean
        # Weight: 0.5 for UCB, 0.5 for EI, but adjust based on uncertainty
        # When uncertainty is high, UCB is more important
        ucb_weight = 0.6
        ei_weight = 0.4
    else:
        ucb_weight = 0.3
        ei_weight = 0.7
    
    # Normalize both components to have similar scales
    ucb_norm = ucb_score / (np.std(ucb_score) + 1e-8)
    ei_norm = ei / (np.std(ei) + 1e-8)
    
    # Combine
    combined = ucb_weight * ucb_norm + ei_weight * ei_norm
    
    # Cost-aware: divide by cost to penalize expensive points
    # But we want to maximize, so we divide by cost
    scores = combined / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return a safe fallback
        return np.zeros(n, dtype=np.float64)
    
    return scores
