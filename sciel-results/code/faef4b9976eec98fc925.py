import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement.
    
    Args:
        mu: Mean of the GP prediction, shape (N,)
        sigma: Standard deviation of the GP prediction, shape (N,)
        f_best: Best observed value so far.
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        scores: Acquisition scores, shape (N,), higher = evaluate next
    """
    start_time = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill
    
    # Input validation and safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure shapes are correct
    n = mu.shape[0]
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Check for non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-10)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)
    
    # Compute Probability of Improvement for minimization
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -20, 20)  # Clip to avoid numerical issues
    
    poi = norm.cdf(z)
    
    # POI is in [0, 1], higher means more likely to improve
    # But we also want to prefer points that are likely to be much better
    # Use a weighted combination: POI * (f_best - mu) to reward expected gain
    # Actually, let's use a different approach: use the quantile of the predictive distribution
    # The point that is likely to be the minimum is the one with low predictive mean
    # but also we want some exploration
    
    # Use a hybrid: primarily POI but with a small UCB component for exploration
    # UCB for minimization: -mu + beta * sigma
    # Let's use a fixed small beta for exploration
    
    # Adaptive beta based on how many points we've evaluated (proxied by cost)
    # Early on: more exploration, later: more exploitation
    mean_cost = np.mean(cost_safe)
    
    # If mean cost is low, we're early; if high, we're late
    # Cost starts around 1-2 and grows
    if mean_cost < 2.0:
        beta = 1.5  # More exploration early
    elif mean_cost < 5.0:
        beta = 1.0  # Moderate exploration
    elif mean_cost < 10.0:
        beta = 0.5  # Less exploration
    else:
        beta = 0.2  # Mostly exploitation
    
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize both to have zero mean and unit variance for combination
    poi_mean = np.mean(poi)
    poi_std = np.std(poi)
    poi_norm = (poi - poi_mean) / (poi_std + 1e-10)
    
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score)
    ucb_norm = (ucb_score - ucb_mean) / (ucb_std + 1e-10)
    
    # Combine: weight POI more heavily since it's bounded and stable
    # But keep some UCB for exploration
    combined = 0.8 * poi_norm + 0.2 * ucb_norm
    
    # Cost-aware: divide by cost to penalize expensive points
    scores = combined / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return a safe fallback
        return np.zeros(n, dtype=np.float64)
    
    return scores
