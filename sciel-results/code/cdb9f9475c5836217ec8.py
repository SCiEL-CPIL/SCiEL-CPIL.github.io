import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization.
    
    Uses a hybrid approach combining:
    1. Cost-aware Expected Improvement (EI) with a floor
    2. Exploration bonus based on uncertainty
    3. Cost normalization
    
    Args:
        mu: array of shape (2000,), mean predictions.
        sigma: array of shape (2000,), standard deviation predictions.
        f_best: float, best observed value.
        cost: array of shape (2000,), cost of evaluating each point.
        
    Returns:
        array of shape (2000,), scores (higher is better).
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Floor for numerical stability
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # For minimization, we want to maximize the expected reduction
    diff = f_best - mu
    z = diff / sigma
    
    # Standard normal CDF and PDF
    # Use scipy if available, otherwise use approximation
    try:
        from scipy.stats import norm
        Phi = norm.cdf(z)
        phi = norm.pdf(z)
    except ImportError:
        # Approximation using erf
        Phi = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
        phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    
    EI = diff * Phi + sigma * phi
    
    # Cost-aware EI: normalize by cost
    # But also add an exploration bonus to encourage exploring uncertain regions
    # The exploration bonus should be weighted by uncertainty and inversely by cost
    
    # Strategy: combine cost-aware EI with an exploration term
    # Exploration term: sigma / cost (encourages high uncertainty, low cost)
    
    # Let's use a weighted combination
    # Weight 1: cost-aware EI (exploitation)
    # Weight 2: exploration bonus (exploration)
    
    # The balance between exploitation and exploration can be tuned
    # For early iterations, we might want more exploration
    # But we don't know the iteration number, so we use a fixed balance
    
    # Let's try: score = (EI / cost) + alpha * (sigma / cost)
    # where alpha controls the exploration-exploitation trade-off
    
    # Alternatively, use a more principled approach:
    # Cost-aware UCB with adaptive beta
    
    # Let's try multiple strategies and pick the best one based on a simple heuristic
    
    # Strategy 1: Cost-aware EI
    score_ei = EI / cost
    
    # Strategy 2: Cost-aware UCB
    # UCB = f_best - mu + beta * sigma
    # We need to choose beta adaptively
    # A common choice is beta = log(n) where n is the number of samples
    # But we don't have n, so let's use a fixed beta that balances exploration and exploitation
    
    # Let's estimate the scale of the problem
    # The sigma values give us a sense of the uncertainty
    # The cost values give us a sense of the cost landscape
    
    # Use a beta that is proportional to the typical uncertainty
    # This helps balance exploration and exploitation
    
    # Compute typical uncertainty (median sigma)
    typical_sigma = np.median(sigma)
    
    # Beta should be in the range of typical uncertainty
    # Let's use beta = 2.0 * typical_sigma / cost_median
    # This normalizes the exploration bonus by the typical cost
    
    cost_median = np.median(cost)
    beta = 2.0 * typical_sigma / cost_median
    
    score_ucb = (f_best - mu + beta * sigma) / cost
    
    # Strategy 3: Hybrid - combine EI and UCB
    # Use a weighted average
    # The weights can be based on the relative magnitudes
    
    # Let's try: score = max(score_ei, score_ucb)
    # This picks the best of the two strategies for each point
    
    score_hybrid = np.maximum(score_ei, score_ucb)
    
    # Ensure all scores are finite
    score_ei = np.where(np.isfinite(score_ei), score_ei, -np.inf)
    score_ucb = np.where(np.isfinite(score_ucb), score_ucb, -np.inf)
    score_hybrid = np.where(np.isfinite(score_hybrid), score_hybrid, -np.inf)
    
    # Return the hybrid score
    # If all scores are -inf, return a small negative value to avoid issues
    if np.all(score_hybrid == -np.inf):
        return np.zeros_like(mu)
    
    return score_hybrid
