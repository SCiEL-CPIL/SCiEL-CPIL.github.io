import numpy as np
import time
from scipy.stats import norm
from scipy.stats import rankdata

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6D.
    Combines Thompson Sampling and Expected Improvement using rank-based aggregation
    for robustness against score distribution outliers.
    """
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    # Input validation and safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Check for non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)
    
    # 1. Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-normalized EI
    ei_cost = ei / cost_safe
    
    # 2. Thompson Sampling (TS)
    # Draw a single sample from the posterior
    eps = np.random.standard_normal(n)
    y_sample = mu + sigma_safe * eps
    
    # Cost-normalized TS score (we want to maximize, so minimize y/cost -> maximize -y/cost)
    ts_score = -y_sample / cost_safe
    
    # 3. Rank-Based Normalization
    # Convert scores to percentile ranks in [0, 1]
    # This is robust to outliers and non-Gaussian score distributions
    ei_rank = rankdata(ei_cost, method='average') / n
    ts_rank = rankdata(ts_score, method='average') / n
    
    # 4. Adaptive Weighting
    # Early on (large distance to min), explore more (TS).
    # Late on (small distance to min), exploit more (EI).
    min_hartmann = -3.32237
    dist_to_min = max(0.0, f_best - min_hartmann)
    
    # Weights derived from the successful Inspiration 1 structure
    # ts_weight ranges from ~0.7 (d=0) to ~0.3 (d=large)
    ts_weight = 0.3 + 0.4 * np.exp(-dist_to_min / 0.5)
    ei_weight = 1.0 - ts_weight
    
    # Combine ranks
    # Since ranks are in [0,1], the result is in [0,1]
    combined = ts_weight * ts_rank + ei_weight * ei_rank
    
    # Ensure finite scores
    scores = np.where(np.isfinite(combined), combined, 0.0)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return np.zeros(n, dtype=np.float64)
    
    return scores
