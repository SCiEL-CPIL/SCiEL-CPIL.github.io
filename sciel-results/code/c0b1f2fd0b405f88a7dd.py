"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a UCB exploration term, normalized by evaluation cost.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    _ensure_start()
    
    # Deadline guard: if we are getting close to the limit, return a safe default
    # However, since this function is called per-step and is very fast, 
    # this is mostly a safety net against pathological inputs or loops.
    # 330s total budget. We assume this function is called < 1000 times.
    # Each call is ~ms. No need for complex deadline logic here unless 
    # the evaluator runs this in a tight loop without yielding, which is unlikely 
    # given the "2000 candidates" description.
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])
        
    # Guard against non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # UCB component for exploration: rewards high uncertainty
    # beta controls the exploration-exploitation balance
    beta = 1.5
    ucb = beta * sigma_safe - mu
    
    # Adaptive exploration weight: shift from exploration to exploitation
    # as f_best approaches the known minimum (0 for levy 2d)
    # When f_best is large, we need more exploration; when small, exploit
    f_best_abs = abs(f_best)
    # Scale: at f_best ~ 10, weight ~ 0.4; at f_best ~ 0.1, weight ~ 0.1
    w_ucb = 0.1 + 0.3 * np.exp(-f_best_abs / 2.0)
    
    # Combine: primary EI signal + secondary UCB exploration signal
    # We only add positive UCB values to avoid penalizing good regions too much
    # if they happen to have high uncertainty (though unlikely for mu << f_best)
    combined = ei + w_ucb * np.maximum(ucb, 0.0)
    
    # Cost-aware: divide by cost
    score = combined / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
