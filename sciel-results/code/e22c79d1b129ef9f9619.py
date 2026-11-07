import time
import numpy as np

_STATE = {
    "start_time": None,
}

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if _STATE["start_time"] is None:
        _STATE["start_time"] = time.time()
    elapsed = time.time() - _STATE["start_time"]
    if elapsed > 300.0:
        # Fallback to simple LCB/Cost if timeout imminent
        sigma_safe = np.maximum(sigma, 1e-10)
        scores = (f_best - mu - 2.0 * sigma_safe) / np.sqrt(np.maximum(cost, 1e-10))
        return np.where(np.isfinite(scores), scores, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.sqrt(np.maximum(cost, eps))

    # Thompson Sampling: Draw noise
    # Using a single draw per call is standard for TS in this context
    z = np.random.randn(len(mu))
    
    # Posterior predictive sample
    theta = mu + sigma_safe * z
    
    # Score: Maximize (f_best - theta) / sqrt(cost)
    scores = (f_best - theta) / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
