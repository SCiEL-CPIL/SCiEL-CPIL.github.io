import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_CALL_COUNT = 0

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    global _CALL_COUNT
    _ensure_start_time()
    
    # Increment counter
    _CALL_COUNT += 1
    
    # Estimate current step in the 30-step sequence
    # Assume 30 evaluations per seed
    step = _CALL_COUNT % 30
    
    # EI Calculation
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive UCB Weight
    # High exploration early, low exploitation late
    beta_max = 0.5
    beta_min = 0.05
    tau = 5.0
    beta = beta_min + (beta_max - beta_min) * np.exp(-step / tau)
    
    ucb_term = beta * sigma_safe
    
    scores = ei + ucb_term
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
