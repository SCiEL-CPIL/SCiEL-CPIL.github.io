import numpy as np
from scipy.stats import norm
import time

# Module-level cache to track elapsed time
_acq_state = {
    'start_time': None,
}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses batch Expected Improvement (min-EI marginal contribution) with cost awareness.
    The marginal contribution of each point accounts for redundancy with other candidates.
    """
    # Initialize timing on first call
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    
    # Deadline guard: if we're running out of time, return a safe fallback
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 320:  # Safety margin before 330s limit
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1e-9)
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure valid inputs
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Compute individual Expected Improvement ---
    z = (f_best - mu) / sigma_safe
    ei_individual = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei_individual = np.maximum(ei_individual, 0.0)
    
    # --- Batch EI: marginal contribution of each point ---
    # p_improve_i = P(f(x_i) < f_best) = Phi(z_i)
    p_improve = norm.cdf(z)
    p_improve = np.clip(p_improve, 1e-12, 1.0 - 1e-12)
    
    # log(1 - p_i) for each point
    log_1_minus_p = np.log1p(-p_improve)
    log_prod_all = np.sum(log_1_minus_p)
    
    # Marginal contribution: ei_i * prod_{j != i}(1 - p_j)
    # = ei_i * exp(log_prod_all - log_1_minus_p_i)
    # Compute in log space for numerical stability
    log_ei = np.log(ei_individual + 1e-300)
    log_marginal = log_ei + log_prod_all - log_1_minus_p
    
    # Clip to avoid overflow/underflow
    log_marginal = np.clip(log_marginal, -700, 700)
    marginal = np.exp(log_marginal)
    
    # Cost-aware score
    score = marginal / cost_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
