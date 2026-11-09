"""Cost-aware acquisition for Ackley 2D Bayesian optimization."""
import time
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using expected improvement with
    adaptive exploration for Ackley 2D minimization.
    
    Args:
        mu: array of GP mean predictions, shape (2000,)
        sigma: array of GP std predictions, shape (2000,)
        f_best: current best observed value (float)
        cost: array of evaluation costs, shape (2000,)
    
    Returns:
        array of acquisition scores, shape (2000,), higher = prefer
    """
    t_start = time.monotonic()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e-2, neginf=1e-10)
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e4, neginf=-1e4)
    cost = np.maximum(cost, 1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e4, neginf=0.0)
    
    # Adaptive beta based on f_best
    if f_best <= 0:
        beta = 0.5
    elif f_best < 0.1:
        beta = 0.8
    elif f_best < 0.5:
        beta = 1.2
    elif f_best < 2.0:
        beta = 1.8
    elif f_best < 5.0:
        beta = 2.5
    else:
        beta = 3.0
    
    ucb_term = beta * sigma - mu
    ucb_term = np.nan_to_num(ucb_term, nan=0.0, posinf=1e4, neginf=-1e4)
    
    # Bounded [0,1] normalization to prevent outlier dominance
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_norm = ei / (ei_max + 1e-10)
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_max = np.max(ucb_term)
    ucb_min = np.min(ucb_term)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-10:
        ucb_norm = (ucb_term - ucb_min) / (ucb_range + 1e-10)
    else:
        ucb_norm = np.zeros_like(ucb_term)
    
    # Adaptive weights: more exploitation as we converge
    if f_best <= 0:
        w_ei = 0.95
        w_ucb = 0.05
    elif f_best < 0.1:
        w_ei = 0.9
        w_ucb = 0.1
    else:
        w_ucb = 0.15 + 0.55 * (1.0 - np.exp(-f_best / 2.0))
        w_ei = 1.0 - w_ucb
    
    combined = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Cost-aware scoring
    scores = combined / cost
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Mild sharpening
    pos_mask = scores > 0
    if np.any(pos_mask):
        max_pos = np.max(scores[pos_mask])
        if max_pos > 1e-10:
            scores[pos_mask] = (scores[pos_mask] / max_pos) ** 1.2 * max_pos
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
