"""Cost-aware acquisition for Ackley 2D Bayesian optimization."""
import time
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with
    adaptive exploration and rank-based cost normalization.
    
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
    
    n = len(mu)
    
    # Sanitize inputs
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e-2, neginf=1e-10)
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e4, neginf=-1e4)
    cost = np.maximum(cost, 1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Deadline guard
    if time.monotonic() - t_start > 20:
        scores = (-mu + 2.0 * sigma) / cost
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e4, neginf=0.0)
    
    # Adaptive beta based on f_best - more exploration when far from optimum
    if f_best <= 0:
        beta = 0.3
    elif f_best < 0.1:
        beta = 0.5
    elif f_best < 0.5:
        beta = 0.8
    elif f_best < 2.0:
        beta = 1.2
    elif f_best < 5.0:
        beta = 1.8
    else:
        beta = 2.5
    
    ucb_term = beta * sigma - mu
    ucb_term = np.nan_to_num(ucb_term, nan=0.0, posinf=1e4, neginf=-1e4)
    
    # Z-score normalization for robustness
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std > 1e-10:
        ei_norm = (ei - ei_mean) / ei_std
    else:
        ei_norm = np.zeros_like(ei)
    ei_norm = np.nan_to_num(ei_norm, nan=0.0, posinf=1e6, neginf=-1e6)
    
    ucb_mean = np.mean(ucb_term)
    ucb_std = np.std(ucb_term)
    if ucb_std > 1e-10:
        ucb_norm = (ucb_term - ucb_mean) / ucb_std
    else:
        ucb_norm = np.zeros_like(ucb_term)
    ucb_norm = np.nan_to_num(ucb_norm, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Adaptive weights: smooth transition from exploration to exploitation
    if f_best <= 0:
        w_ei = 0.9
        w_ucb = 0.1
    else:
        # Smooth decay: as f_best increases, explore more
        w_ucb = 0.1 + 0.5 * (1.0 - np.exp(-f_best / 3.0))
        w_ei = 1.0 - w_ucb
    
    combined = w_ei * ei_norm + w_ucb * ucb_norm
    combined = np.nan_to_num(combined, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Cost-aware scoring: divide by cost (not sqrt) for stronger penalty
    scores = combined / cost
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline check
    if time.monotonic() - t_start > 28:
        return scores
    
    # Rank-based shaping: amplify top candidates to sharpen selection
    # This helps the "pick best of 2000" strategy
    pos_mask = scores > 0
    if np.any(pos_mask):
        max_pos = np.max(scores[pos_mask])
        if max_pos > 1e-10:
            # Power transformation: alpha > 1 amplifies high scores
            scores[pos_mask] = (scores[pos_mask] / max_pos) ** 1.4 * max_pos
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
