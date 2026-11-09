"""Cost-aware acquisition for Ackley 2D Bayesian optimization."""
import time
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Probability of Improvement (PI)
    with adaptive exploration for Ackley 2D minimization.
    
    Uses a PI-based metric normalized by sqrt(cost) for stable gradients,
    blended with EI for exploitation and UCB for exploration.
    
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
        scores = (-mu + 2.0 * sigma) / np.sqrt(cost)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Probability of Improvement for minimization
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    improvement = f_best - mu
    z = improvement / sigma
    pi = norm.cdf(z)
    pi = np.clip(pi, 0.0, 1.0)
    pi = np.nan_to_num(pi, nan=0.5, posinf=1.0, neginf=0.0)
    
    # Expected Improvement for exploitation signal
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
    
    # Z-score normalization for robustness
    pi_mean = np.mean(pi)
    pi_std = np.std(pi)
    if pi_std > 1e-10:
        pi_norm = (pi - pi_mean) / pi_std
    else:
        pi_norm = np.zeros_like(pi)
    
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std > 1e-10:
        ei_norm = (ei - ei_mean) / ei_std
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_mean = np.mean(ucb_term)
    ucb_std = np.std(ucb_term)
    if ucb_std > 1e-10:
        ucb_norm = (ucb_term - ucb_mean) / ucb_std
    else:
        ucb_norm = np.zeros_like(ucb_term)
    
    # Adaptive weights: PI is primary for smooth gradients, EI for exploitation, UCB for exploration
    # When f_best is low (near optimum), exploit more
    # When f_best is high (far from optimum), explore more
    if f_best <= 0:
        w_pi = 0.6
        w_ei = 0.3
        w_ucb = 0.1
    elif f_best < 1.0:
        w_pi = 0.5
        w_ei = 0.3
        w_ucb = 0.2
    elif f_best < 3.0:
        w_pi = 0.4
        w_ei = 0.3
        w_ucb = 0.3
    else:
        w_pi = 0.3
        w_ei = 0.3
        w_ucb = 0.4
    
    combined = w_pi * pi_norm + w_ei * ei_norm + w_ucb * ucb_norm
    
    # Cost-aware scoring: divide by sqrt(cost) for balanced exploration efficiency
    # Cost ranges from 1 to 2, so sqrt(cost) ranges from 1 to ~1.41
    scores = combined / np.sqrt(cost)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline check
    if time.monotonic() - t_start > 28:
        return scores
    
    # Mild sharpening to emphasize top candidates
    pos_mask = scores > 0
    if np.any(pos_mask):
        max_pos = np.max(scores[pos_mask])
        if max_pos > 1e-10:
            scores[pos_mask] = (scores[pos_mask] / max_pos) ** 1.25 * max_pos
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
