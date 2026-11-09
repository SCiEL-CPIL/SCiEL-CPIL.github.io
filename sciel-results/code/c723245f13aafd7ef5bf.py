"""Cost-aware acquisition for Ackley 2D Bayesian optimization."""
import time
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Ackley 2D minimization.
    
    Uses a blended approach:
    - Probability of Improvement (PI) for smooth exploration
    - Expected Improvement (EI) for exploitation
    - Adaptive UCB for exploration in early stages
    
    Normalized by sqrt(cost) to balance exploration efficiency.
    
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
    
    # Normalize each component to [0, 1] range for stable blending
    # PI is already in [0, 1]
    pi_norm = pi
    
    # EI normalization: scale by typical EI magnitude
    ei_max = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    ei_norm = ei / ei_max
    ei_norm = np.nan_to_num(ei_norm, nan=0.0, posinf=1.0, neginf=0.0)
    
    # UCB normalization: center and scale
    ucb_mean = np.mean(ucb_term)
    ucb_std = np.std(ucb_term)
    if ucb_std > 1e-10:
        ucb_norm = (ucb_term - ucb_mean) / (ucb_std * 3)
        ucb_norm = np.clip(ucb_norm, -1, 1)
        ucb_norm = (ucb_norm + 1) / 2  # map to [0, 1]
    else:
        ucb_norm = np.full_like(ucb_term, 0.5)
    ucb_norm = np.nan_to_num(ucb_norm, nan=0.5, posinf=1.0, neginf=0.0)
    
    # Adaptive weights based on convergence stage
    # Early (high f_best): explore more with UCB
    # Late (low f_best): exploit with PI and EI
    if f_best <= 0:
        w_pi = 0.7
        w_ei = 0.25
        w_ucb = 0.05
    elif f_best < 0.1:
        w_pi = 0.6
        w_ei = 0.3
        w_ucb = 0.1
    elif f_best < 0.5:
        w_pi = 0.5
        w_ei = 0.3
        w_ucb = 0.2
    elif f_best < 2.0:
        w_pi = 0.4
        w_ei = 0.3
        w_ucb = 0.3
    elif f_best < 5.0:
        w_pi = 0.35
        w_ei = 0.3
        w_ucb = 0.35
    else:
        w_pi = 0.3
        w_ei = 0.25
        w_ucb = 0.45
    
    combined = w_pi * pi_norm + w_ei * ei_norm + w_ucb * ucb_norm
    combined = np.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Cost-aware scoring: divide by sqrt(cost)
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
            scores[pos_mask] = (scores[pos_mask] / max_pos) ** 1.3 * max_pos
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
