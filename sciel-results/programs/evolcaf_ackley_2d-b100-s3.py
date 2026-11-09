"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_EPS = 1e-12

def _ackley_2d(x):
    """Ackley 2D function (standard, minimum at 0,0)."""
    x1 = x[:, 0]
    x2 = x[:, 1]
    sum_sq = x1**2 + x2**2
    sum_cos = np.cos(2.0 * np.pi * x1) + np.cos(2.0 * np.pi * x2)
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * sum_sq)) - np.exp(0.5 * sum_cos) + np.e + 20.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a blend of Expected Improvement (EI) and Probability of Improvement (PI)
    with UCB exploration, all cost-adjusted. PI provides smoother exploration
    signal in multimodal landscapes. Adaptive weighting shifts from exploration
    to exploitation as f_best approaches 0.
    
    mu, sigma: (2000,) arrays of GP mean and std in objective units
    f_best: scalar, lowest observed value
    cost: (2000,) array of evaluation costs
    
    Returns: (2000,) finite scores, higher = prefer to evaluate next
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s limit (killed at 330s)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, _EPS)
    
    # Expected Improvement (for minimization)
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (for minimization): P(mu < f_best)
    pi = norm.cdf(z)
    
    # UCB (lower is better for minimization)
    # Adaptive beta: explore more when f_best is far from 0
    f_scale = 30.0
    f_norm = min(max(f_best / f_scale, 0.0), 1.0)
    
    # Beta: high early (exploration), low late (exploitation)
    beta = 0.5 + 2.0 * f_norm
    
    ucb = mu + beta * sigma
    
    # Cost-adjusted scoring
    cost_safe = np.maximum(cost, _EPS)
    
    # Normalize each component to [0, 1]
    ei_max = np.max(ei) if np.max(ei) > _EPS else 1.0
    ei_norm = ei / ei_max
    
    # PI is already in [0, 1]
    pi_norm = pi
    
    # UCB: lower is better. Normalize so that lower UCB gives higher score
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min if ucb_max - ucb_min > _EPS else 1.0
    ucb_norm = (ucb_max - ucb) / ucb_range
    
    # Adaptive weighting: 
    # Early (high f_norm): weight UCB and PI more for exploration
    # Late (low f_norm): weight EI more for exploitation
    # w_ei goes from 0.2 (early) to 0.7 (late)
    # w_pi goes from 0.4 (early) to 0.15 (late)
    # w_ucb goes from 0.4 (early) to 0.15 (late)
    w_ei = 0.2 + 0.5 * (1.0 - f_norm)
    w_pi = 0.4 - 0.25 * (1.0 - f_norm)
    w_ucb = 0.4 - 0.25 * (1.0 - f_norm)
    
    # Score: blend of cost-adjusted components
    # Use cost^0.5 in denominator for smoother cost trade-off
    cost_adj = np.sqrt(cost_safe)
    score = (w_ei * ei_norm + w_pi * pi_norm + w_ucb * ucb_norm) / cost_adj
    
    # Add a small bonus for very low cost to break ties
    cost_bonus = 0.05 * (1.0 / cost_safe - 0.5)
    score += cost_bonus
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
