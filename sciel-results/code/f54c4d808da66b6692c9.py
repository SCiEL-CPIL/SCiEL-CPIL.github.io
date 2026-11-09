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
    
    Blends Expected Improvement (EI), Probability of Improvement (PI), and UCB
    with adaptive weighting and cost^0.5 normalization. PI provides smooth
    exploration signal in multimodal landscapes. Weighting shifts from
    exploration to exploitation as f_best approaches 0.
    
    mu, sigma: (2000,) arrays of GP mean and std in objective units
    f_best: scalar, lowest observed value
    cost: (2000,) array of evaluation costs
    
    Returns: (2000,) finite scores, higher = prefer to evaluate next
    """
    start_time = time.time()
    deadline = start_time + 250.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)

    sigma = np.maximum(sigma, _EPS)

    # Expected Improvement (for minimization)
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Probability of Improvement (for minimization): P(mu < f_best)
    pi = norm.cdf(z)

    # UCB (lower is better for minimization)
    # Adaptive beta with smoother decay
    f_scale = 25.0
    f_norm = min(max(f_best / f_scale, 0.0), 1.0)

    # Use a smooth sigmoid-like decay for beta
    # beta starts around 2.0 when f_norm=1, decays to ~0.4 when f_norm=0
    beta = 0.4 + 1.6 * f_norm**0.7

    ucb = mu + beta * sigma

    # Cost-adjusted scoring
    cost_safe = np.maximum(cost, _EPS)

    # Normalize each component to [0, 1]
    ei_max = np.max(ei) if np.max(ei) > _EPS else 1.0
    ei_norm = ei / ei_max

    pi_norm = pi

    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min if ucb_max - ucb_min > _EPS else 1.0
    ucb_norm = (ucb_max - ucb) / ucb_range

    # Adaptive weighting with smoother transition
    # Use a smooth interpolation based on f_norm
    # Early: w_ei~0.25, w_pi~0.35, w_ucb~0.40
    # Late:  w_ei~0.70, w_pi~0.15, w_ucb~0.15
    t = 1.0 - f_norm  # t=0 early, t=1 late
    # Smooth step for better transition
    t_smooth = t * t * (3.0 - 2.0 * t)

    w_ei = 0.25 + 0.45 * t_smooth
    w_pi = 0.35 - 0.20 * t_smooth
    w_ucb = 0.40 - 0.25 * t_smooth

    # Normalize weights to sum to 1
    w_sum = w_ei + w_pi + w_ucb
    w_ei /= w_sum
    w_pi /= w_sum
    w_ucb /= w_sum

    # Score: blend of cost-adjusted components
    # Use cost^0.5 in denominator for smoother cost trade-off
    cost_adj = np.sqrt(cost_safe)
    score = (w_ei * ei_norm + w_pi * pi_norm + w_ucb * ucb_norm) / cost_adj

    # Add a small bonus for low cost to break ties
    # cost is in [1, 2], so (1/cost - 0.5) is in [0, 0.5]
    cost_bonus = 0.03 * (1.0 / cost_safe - 0.5)
    score += cost_bonus

    # Check deadline
    if time.time() > deadline:
        pass

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
