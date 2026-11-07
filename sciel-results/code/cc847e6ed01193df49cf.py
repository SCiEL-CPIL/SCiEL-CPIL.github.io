"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track global start time for deadline safety
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to spare before 330s kill

# Constants
MIN_POSSIBLE = -0.8
MAX_POSSIBLE = 1.0  # Approx max of cosine8

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a stochastic knapsack-inspired metric.
    
    Structural Change:
    Replaces raw Expected Improvement (EI) with a probability-weighted improvement
    metric: P(f(x) < f_best) * (f_best - mu) / cost.
    
    This formulation directly maximizes the expected regret reduction per unit cost
    under a probabilistic decision rule, rather than the raw expected improvement magnitude.
    It inherently down-weights high-cost candidates with marginal improvement potential
    and high-uncertainty candidates that are unlikely to beat the current best.
    
    To balance exploration and exploitation, we blend this core metric with a
    robustly normalized exploration term (sigma / cost). The blend weight adapts
    based on progress: when f_best is far from the known minimum, we explore more;
    when close, we exploit the probability-weighted improvement.
    
    Robust normalization (median/MAD) is applied to both terms to ensure they are
    on comparable scales, preventing one term from dominating due to outlier values.
    """
    # Deadline guard
    if time.time() - _START_TIME > 330.0 - _DEADLINE_MARGIN:
        return np.zeros_like(mu)

    n = len(mu)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1.0)
    
    # 1. Core Metric: Probability-Weighted Improvement per Cost
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # Expected Improvement Magnitude (approx) = (f_best - mu) when mu < f_best
    # We use the product P * (f_best - mu) which is related to the partial expectation.
    # Actually, E[max(0, f_best - f)] = (f_best - mu)Phi(z) + sigma*phi(z)
    # The proposed metric is P * (f_best - mu) / cost.
    # Let's stick to the directive: P(mu < f_best) * (f_best - mu) / cost
    # Note: (f_best - mu) can be negative if mu > f_best.
    # If mu > f_best, P is small, but (f_best - mu) is negative.
    # We should clamp the improvement part to be non-negative?
    # The directive says "P(mu < f_best) * (f_best - mu) / cost".
    # If mu > f_best, (f_best - mu) is negative. The product would be negative.
    # Since we want to maximize score, negative scores are bad.
    # However, P(mu < f_best) is already small.
    # Let's use max(0, f_best - mu) to ensure we only consider potential improvements.
    # Or strictly follow the formula?
    # "optimizing the probability-weighted improvement P(mu < f_best) * (f_best - mu) / cost"
    # Usually improvement is defined as max(0, f_best - f).
    # Let's use np.maximum(f_best - mu, 0) to be safe and align with "improvement".
    
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -20, 20)
    poi = norm.cdf(z)  # P(f(x) < f_best)
    
    improvement_magnitude = np.maximum(f_best - mu, 0.0)
    
    # Core score: Probability-weighted improvement per cost
    core_score = poi * improvement_magnitude / cost_safe
    
    # 2. Exploration Term: Uncertainty per Cost
    # High sigma indicates potential for finding better points.
    explore_score = sigma_safe / cost_safe
    
    # 3. Robust Normalization (Median/MAD)
    # Normalize core_score
    core_med = np.median(core_score)
    core_mad = np.median(np.abs(core_score - core_med)) + 1e-12
    core_norm = (core_score - core_med) / core_mad
    
    # Normalize explore_score
    expl_med = np.median(explore_score)
    expl_mad = np.median(np.abs(explore_score - expl_med)) + 1e-12
    expl_norm = (explore_score - expl_med) / expl_mad
    
    # 4. Adaptive Blending
    # Progress: 0 when f_best is near -0.8, 1 when f_best is far (e.g., 0.0)
    # f_range is roughly 1.8
    f_range = MAX_POSSIBLE - MIN_POSSIBLE
    if f_range <= 0:
        progress = 0.5
    else:
        progress = np.clip((f_best - MIN_POSSIBLE) / f_range, 0.0, 1.0)
    
    # Exploration weight: higher when progress is high (far from optimum)
    # Use a smooth transition.
    # When progress is 1 (bad best), we want more exploration.
    # When progress is 0 (good best), we want more exploitation (core_score).
    # w_explore = 0.2 + 0.6 * progress**1.5
    w_explore = 0.2 + 0.6 * (progress ** 1.5)
    w_exploit = 1.0 - w_explore
    
    # Combine
    score = w_exploit * core_norm + w_explore * expl_norm
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
