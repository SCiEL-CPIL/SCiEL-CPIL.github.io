"""Cost-aware acquisition for cosine8 8d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level cache for time tracking
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s kill to stop heavy work


def _elapsed():
    return time.time() - _start_time


def _time_exhausted():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition with dynamic xi, smooth uncertainty penalty, and
    adaptive exploration weight.
    
    Key refinements over baseline:
    1. Dynamic xi (exploitation bias) that grows with budget consumption
    2. Smooth uncertainty penalty using a soft ramp instead of hard max
    3. Adaptive PoI weight that decays as budget depletes (exploit more late)
    4. Cost-scaled xi: expensive points need more confidence to justify
    """
    n = len(mu)
    
    # Safety: if time is running out, return a simple score quickly
    if _time_exhausted():
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # --- Dynamic xi: exploitation bias grows with budget consumption ---
    # cost is in [1, 2] (1 + mean(x_scaled)). Total budget is 30.
    # Early: xi small (explore). Late: xi larger (exploit).
    # Estimate budget fraction used: cost array represents per-point costs.
    # We don't have total_cost directly, but f_best and the number of evaluations
    # give a proxy. Use the mean cost as a proxy for "how expensive evaluations are."
    # A better proxy: the spread of sigma tells us how much we've learned.
    # Low sigma spread = late in optimization (exploit). High spread = early (explore).
    
    sigma_median = np.median(sigma_safe)
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe)
    
    # Normalized uncertainty: low when we've learned a lot (late), high when uncertain (early)
    # Use ratio of std to median as a "learning progress" indicator
    uncertainty_ratio = sigma_std / max(sigma_median, 1e-8)
    # uncertainty_ratio ~ 0.5+ early, ~0.1+ late
    # Map to xi: early xi=0.01, late xi=0.05
    xi = 0.01 + 0.04 * (1.0 - min(uncertainty_ratio, 1.0))
    # Also scale xi by cost: expensive points should have higher hurdle
    cost_mean = np.mean(cost)
    cost_scale = (cost_mean - 1.0)  # in [0, 1]
    xi = xi * (1.0 + 0.5 * cost_scale)
    
    # --- EI with xi ---
    # EI_xi = E[max(0, f_best - xi - f(x))]
    # z = (f_best - xi - mu) / sigma
    z = (f_best - xi - mu) / sigma_safe
    ei_xi = (f_best - xi - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Clip negative EI to 0 (EI is always non-negative by definition, but numerically)
    ei_xi = np.maximum(ei_xi, 0.0)
    
    # --- PoI with xi ---
    # P(f(x) < f_best - xi) = Phi((f_best - xi - mu) / sigma)
    poi_xi = norm.cdf(z)
    
    # --- Adaptive PoI weight: decay as budget depletes ---
    # Early: weight PoI more (exploration). Late: weight EI more (exploitation).
    # Use uncertainty_ratio: high = early, low = late
    poi_weight = 0.3 + 0.4 * min(uncertainty_ratio, 1.0)
    
    # Blend EI and PoI
    blended = ei_xi + poi_weight * sigma_safe * poi_xi
    
    # --- Smooth uncertainty penalty ---
    # Instead of hard max(0, median - sigma), use a smooth ramp:
    # penalty = median * (1 - exp(-(median - sigma) / (median + eps)))
    # This is 0 when sigma >= median, and smoothly approaches median as sigma -> 0
    eps = 1e-8
    delta_sigma = np.maximum(0.0, sigma_median - sigma_safe)
    # Soft ramp: smooth approximation of min(delta, some_cap)
    penalty_raw = sigma_median * (1.0 - np.exp(-delta_sigma / (sigma_median + eps)))
    penalty_raw = penalty_raw / max(sigma_median, eps)
    
    # Scale penalty
    lambda_penalty = 0.25
    penalty = lambda_penalty * penalty_raw
    
    # Cost normalization
    cost_safe = np.maximum(cost, 1e-6)
    
    # Final score: blended improvement per unit cost, minus uncertainty penalty
    scores = (blended - penalty * sigma_safe) / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.clip(scores, -10.0, 1e6)
    
    return scores
# EVOLVE-BLOCK-END
