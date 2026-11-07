"""Cost-aware acquisition for BO on three hump camel 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Expected Improvement with information-gain
    weighting to balance exploitation and exploration efficiency.
    
    Strategy:
    1. Compute Expected Improvement (EI) for minimization.
    2. Weight by an entropy-based information gain proxy: regions with high
       uncertainty relative to current best are more informative.
    3. Apply cost-aware scaling to prefer cheaper evaluations.
    
    Parameters:
        mu: (2000,) GP mean predictions in objective units
        sigma: (2000,) GP std dev predictions (positive) in objective units
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost per candidate
    
    Returns:
        (2000,) finite acquisition scores, higher = better
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return np.zeros(len(mu), dtype=np.float64)
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standardized improvement for minimization
    Z = (f_best - mu) / sigma_safe
    Z = np.clip(Z, -10.0, 10.0)
    
    from scipy.stats import norm
    phi = norm.pdf(Z)
    Phi = norm.cdf(Z)
    
    # Expected Improvement
    EI = (f_best - mu) * Phi + sigma_safe * phi
    EI = np.maximum(EI, 0.0)
    
    # Information gain proxy: high when sigma is large relative to f_best scale
    # This encourages exploring uncertain regions that could contain better minima
    # Use normalized uncertainty: sigma / (|f_best| + sigma_max + eps)
    sigma_max = np.max(sigma_safe) if n > 0 else 1.0
    f_best_scale = abs(f_best) + sigma_max + 1e-8
    info_gain = sigma_safe / f_best_scale
    info_gain = np.clip(info_gain, 0.0, 1.0)
    
    # Combine EI with information gain:
    # score = EI * (1 + beta * info_gain)
    # This boosts exploration in uncertain regions while keeping exploitation
    beta = 0.5
    combined = EI * (1.0 + beta * info_gain)
    
    # Cost-aware scaling
    # cost in [1, 2], prefer cheaper points
    cost_clamped = np.clip(cost, 1.0, 2.0)
    cost_penalty = 1.0 + 0.3 * (cost_clamped - 1.0)
    score = combined / cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
