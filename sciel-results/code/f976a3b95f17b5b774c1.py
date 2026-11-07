# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Pareto-filtered blend of Expected Improvement (EI)
    and Upper Confidence Bound (UCB).
    
    Strategy:
    1. Filter out hopeless candidates where the predicted mean is significantly above f_best.
       This focuses the search on regions where improvement is plausible.
    2. Compute EI for minimization: E[f_best - f(x)] where f(x) ~ N(mu, sigma).
    3. Compute UCB (exploit) and LCB (explore) scores.
    4. Blend EI and UCB with an adaptive weight that shifts towards exploitation
       as f_best approaches the minimum (0).
    5. Return higher scores for more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: ensure finite and positive sigma
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # f_best is the lowest observed value
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 0.0
    
    # Pareto filtering: mask out candidates where mu is much higher than f_best
    # This reduces wasted evaluations in obviously poor regions
    # Use a threshold: if mu > f_best + 2*sigma_median, it's unlikely to be better
    sigma_median = np.median(sigma)
    threshold = f_best + 2.0 * sigma_median
    
    # Compute Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)] where f(x) ~ N(mu, sigma)
    # = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB score (higher is better for exploitation + exploration)
    # UCB = mu + beta_ucb * sigma  (we want to find low values, so we actually want LCB for minimization)
    # For minimization, LCB = mu - beta * sigma is the lower bound
    # We want to minimize LCB, so score_ucb = -LCB = beta * sigma - mu
    # But let's use a standard UCB formulation for maximization of the negative
    # Actually, for minimization, we want to find points with low mu and high sigma
    # UCB for minimization: score = mu + beta * sigma (this is an upper bound, but we want lower)
    # Let's use: score_ucb = -mu + beta_ucb * sigma (encourages low mu and high sigma)
    
    # Adaptive beta: start high for exploration, decrease for exploitation
    # f_best ranges from ~1.0 (early) to ~0.0 (late)
    f_best_clamped = np.clip(f_best, 0.0, 2.0)
    beta_ucb = 2.0 + 3.0 * f_best_clamped
    
    score_ucb = -mu + beta_ucb * sigma
    
    # Blend EI and UCB
    # Early: more UCB (exploration), Late: more EI (exploitation)
    # weight_ei goes from 0.3 to 0.7 as f_best goes from 1.0 to 0.0
    weight_ei = 0.3 + 0.4 * (1.0 - f_best_clamped)
    weight_ucb = 1.0 - weight_ei
    
    # Normalize scores for blending
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    ucb_max = np.max(score_ucb)
    ucb_min = np.min(score_ucb)
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = (score_ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = score_ucb
    
    # Blend
    score = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Apply Pareto filter: heavily penalize hopeless candidates
    # If mu > threshold, reduce score significantly
    mask = mu <= threshold
    score = np.where(mask, score, score * 0.1)
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e6)
    
    return score
# EVOLVE-BLOCK-END
