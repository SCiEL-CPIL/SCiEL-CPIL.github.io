# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a smooth blend of Expected Improvement (EI)
    and Lower Confidence Bound (LCB) for minimization.
    
    Strategy:
    1. Compute EI for minimization: E[max(f_best - f(x), 0)] where f(x) ~ N(mu, sigma).
    2. Compute LCB score: -mu + beta * sigma (encourages low mu and high sigma).
    3. Use an adaptive weight that shifts from exploration (LCB) to exploitation (EI)
       as f_best approaches the minimum (0).
    4. Apply a smooth penalty for candidates with high predicted mean relative to f_best.
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
    
    # Compute Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)] where f(x) ~ N(mu, sigma)
    # = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute LCB score for minimization
    # LCB = mu - beta * sigma (lower bound on f(x))
    # We want to minimize LCB, so score_lcb = -LCB = beta * sigma - mu
    # This encourages low mu and high sigma (exploration)
    
    # Adaptive beta: start high for exploration, decrease for exploitation
    # f_best ranges from ~1.0 (early) to ~0.0 (late)
    f_best_clamped = np.clip(f_best, 0.0, 2.0)
    beta_lcb = 1.5 + 2.5 * f_best_clamped
    
    score_lcb = beta_lcb * sigma - mu
    
    # Normalize EI and LCB scores to [0, 1] for blending
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max - ei_min > 1e-12:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    lcb_min = np.min(score_lcb)
    lcb_max = np.max(score_lcb)
    if lcb_max - lcb_min > 1e-12:
        lcb_norm = (score_lcb - lcb_min) / (lcb_max - lcb_min)
    else:
        lcb_norm = np.zeros_like(score_lcb)
    
    # Adaptive weight: shift from LCB (exploration) to EI (exploitation)
    # When f_best is high (early), weight more on LCB
    # When f_best is low (late), weight more on EI
    weight_ei = 0.2 + 0.6 * (1.0 - f_best_clamped / 2.0)
    weight_lcb = 1.0 - weight_ei
    
    # Blend
    score = weight_ei * ei_norm + weight_lcb * lcb_norm
    
    # Smooth penalty for candidates with high predicted mean
    # Use a sigmoid-like penalty that increases smoothly as mu exceeds f_best
    sigma_median = np.median(sigma)
    excess = (mu - f_best) / (sigma_median + 1e-9)
    # Penalty: 0 when excess <= 0, approaches 1 as excess increases
    penalty = 1.0 / (1.0 + np.exp(-excess))
    # Apply penalty: reduce score for high-mu candidates
    score = score * (1.0 - 0.5 * penalty)
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e6)
    
    return score
# EVOLVE-BLOCK-END
