"""Rewritten acquisition for Bayesian optimization of the 2-D Beale function."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330 s wall-clock.
# We keep a small safety margin.
_DEADLINE = 320.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def acquisition(mu, sigma, f_best):
    """
    Return 2000 finite scores for the candidates.
    Higher score means "evaluate next".

    We use a combination of:
      - Expected Improvement (EI)
      - Lower Confidence Bound (LCB)
    blended with a dynamic weight that shifts from exploration (LCB) to
    exploitation (EI) as f_best improves.

    Additionally, we add a small uncertainty bonus to avoid zero-gradient
    regions and encourage exploration in high-uncertainty areas.
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        # Fallback: return uniform random scores
        return np.random.rand(len(mu))

    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)

    # --- Expected Improvement ---
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)

    # --- Lower Confidence Bound ---
    # beta controls exploration; we use a moderate value
    beta = 2.0
    lcb = mu - beta * sigma_safe

    # --- Uncertainty bonus ---
    # Encourage exploration where sigma is large
    unc_bonus = sigma_safe

    # Dynamic blending:
    # When f_best is still large (far from optimum), explore more (weight LCB).
    # As f_best approaches 0, exploit more (weight EI).
    # Beale min is 0, domain [-4.5, 4.5]^2. Initial random f_best ~ O(10).
    # Use a smooth transition based on |f_best|.
    # scale: when f_best ~ 0, weight_ei -> 1; when f_best ~ 10, weight_ei -> 0.2
    weight_ei = 1.0 / (1.0 + np.exp(-0.5 * np.log1p(max(f_best, 1e-6))))
    # Actually simpler: use a fixed blend that works well in practice
    # Let's use: score = w1 * EI + w2 * (-LCB) + w3 * unc_bonus
    # Normalize EI and LCB to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    lcb_min = np.min(lcb) if np.min(lcb) < 0 else 0.0
    lcb_range = -lcb_min if lcb_min < 0 else 1.0

    # Simple robust approach: use EI as primary, add small LCB and uncertainty
    # This is a structural change from pure EI: we blend multiple acquisition
    # functions with adaptive weights based on the current f_best.

    # Adaptive beta: start with more exploration, reduce as we get closer
    # f_best is the best observed so far; Beale's minimum is 0
    # Use log-scale: when f_best is large, explore more
    if f_best > 1.0:
        beta_adaptive = 3.0
    elif f_best > 0.1:
        beta_adaptive = 2.0
    else:
        beta_adaptive = 1.0

    lcb_adaptive = mu - beta_adaptive * sigma_safe

    # Blend: primarily EI, with LCB and uncertainty as secondary signals
    # Normalize each component to [0, 1] range for fair blending
    ei_norm = ei / (ei_max + 1e-10)
    lcb_norm = (lcb_adaptive - np.min(lcb_adaptive)) / (np.max(lcb_adaptive) - np.min(lcb_adaptive) + 1e-10)
    unc_norm = sigma_safe / (np.max(sigma_safe) + 1e-10)

    # Weights: as f_best decreases, shift from exploration to exploitation
    if f_best > 5.0:
        w_ei, w_lcb, w_unc = 0.3, 0.4, 0.3
    elif f_best > 1.0:
        w_ei, w_lcb, w_unc = 0.5, 0.3, 0.2
    elif f_best > 0.1:
        w_ei, w_lcb, w_unc = 0.7, 0.2, 0.1
    else:
        w_ei, w_lcb, w_unc = 0.85, 0.1, 0.05

    score = w_ei * ei_norm + w_lcb * lcb_norm + w_unc * unc_norm

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
