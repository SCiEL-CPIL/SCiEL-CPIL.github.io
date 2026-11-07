# EVOLVE-BLOCK-START
import time
import numpy as np
from math import erf, sqrt, log, exp, pi

_T0 = time.time()
_DEADLINE = _T0 + 320.0  # safety margin vs 330s kill

_INV_SQRT2PI = 1.0 / sqrt(2.0 * pi)
_INV_SQRT2 = 1.0 / sqrt(2.0)


def _norm_cdf_vec(z):
    # Vectorized normal CDF using erf
    return 0.5 * (1.0 + np.vectorize(erf, otypes=[float])(z * _INV_SQRT2))


def _norm_pdf_vec(z):
    return _INV_SQRT2PI * np.exp(-0.5 * z * z)


def _ei(mu, sigma, fb):
    s = np.maximum(sigma, 1e-8)
    z = (fb - mu) / s
    return (fb - mu) * _norm_cdf_vec(z) + s * _norm_pdf_vec(z)


def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    fb = float(f_best)

    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe default (e.g., pure exploitation) if we are out of time
        return -mu

    s = np.maximum(sigma, 1e-8)

    # 1. Compute Expected Improvement (EI)
    ei_scores = _ei(mu, s, fb)
    
    # 2. Compute Lower Confidence Bound (LCB)
    # LCB = mu - beta * sigma
    # We need to determine a good beta.
    
    # Adaptive Beta Strategy:
    # The standard UCB/LCB uses a beta that might be too high early on or too low later.
    # For minimization, we want to explore where sigma is high, but focus where mu is low.
    # A common adaptive strategy is to scale beta with the current uncertainty or progress.
    
    # Let's look at the distribution of mu and sigma.
    # If f_best is close to the minimum (0 for sphere), we should be very conservative.
    # If f_best is large, we should explore more.
    
    # Heuristic for beta:
    # Start with a moderate exploration factor and decay it as f_best improves.
    # Also, consider the relative scale of sigma.
    
    # Let's use a dynamic beta based on the ratio of current best to initial scale (or just a decay).
    # Since we don't know the step count directly in this function signature (only mu, sigma, f_best),
    # we can infer "progress" from f_best itself.
    # Sphere 1d min is 0. Initial random points on [-5, 5] might have f_best ~ 5-25.
    
    # If f_best is large, explore more. If f_best is small, exploit more.
    # Let's map f_best to a beta range, e.g., [0.5, 2.0].
    # Use a smooth decay.
    
    # Alternatively, use the mean sigma to scale beta.
    mean_sigma = float(np.mean(s))
    
    # If mean_sigma is very small, the GP is confident, so we should exploit (low beta).
    # If mean_sigma is large, we should explore (high beta).
    
    # Let's combine:
    # beta = base_beta * (mean_sigma / (mean_sigma + 1.0)) + progress_factor
    
    # Actually, a simpler and robust adaptive approach for BO is to use the 
    # "Knowledge Gradient" or "EI" which naturally balances this.
    # However, the prompt asks for a *different* strategy. The current one is a portfolio.
    # This one is a *dynamic blend*.
    
    # Let's try a specific dynamic blend:
    # Score = w * EI + (1-w) * LCB
    # where w depends on how close we are to the optimum.
    
    # Proxy for "closeness to optimum":
    # If f_best is small, we are close.
    # Let's assume a typical scale. For sphere 1d on [-5,5], max f is 25.
    # Let's define a progress metric p = 1 - min(f_best / 25, 1.0)
    # p is 0 at start (f_best=25), 1 at end (f_best=0).
    
    # But f_best can be negative? No, sphere is x^2, min 0.
    # So f_best >= 0.
    
    progress = 1.0 - min(fb / 25.0, 1.0)
    
    # Weight for EI (exploitation) vs LCB (exploration)
    # Early on (progress=0), we want more exploration (LCB).
    # Late on (progress=1), we want more exploitation (EI).
    # Let w_ei = 0.5 + 0.5 * progress
    # Let w_lcb = 1.0 - w_ei
    
    w_ei = 0.5 + 0.5 * progress
    w_lcb = 1.0 - w_ei
    
    # For LCB, we need a beta.
    # A common choice is beta = log(n) or similar, but we don't have n.
    # Let's use a fixed beta that is scaled by the uncertainty.
    # Or, simpler: use a beta that ensures the LCB is comparable to EI.
    
    # Let's use beta = 1.0 * mean_sigma? No, LCB = mu - beta*sigma.
    # If beta is 1, it's 1 std dev.
    # Let's use a beta that decays with progress too?
    # If we are close to optimum, we want a smaller beta (more precise exploitation).
    # beta = 2.0 - 1.5 * progress  (ranges from 2.0 to 0.5)
    
    beta = 2.0 - 1.5 * progress
    
    lcb_scores = mu - beta * s
    
    # Normalize scores to have similar magnitude before blending?
    # EI is in objective units. LCB is in objective units.
    # They should be comparable.
    
    # Blend
    scores = w_ei * ei_scores + w_lcb * lcb_scores
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: Add a tiny noise to break ties? No, deterministic is fine.
    
    return scores
# EVOLVE-BLOCK-END
