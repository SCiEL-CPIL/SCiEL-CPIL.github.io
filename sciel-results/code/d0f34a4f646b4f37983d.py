"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_TIME_LIMIT = 300.0  # seconds, with safety margin

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using Thompson Sampling with local GP approximation.
    
    Instead of a single heuristic score, we approximate the posterior over the 2000
    candidates using a local GP fit on the top uncertainty/mean candidates, then
    sample a function value for each candidate from this local posterior.
    To keep it efficient and robust, we use a simpler approach:
    We sample a single scalar 'noise' level based on the global uncertainty,
    but more effectively, we treat the problem as selecting the argmin of
    mu(x) - epsilon * sigma(x) where epsilon is sampled from a distribution
    that adapts to the progress. However, true Thompson Sampling requires
    sampling from the joint posterior.
    
    Given the constraint of returning scores for 2000 points and the high dimension
    of the latent space, a full joint sample is expensive. 
    Alternative Structural Change: **Probability of Improvement (PoI) with adaptive exploration**
    or **Knowledge Gradient (KG) approximation**.
    
    Let's use a **Thompson Sampling** approximation that is computationally feasible:
    We sample a single parameter 'beta' from a Gamma distribution or similar,
    but that's just UCB with random beta.
    
    A better structural change: **Expected Improvement (EI) with a corrected baseline**.
    The current code uses UCB. EI is often superior for minimization when the GP is well-calibrated.
    However, the inspiration suggests EI scored well (2.15). The current code (0.53) is UCB.
    Let's switch to **Expected Improvement (EI)** but with a dynamic exploitation/exploration
    balance handled by the GP's inherent properties, or a **Local Expected Improvement (LEI)**
    if we had coordinates. We don't have coordinates.
    
    Wait, we DO have mu and sigma for 2000 points. We don't have x.
    This limits us to acquisition functions that depend only on mu and sigma.
    Common ones: UCB, EI, PI, TS (approx).
    
    The inspiration with 2.15 used EI. The current 0.53 uses UCB.
    Why did UCB fail? Likely because the Weierstrass function is very noisy/fractal.
    UCB with a fixed or poorly tuned beta can get stuck in high-variance regions
    that are not actually low-mean regions.
    
    **New Strategy: Expected Improvement (EI)**
    EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    This naturally balances exploration (sigma term) and exploitation (mu term)
    without an arbitrary beta parameter. It is theoretically grounded.
    
    To make it a *different* strategy from the current UCB, we implement EI.
    To make it robust, we handle sigma=0 cases.
    """
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - 5.0:
        # Fallback: simple exploitation
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        sigma = np.where(np.isfinite(sigma), sigma, 0.0)
        mu = np.where(np.isfinite(mu), mu, f_best)
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle numerical issues
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-10)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # Where Phi is the standard normal CDF and phi is the standard normal PDF
    
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical overflow in exp
    z = np.clip(z, -10, 10)
    
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = exp(-z^2 / 2) / sqrt(2 * pi)
    
    # Using scipy.stats.norm for accuracy
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    improvement = f_best - mu
    ei = improvement * Phi_z + sigma * phi_z
    
    # EI is positive when improvement is likely.
    # We want to maximize EI.
    # If sigma is very small and mu < f_best, EI is large.
    # If sigma is large, EI is driven by the sigma * phi term.
    
    # To prevent EI from being dominated by points with extremely high sigma
    # (which might be artifacts or just unexplored regions), we can optionally
    # blend with a small UCB term or just use EI as is.
    # Standard EI is usually sufficient.
    
    # Ensure finite output
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    # If all scores are 0 or negative (shouldn't happen if f_best is min),
    # but just in case, we can shift them or handle.
    # EI is always >= 0 for minimization if we define it correctly?
    # If mu < f_best, improvement > 0. If mu > f_best, improvement < 0.
    # The formula handles this.
    
    return scores
# EVOLVE-BLOCK-END
