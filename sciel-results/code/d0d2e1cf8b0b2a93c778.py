import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_TIMEOUT = 325.0

def acquisition(mu, sigma, f_best):
    """
    Rank-Aggregated Acquisition Function.
    
    Instead of a fixed linear combination of acquisition signals (which requires
    careful tuning of weights and normalization), this function computes multiple
    standard acquisition signals (EI, LCB, PI, Sigma), converts each to a rank
    (1st best to Nth best), and returns the average rank as the score.
    
    This approach is:
    1. Structurally different from linear blending (non-linear aggregation).
    2. Robust to outliers in mu/sigma (ranks are invariant to monotonic transformations).
    3. Automatically balances exploration and exploitation by giving equal weight
       to the relative quality of each signal.
    """
    # Deadline guard
    if time.time() - _T0 > _TIMEOUT:
        # Fallback: Pure EI (deterministic, fast)
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    n = len(mu)
    if n == 0:
        return np.array([])

    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI)
    z_ei = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in CDF/PDF
    z_ei_clipped = np.clip(z_ei, -50, 50)
    ei = (f_best - mu) * norm.cdf(z_ei_clipped) + sigma_safe * norm.pdf(z_ei_clipped)
    
    # 2. Lower Confidence Bound (LCB)
    # We minimize f, so LCB = mu - beta * sigma.
    # We want to MAXIMIZE the score, so we negate it: score = -mu + beta * sigma.
    beta_lcb = 2.0
    lcb_score = -mu + beta_lcb * sigma_safe
    
    # 3. Probability of Improvement (PI)
    z_pi = (f_best - mu) / sigma_safe
    z_pi_clipped = np.clip(z_pi, -50, 50)
    pi = norm.cdf(z_pi_clipped)
    
    # 4. Uncertainty (Sigma)
    # Pure exploration signal
    sig_score = sigma_safe
    
    # Convert each signal to ranks
    # np.argsort(np.argsort(x)) gives 0-based rank (0 is smallest value)
    # We want highest score to be best.
    # For EI, higher is better. Rank 0 = lowest EI (worst). Rank N-1 = highest EI (best).
    # So we can just use the rank directly if we want "higher rank = better".
    # np.argsort returns indices that would sort the array.
    # np.argsort(np.argsort(x)) gives the rank of each element in the sorted array.
    # Smallest x gets rank 0. Largest x gets rank N-1.
    # This is exactly what we want for "higher score is better".
    
    rank_ei = np.argsort(np.argsort(ei))
    rank_lcb = np.argsort(np.argsort(lcb_score))
    rank_pi = np.argsort(np.argsort(pi))
    rank_sig = np.argsort(np.argsort(sig_score))
    
    # Average the ranks
    # This produces a score in [0, N-1]
    # Higher score means better across all metrics
    scores = (rank_ei + rank_lcb + rank_pi + rank_sig) / 4.0
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
