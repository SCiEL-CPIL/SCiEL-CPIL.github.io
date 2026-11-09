"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = _start_time + 320.0  # Safety margin before 330s kill

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Set-based greedy selection with spatial redundancy penalty.
    
    Strategy:
    Instead of pointwise scoring (EI / cost^gamma), we construct scores that
    account for spatial redundancy. We use a "soft" batch construction approach:
    1. Compute base score: EI / cost.
    2. Iteratively select the top candidate that maximizes (base_score - redundancy_penalty).
    3. The redundancy penalty is based on the GP covariance between the candidate
       and already-selected points. This encourages diversity.
    4. However, since we must return 2000 scores for a single selection, we approximate
       the "marginal value" of each point assuming a small batch size K.
       
    A more direct structural change: Use a "Diversity-Adjusted Expected Improvement".
    We estimate the spatial density of candidates around each point and penalize
    points in dense regions, assuming that nearby points provide redundant information.
    
    Specifically:
    1. Compute EI for all candidates.
    2. Compute a "crowding" metric for each candidate based on the number of other
       candidates within a small radius (using scaled coordinates).
    3. Score = (EI / cost) * (1 / (1 + crowding_weight * crowding_count)).
    
    This is a different design family: it explicitly models spatial redundancy
    rather than just cost.
    """
    if _check_deadline():
        # Fallback: standard EI / cost
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei / cost_safe
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Base score: EI / cost
    base_score = ei / cost_safe
    
    # 3. Compute spatial crowding
    # We don't have coordinates, but we have mu and sigma which are correlated with location.
    # However, mu and sigma are in objective units.
    # A proxy for spatial proximity is the correlation of mu values.
    # If two points have very similar mu and sigma, they are likely close in the input space
    # (especially early on when the GP is smooth).
    
    # Let's use a simple clustering-like penalty based on mu similarity.
    # We can't do O(N^2) for 2000 points easily in pure Python, but NumPy can handle it.
    # 2000x2000 matrix is 4M elements, which is fine.
    
    # Use mu as a proxy for location.
    # Normalize mu for distance calculation.
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-10:
        mu_norm = mu
    else:
        mu_norm = (mu - np.min(mu)) / mu_range
    
    # Compute pairwise distances (using L2 norm in mu-space as a proxy)
    # To save memory and time, we can use a binned approach or just a simple kernel.
    # Let's use a Gaussian kernel with a fixed bandwidth.
    # Bandwidth should be small, e.g., 0.1 in normalized space.
    bandwidth = 0.1
    
    # For each point i, count how many points j are "close" to it.
    # dist_ij = |mu_i - mu_j|
    # crowding_i = sum_j exp(-dist_ij^2 / (2 * bandwidth^2))
    
    # This is O(N^2). N=2000 -> 4M operations. NumPy is fast.
    # Let's try it.
    mu_diff = mu_norm[:, np.newaxis] - mu_norm[np.newaxis, :]
    dist_sq = mu_diff ** 2
    kernel = np.exp(-dist_sq / (2 * bandwidth ** 2))
    
    # Exclude self
    np.fill_diagonal(kernel, 0)
    
    crowding = np.sum(kernel, axis=1)
    
    # 4. Adjust score by crowding
    # Higher crowding -> lower score
    # We want to encourage diversity.
    # Score = base_score / (1 + crowding)
    # Or Score = base_score * exp(-crowding * weight)
    
    # Let's use a multiplicative penalty.
    # If a point is in a dense cluster, its marginal information gain is lower.
    weight = 0.5
    penalty = np.exp(-weight * crowding)
    
    scores = base_score * penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
