"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

def _weibull_cdf(x, k=1.5, lam=1.0):
    """Compute Weibull CDF for array x."""
    x = np.asarray(x)
    cdf = np.zeros_like(x)
    mask = x > 0
    if np.any(mask):
        cdf[mask] = 1.0 - np.exp(-((x[mask] / lam) ** k))
    return cdf

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Probabilistic Gap Penalty with Polarization Bias.
    1. Base score: Minimize remaining space (Best Fit) using Weibull CDF.
       Score = -CDF(remaining). This favors small remainders, but penalizes
       remainders that are likely to be wasted (smaller than expected next item).
    2. Polarization Bias: Add a term that encourages remainders to be either
       very small (close to 0) or very large (close to capacity), avoiding
       mid-range fragments that are hard to fill.
       The current program used a blend of best-fit and gentler scores.
       Inspiration 2 used polarization. We combine them:
       Use CDF to model the "waste" probability, and add a polarization term
       to avoid mid-sized gaps.
    """
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores

    remaining_after = bins - item
    feasible = remaining_after >= 0

    if not np.any(feasible):
        return np.full(len(bins), -np.inf)

    scores = np.full(len(bins), -np.inf)
    
    rem = remaining_after[feasible]
    
    # 1. Probabilistic Waste Penalty (from Inspiration 1)
    # We want to avoid creating gaps that are too small to be filled by future items.
    # If the gap is r, the probability that a random Weibull item fits is CDF(r).
    # If CDF(r) is low, it's unlikely to be filled -> high waste risk.
    # So we want to MINIMIZE the probability that it IS filled? No.
    # If CDF(r) is high, it's likely to be filled -> good.
    # If CDF(r) is low, it's unlikely to be filled -> bad (waste).
    # So we want to MAXIMIZE CDF(r)?
    # Wait, if r is very small, CDF(r) is small (unlikely to fit any item).
    # If r is very large, CDF(r) is close to 1 (likely to fit many items).
    # But large r means we used a lot of capacity for this item? No, r is remaining.
    # Large r means the bin was very empty.
    # Best Fit chooses small r.
    # If r is small, CDF(r) is small.
    # If we score by CDF(r), we prefer large r (Worst Fit).
    # If we score by -CDF(r), we prefer small r (Best Fit).
    
    # Let's re-evaluate Inspiration 1's logic.
    # "Score = -CDF(remaining)".
    # If remaining is small, CDF is small, score is close to 0 (high).
    # If remaining is large, CDF is large, score is very negative (low).
    # So -CDF(remaining) favors SMALL remainders. This is Best Fit.
    
    # The "Probabilistic Gap Penalty" idea from the knowledge base:
    # "avoids creating fragments smaller than the expected next Weibull item".
    # If the fragment r is smaller than the typical item, it's wasted.
    # The expected item size is lambda * Gamma(1 + 1/k).
    # For k=1.5, lambda=1, mean = Gamma(1.666) ~ 0.9.
    # So if r < 0.9, it's likely wasted.
    # We should penalize r in the range (0, 1) heavily.
    # But Best Fit already minimizes r.
    # Maybe we should prefer r=0 (perfect fit) or r > 1 (can fit another item).
    # This is the Polarization idea!
    
    # Let's combine:
    # Score = -CDF(r) + Polarization_Bonus(r)
    
    cdf_vals = _weibull_cdf(rem)
    
    # 2. Polarization Bias (from Inspiration 2)
    # Prefer r close to 0 or r close to Capacity (100).
    # Center at 50.
    # (r - 50)^2 is high when r is near 0 or 100.
    # But we need to scale it appropriately.
    # The CDF term ranges from ~0 to 1.
    # The polarization term (r-50)^2 ranges from 0 to 2500.
    # We need to scale it down.
    # Let's use a normalized polarization.
    # polar = ((rem - 50.0) / 50.0) ** 2
    # This ranges from 0 to 1.
    # So Score = -cdf_vals + alpha * polar
    
    # What is a good alpha?
    # If alpha is large, we strongly prefer r=0 or r=100.
    # If alpha is small, we mostly follow Best Fit (-CDF).
    # Best Fit prefers r=0.
    # Polarization prefers r=0 or r=100.
    # So they agree on r=0.
    # They disagree on r=50. Best Fit prefers r=50 over r=100 (since 50 < 100).
    # Polarization prefers r=100 over r=50.
    # So this will push us away from mid-sized gaps, towards either tight fits or empty bins.
    # This should reduce fragmentation.
    
    polar = ((rem - 50.0) / 50.0) ** 2
    
    # Combine
    # We want to maximize score.
    # -cdf_vals is high (close to 0) for small r.
    # polar is high (close to 1) for r near 0 or 100.
    # So for r near 0, both are high.
    # For r near 100, -cdf is low (close to -1), polar is high (close to 1).
    # For r near 50, -cdf is medium, polar is low (0).
    
    # Let's try alpha = 0.5
    scores[feasible] = -cdf_vals + 0.5 * polar
    
    return scores
# EVOLVE-BLOCK-END
