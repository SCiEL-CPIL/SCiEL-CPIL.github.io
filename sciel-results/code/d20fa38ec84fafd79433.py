"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = better choice.
    
    Strategy: Adaptive Expected Future Utilization.
    
    Instead of using a fixed distribution model (like Weibull with hardcoded params),
    this strategy infers the statistical properties of the incoming item stream
    from the current state of the bins.
    
    1. Estimate the mean item size (mu) and standard deviation (sigma) from the
       'holes' (remaining capacities) in the bins. The distribution of remaining
       capacities reflects the interplay between bin capacity and item sizes.
       Specifically, if items are roughly uniform or log-normal, the spread of
       remainders gives a proxy for the scale of items.
       
    2. For each valid bin, calculate the expected amount of space that will be
       filled by future items. 
       Let r be the remaining space after placing the current item.
       We model future items as coming from a distribution with estimated mean mu.
       The expected number of items that fit in space r is roughly r / mu.
       However, we also care about the "waste" at the end. A simpler proxy for
       "goodness" is how well the remainder r aligns with typical item sizes.
       
       A robust metric for online bin packing is to minimize the "regret" of
       leaving space that is too small to be useful but too large to be ignored.
       
       We use a score based on the probability that the remainder r can be
       efficiently filled. 
       Let's define the score as the expected density of the bin after this placement
       plus a bonus for remainders that are likely to be filled.
       
       A more direct approach inspired by "Next Fit Decreasing" heuristics but
       online: 
       We want to avoid remainders that are "awkward" (e.g., slightly larger than
       the median item size, causing one item to fit and a large gap to remain,
       or slightly smaller, causing no items to fit).
       
       Let's estimate the item distribution parameters from the bins array.
       The bins array contains the *current* remaining capacities.
       If we assume the items are i.i.d. from a distribution with mean mu,
       the sum of items in a bin is Cap - r.
       
       A simpler, robust heuristic that adapts:
       Estimate the "typical" item size from the gaps between sorted bin remainders?
       No, that's complex.
       
       Let's use the mean of the non-zero remainders as a proxy for the "waste" scale,
       but that's biased.
       
       Better: Assume the items are roughly uniform or exponential.
       If we look at the distribution of remainders, the "mode" or median of the
       remainders often correlates with the typical item size if the packing is
       efficient.
       
       Let's try a different structural change: **Lookahead via Distribution Matching**.
       Estimate the CDF of item sizes from the problem context (Weibull).
       Since we don't know the exact params, we can fit a simple distribution to the
       *items placed so far*? We don't have access to placed items, only current bins.
       
       However, the bins' remainders are a result of packing.
       If we assume the packing is reasonably good, the remainders are "small" relative
       to the item size.
       
       Let's use a **Dynamic Best-Fit with a Twist**:
       Score = - (remaining_space) + alpha * (probability that remaining_space is "useful")
       
       "Useful" means that the remaining space is large enough to fit at least one
       more item with high probability.
       
       We estimate the item size distribution parameters (k, lambda) for Weibull
       using the method of moments on the *assumed* item sizes. 
       Wait, we can't see the item sizes.
       
       Alternative Structural Change: **Combinatorial Gap Analysis**.
       Instead of a probabilistic model, use a deterministic gap analysis.
       For each bin, look at the remaining space r.
       Check if r is "compatible" with other bins' remainders? No, items are independent.
       
       Let's go back to **Adaptive Parameters**.
       We can estimate the mean item size `mu` by observing that in a well-packed bin,
       the remainder `r` is typically less than the mean item size.
       If we take the median of all non-zero remainders, call it `r_med`,
       it is a lower bound on the typical item size.
       Let's assume `mu ≈ 2 * r_med` (heuristic).
       
       Then, for a new remainder `r_new`, we score it based on how "fillable" it is.
       If `r_new` is close to `mu`, it's good (likely to fit one item).
       If `r_new` is much larger than `2*mu`, it's "too big" (might waste space if
       items are small).
       If `r_new` is much smaller than `mu`, it's "dead" (no item fits).
       
       Score function: A Gaussian-like bump centered at `mu` with width `sigma`.
       This encourages remainders that are "just right" to be filled by a typical item.
       
       Let's refine the estimate of `mu`.
       The bins array `bins` contains current remainders.
       Let `non_zero_bins = bins[bins > 0]`.
       If `len(non_zero_bins) > 0`:
           `r_median = np.median(non_zero_bins)`
           `mu_est = max(1.0, 1.5 * r_median)` # Heuristic scaling
           `sigma_est = max(1.0, 0.5 * r_median)`
       Else:
           `mu_est = 10.0` # Default
           `sigma_est = 5.0`
           
       Score for remainder `r`:
       `score = np.exp(-((r - mu_est) ** 2) / (2 * sigma_est ** 2))`
       
       This is a "Target Fit" strategy where the target is the estimated mean item size.
       It differs from the current PDF approach by adapting `mu` and `sigma` to the
       current state of the bins, rather than using fixed constants.
       
       Additionally, we should strongly penalize remainders that are too small to be
       useful (r < 0.5 * mu_est) and too large (r > 2 * mu_est).
       The Gaussian does this naturally.
       
       Special case: If r == 0, perfect fit. Give it the highest score.
    """
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: Simple Best Fit
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full_like(bins, -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    remaining = bins - item
    valid = remaining >= 0
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if not np.any(valid):
        return scores
    
    valid_remaining = remaining[valid]
    
    # Estimate item size distribution from current bin states
    # The remainders in the bins provide a signal about the "granularity" of the packing.
    # If items are large, remainders tend to be small (if packed well) or large (if packed poorly).
    # A robust estimate of the "typical gap" is the median of non-zero remainders.
    
    non_zero_bins = bins[bins > 0]
    if len(non_zero_bins) > 0:
        r_median = np.median(non_zero_bins)
        # Heuristic: The mean item size is likely somewhat larger than the median remainder.
        # If the packing is efficient, remainders are < item_size.
        # Let's assume mu is proportional to the median remainder.
        # If median remainder is 5, items might be around 10-15.
        mu_est = max(1.0, 2.0 * r_median)
        sigma_est = max(1.0, 0.5 * r_median)
    else:
        # No information, use defaults for capacity 100
        mu_est = 10.0
        sigma_est = 5.0
        
    # Calculate score based on Gaussian likelihood of the remainder being "fillable"
    # We want r to be close to mu_est.
    # If r is very small, it's dead space (unless 0).
    # If r is very large, it's inefficient (could have used a smaller bin or kept it open for larger items).
    
    r = valid_remaining
    
    # Initialize scores to 0
    scores_valid = np.zeros_like(r)
    
    # Perfect fit bonus
    perfect_mask = (r == 0)
    if np.any(perfect_mask):
        # Find max possible Gaussian score to set a higher bonus
        # Max of exp(-((x-mu)^2)/(2*sig^2)) is 1 when x=mu.
        # But we want perfect fit to be best.
        scores_valid[perfect_mask] = 2.0
        
    # Non-perfect fits
    non_perfect_mask = (r > 0)
    if np.any(non_perfect_mask):
        r_nz = r[non_perfect_mask]
        # Gaussian score
        z = (r_nz - mu_est) / sigma_est
        scores_valid[non_perfect_mask] = np.exp(-0.5 * z * z)
        
        # Optional: Penalize very large remainders more heavily if they are much larger than mu
        # The Gaussian already does this, but maybe we want a sharper drop-off?
        # Let's stick to the Gaussian for now as it is smooth and adaptive.
        
    scores[valid] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END
