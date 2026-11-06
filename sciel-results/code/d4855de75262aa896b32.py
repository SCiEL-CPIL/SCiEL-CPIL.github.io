"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    The item goes to the highest-scoring bin.
    
    Strategy: We want to minimize the number of bins used (which minimizes the
    percent excess over the L1 lower bound). The L1 lower bound is sum(items)/capacity.
    To minimize bins used, we should try to fill bins as completely as possible
    (Best Fit Decreasing intuition). However, we also want to keep some bins
    with moderate remaining space to accommodate larger items that might come later.
    
    A good heuristic for online bin packing is "Best Fit" which picks the bin with
    the least remaining space that can still fit the item. This tends to fill bins
    up, reducing the total number of bins needed.
    
    Score = -(bins - item) for feasible bins, -inf for infeasible bins.
    This makes the bin with the smallest (bins - item) (i.e., best fit) have the
    highest score (least negative).
    
    However, pure best-fit can sometimes be suboptimal. A slight modification:
    Prefer bins that are "full enough" but not "too full" in a way that leaves
    awkward small gaps. But for simplicity and robustness, let's stick with a
    variant of best-fit that also considers the relative fill level.
    
    Let's use a score that rewards bins that are close to being full but leaves
    a small margin. Actually, the standard best-fit (minimize remaining space)
    is a strong baseline. Let's try to enhance it slightly by preferring bins
    where the remaining space after placement is not too small (to avoid
    fragmentation) but also not too large (to avoid wasting space).
    
    A common improvement is to use a penalty for very small remaining spaces
    (which might be useless for future items) and for very large remaining
    spaces (which waste capacity). But this is hard to tune without knowing
    the item distribution.
    
    Given the Weibull distribution of items, items are likely to be skewed.
    Let's try a score that is a combination of best-fit and a preference for
    bins that are already somewhat filled.
    
    Actually, let's reconsider. The score is "mean percent excess of bins used
    over the L1 lower bound". This is (bins_used / (sum_items / capacity) - 1) * 100.
    Minimizing bins_used is the primary goal.
    
    Best-fit is known to perform well. Let's stick with it but ensure we handle
    the case where no bin can fit the item (though the problem implies there's
    always a bin with enough capacity, or we should handle it gracefully).
    
    If no bin can fit the item, we should return -inf for all bins, but the
    problem statement says "bins with enough remaining capacity", implying we
    only score those. But the function signature returns a score for ALL bins.
    So we should return -inf for bins that can't fit the item.
    
    Let's refine: score = -(bins - item) for bins >= item, -inf otherwise.
    """
    remaining = bins - item
    feasible = remaining >= 0
    
    # Best-fit: minimize remaining space
    # Score is -remaining, so higher score = less remaining
    scores = np.where(feasible, -remaining, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
