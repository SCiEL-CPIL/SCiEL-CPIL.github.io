"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Mask out bins that cannot fit the item
    valid = bins >= item
    if not np.any(valid):
        return np.full_like(bins, -np.inf, dtype=float)
    
    # We want to maximize the score for valid bins.
    # The heuristic is based on "Next-Item Compatibility".
    # We estimate the expected size of the next item from the Weibull distribution.
    # Since we don't know the exact parameters, we use a robust estimate.
    # For Weibull(k, lambda) with k=2, lambda=10, mean is ~8.8.
    # For k=1, lambda=10, mean is 10.
    # Let's assume a typical mean item size for this benchmark.
    # A common setup is items in [0, 100] or similar.
    # Given capacity 100, and 1000 items, if mean is 10, total size is 10000, LB is 100 bins.
    # If mean is 50, total size is 50000, LB is 500 bins.
    # Best-Fit is strong regardless.
    
    # The directive suggests: "score each bin by the probability that the residual capacity 
    # will successfully accommodate that estimated next item".
    # Let E be the expected next item size.
    # Residual capacity r = bins[i] - item.
    # We want r to be large enough to fit a typical next item, but not so large that we waste space.
    # Actually, if r >= E, it's a "good" bin for the next item.
    # If r < E, it might be too small for the next item, leading to fragmentation.
    # So we prefer bins where r is close to E or slightly larger?
    # Or do we prefer bins where r is very small (Best-Fit) because it minimizes waste?
    
    # Let's look at the "Bimodal scoring function -r*(C-r)" from the knowledge.
    # That function prefers r near 0 or r near C.
    # r near 0 means the bin is almost full (Best-Fit).
    # r near C means the bin is almost empty (Worst-Fit).
    # This bimodality was measured to improve score from 1.0 to 1.02.
    
    # Let's try a smooth approximation of this bimodal preference.
    # We want to maximize score.
    # Let's use a score that is high when r is small OR r is large.
    # But wait, if r is large, we are not packing efficiently.
    # Why would -r*(C-r) work?
    # If r is small, -r*(C-r) is small negative (close to 0).
    # If r is large (near C), -r*(C-r) is small negative (close to 0).
    # If r is C/2, -r*(C-r) is -C^2/4 (very negative).
    # So maximizing this score means avoiding the middle.
    # This effectively implements a "Best-Fit or Worst-Fit" hybrid, avoiding bins that are half-full.
    # This can reduce fragmentation by either filling bins up or keeping them empty.
    
    # Let's implement this bimodal score for valid bins.
    # r = bins[valid] - item
    # C = 100 (Capacity)
    # score = -r * (C - r)
    
    # However, we must be careful. If r is very small, score is ~0.
    # If r is large, score is ~0.
    # If r is medium, score is very negative.
    # The argmax will pick a bin with r near 0 or r near C.
    # If there is a bin with r=1 and a bin with r=90, both have score ~0.
    # Which one is better?
    # r=1 is better for packing efficiency.
    # We can add a small bias towards smaller r to break ties in favor of Best-Fit.
    # score = -r * (C - r) - epsilon * r
    
    # Let's refine the parameters.
    # C = 100.
    # epsilon = 0.01?
    
    # Let's try this.
    
    C = 100.0
    r = bins[valid] - item
    # Bimodal score: prefer r near 0 or r near C.
    # The term -r*(C-r) is a parabola opening downwards, max at C/2.
    # We want to maximize score, so we want to be far from the max.
    # So we want to minimize r*(C-r).
    # This is equivalent to maximizing -r*(C-r).
    
    # Let's add a linear term to prefer smaller r (Best-Fit) when bimodality is tied.
    # score = -r * (C - r) - 0.1 * r
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    scores[valid] = -r * (C - r) - 0.1 * r
    
    return scores

# EVOLVE-BLOCK-END
