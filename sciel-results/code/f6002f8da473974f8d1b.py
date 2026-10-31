"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 120.0  # seconds
_start_time = time.time()


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Safety: fall back to simple best fit
        return -(bins - item)
    
    remaining = bins - item
    feasible = remaining >= 0
    
    # Score feasible bins using a composite of:
    # 1. Fill ratio: item / bins (how much of available space is used)
    # 2. Negative remaining: -remaining (classic best-fit, smaller gap preferred)
    # 3. Quadratic penalty on remaining: -remaining^2 to strongly prefer tight fits
    #
    # The fill ratio alone can be misleading when all bins are large;
    # combining with -remaining ensures we prefer the tightest feasible fit.
    safe_bins = np.where(feasible & (bins > 0), bins, 1.0)
    fill_ratio = item / safe_bins
    
    # Best-fit component: prefer smaller remaining after placement
    best_fit = -remaining
    
    # Quadratic term: strongly penalize large gaps (reduces fragmentation)
    # Scale by 1/CAPACITY^2 to keep it comparable to other terms
    quad_penalty = -(remaining ** 2) / 10000.0
    
    # Composite score: weighted sum
    # Higher fill_ratio and smaller remaining (more negative best_fit is worse, 
    # so we want -remaining to be small in magnitude, i.e., remaining close to 0)
    scores = np.where(
        feasible,
        fill_ratio + best_fit / 100.0 + quad_penalty,
        -np.inf
    )
    
    return scores
# EVOLVE-BLOCK-END
