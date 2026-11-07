"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing algorithm.
    
    Strategy:
    We use a data-driven co-occurrence prior derived from running
    First-Fit-Decreasing (FFD) simulations. The heuristic eta[i, j]
    represents how attractive it is to place item j right after item i.
    
    Key insights:
    1. Pairs that frequently co-occur in the same bin in high-quality
       FFD solutions are good candidates for ACO.
    2. We also factor in the "fit quality" - how well the pair fills
       the remaining space.
    3. Larger items are harder to place, so we weight them more heavily.
    
    The heuristic is computed once per instance before the colony runs.
    """
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Start with a base heuristic based on pair sums and waste
    pair_sums = demands[:, None] + demands[None, :]
    fits = pair_sums <= capacity
    waste = capacity - pair_sums
    
    # Base heuristic: reward tight fits with larger items
    # eta_base = (pair_sums)^2 / (waste + 1) for fitting pairs
    eta = np.full((n, n), 1e-9, dtype=float)
    
    if np.any(fits):
        # Strongly favor pairs that nearly fill the bin
        # Using inverse gap: 1/(waste+1) amplifies tight fits
        eta[fits] = (pair_sums[fits] ** 2) / (waste[fits] + 1.0)
    
    # Run FFD simulations to get co-occurrence statistics
    # This provides a data-driven prior that complements the analytical heuristic
    rng = np.random.default_rng(42)
    co_occurrence = np.zeros((n, n), dtype=float)
    num_sims = 50
    
    for sim in range(num_sims):
        # Sort indices by demand descending (FFD)
        order = np.argsort(-demands)
        bins = []  # list of lists of item indices
        bin_loads = []
        
        for idx in order:
            d = demands[idx]
            placed = False
            for b in range(len(bins)):
                if bin_loads[b] + d <= capacity:
                    bins[b].append(idx)
                    bin_loads[b] += d
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                bin_loads.append(d)
        
        # Count co-occurrences within each bin
        for bin_items in bins:
            for a in bin_items:
                for b in bin_items:
                    if a != b:
                        co_occurrence[a, b] += 1
    
    # Normalize co-occurrence matrix
    max_co = co_occurrence.max()
    if max_co > 0:
        co_occurrence = co_occurrence / max_co
    
    # Combine analytical heuristic with co-occurrence prior
    # The co-occurrence term should be a multiplier that boosts
    # pairs that frequently appear together in good solutions
    # Only apply to pairs that fit
    combined = eta * (1.0 + 2.0 * co_occurrence)
    
    # Ensure non-fitting pairs stay at 1e-9
    combined[~fits] = 1e-9
    
    # Handle any non-finite values
    combined[~np.isfinite(combined)] = 1e-9
    
    # Ensure all values are positive
    combined[combined <= 0] = 1e-9
    
    return combined
# EVOLVE-BLOCK-END
