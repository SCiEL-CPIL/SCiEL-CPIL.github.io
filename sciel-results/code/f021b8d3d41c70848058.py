"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(demands, capacity):
    """
    Computes a heuristic matrix based on 'complementarity' and 'gap utility'.
    
    Strategy:
    1. Calculate pair sums.
    2. Identify valid pairs (sum <= capacity).
    3. For valid pairs, calculate the remaining space (waste).
    4. Determine the smallest item size in the entire set (min_d).
    5. A gap is 'useful' if it can still fit at least one more item (gap >= min_d).
    6. Attractiveness is high when:
       - The pair fits.
       - The remaining gap is small (tight packing) OR
       - The remaining gap is large enough to fit another item (flexibility).
       
    Specifically, we use a score that favors tight packing (low waste) but 
    heavily penalizes 'dead' gaps (gaps that are > 0 but < min_d).
    
    Formula for valid pairs:
    - If gap == 0: Max score (perfect fit).
    - If gap >= min_d: Score based on inverse of gap (tighter is better, but 
      not as critical as avoiding dead gaps).
    - If 0 < gap < min_d: Very low score (dead gap, bad for future packing).
    
    This encourages ants to form pairs that either fill the bin completely or 
    leave a standard-sized hole, avoiding the fragmentation caused by tiny 
    unusable slivers of space.
    """
    n = len(demands)
    demands = np.asarray(demands, dtype=float)
    
    # Pair sums
    pair_sums = demands[:, None] + demands[None, :]
    
    # Validity mask
    valid = pair_sums <= capacity
    
    # Remaining space (waste)
    remaining = capacity - pair_sums
    
    # Smallest item in the set
    min_item = np.min(demands)
    
    # Initialize heuristic with a very small value for invalid pairs
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # Calculate scores for valid pairs
    # We want to maximize attractiveness.
    
    # 1. Perfect fits (remaining == 0) get the highest base score.
    perfect_fit = valid & (remaining < 1e-9)
    
    # 2. Usable gaps (remaining >= min_item). 
    #    We want these to be tight, so score = 1 / (remaining + 1).
    #    But we should distinguish them from perfect fits.
    usable_gap = valid & (remaining >= min_item)
    
    # 3. Dead gaps (0 < remaining < min_item).
    #    These are the worst outcome for future packing.
    dead_gap = valid & (remaining > 1e-9) & (remaining < min_item)
    
    # Base score for usable gaps: Inverse remaining space.
    # To make perfect fits stand out, we can scale them or just let the 1/(0+1)=1 
    # vs 1/(large+1) handle it. But 1/1 is not much bigger than 1/20.
    # Let's use a stronger reward for tightness.
    # Score = 1 / (remaining + 1) ** 2 ? Or just 1/(remaining+1)?
    # The previous successful strategy used 1/(remaining+1) and got 1.032.
    # Let's try to explicitly suppress dead gaps to near-zero to force the ACO 
    # to avoid them, and boost perfect fits.
    
    # Score for usable gaps
    score_usable = 1.0 / (remaining[usable_gap] + 1.0)
    
    # Score for perfect fits
    score_perfect = 10.0  # A strong boost for exact fills
    
    # Score for dead gaps
    # Make them very unattractive, but not zero (to allow exploration if needed, 
    # though 1e-9 is effectively zero in probability proportional to pheromone*heuristic 
    # if pheromone starts at 1 and decays). 
    # Actually, if heuristic is 1e-9, and pheromone is 1, prob is ~0. 
    # Let's set dead gaps to 1e-9 as well, effectively banning them.
    score_dead = 1e-9
    
    # Apply scores
    eta[usable_gap] = score_usable
    eta[perfect_fit] = score_perfect
    eta[dead_gap] = score_dead
    
    # Ensure diagonal is not accidentally high if self-pairing is allowed?
    # The task says "placing item j right after item i". Usually i != j in bin packing 
    # unless multiple copies exist. The demands are passed as a list of items. 
    # If items are distinct indices, diagonal should be invalid or low.
    # The prompt says "item-by-item matrix... placing item j right after item i".
    # It doesn't explicitly forbid i=j, but physically you can't place the same physical 
    # item twice. However, in ACO bin packing, the matrix is usually over item types 
    # or indices. If indices are unique, i=j is impossible. 
    # Let's zero out the diagonal to be safe, as placing item i after item i makes no sense.
    np.fill_diagonal(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
