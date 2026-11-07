"""Heuristic via COMPLEMENTARITY-SCORE: use a two-pointer sweep on sorted demands
to identify ideal partner pairs, then assign high attractiveness to pairs that
are near-optimal complements in sorted order."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _two_pointer_complements(demands, capacity):
    """
    For each item (in original index space), find its ideal complement:
    the item that, when paired with it, minimizes waste while staying <= capacity.
    Uses two-pointer on sorted demands.
    
    Returns:
        ideal_partner: array of shape (n,) where ideal_partner[i] is the index
                       of the best single complement for item i (or -1 if none fits
                       as a pair).
        ideal_waste: array of shape (n,) with the waste when paired with ideal partner.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    # Sort indices by demand
    sorted_idx = np.argsort(d)
    sorted_d = d[sorted_idx]
    
    # For each item in sorted order, find the largest item that fits with it
    # Two-pointer: for item at position p (in sorted order), find the largest
    # item at position q > p such that sorted_d[p] + sorted_d[q] <= capacity
    # Also check q < p for the largest fitting item below it.
    
    ideal_partner = np.full(n, -1, dtype=int)
    ideal_waste = np.full(n, capacity, dtype=float)  # max waste = capacity (no partner)
    
    # For each position in sorted order, find best complement
    # We want the item j such that d[i] + d[j] is maximized but <= capacity
    # This means we want d[j] to be as close to (capacity - d[i]) as possible from below.
    
    for pos in range(n):
        orig_i = sorted_idx[pos]
        target = capacity - sorted_d[pos]
        
        if target < 0:
            continue
        
        # Find the largest item (in sorted order) with demand <= target
        # Use binary search on sorted_d
        # We want items at positions != pos
        # Find rightmost position where sorted_d <= target
        right = np.searchsorted(sorted_d, target, side='right') - 1
        
        best_val = -1
        best_pos = -1
        
        # Check the rightmost fitting item
        if right >= 0 and right != pos:
            best_val = sorted_d[right]
            best_pos = right
        # If right == pos, check right-1
        if right == pos and right - 1 >= 0:
            if best_val < 0:
                best_val = sorted_d[right - 1]
                best_pos = right - 1
            elif sorted_d[right - 1] > best_val:
                best_val = sorted_d[right - 1]
                best_pos = right - 1
        # If right < pos, check right (already handled) and also check if there's
        # something between pos+1 and right that's better... actually right is the
        # largest fitting, so if right < pos, we need to check right.
        
        # Also need to handle: what if the best is at a position < pos but not
        # the immediate one? The two-pointer approach: for item at pos, the best
        # complement is the largest item <= target. That's at position 'right' if
        # right != pos, or right-1 if right == pos.
        
        # But we also should consider items at positions > right that don't fit
        # with this item but might be relevant for the "structure". Actually no,
        # we only want fitting pairs.
        
        if best_pos >= 0:
            ideal_partner[orig_i] = sorted_idx[best_pos]
            ideal_waste[orig_i] = capacity - (sorted_d[pos] + sorted_d[best_pos])
    
    return ideal_partner, ideal_waste


def _build_complementarity_matrix(demands, capacity, sigma=25.0, base=1.0):
    """
    Build attractiveness matrix based on complementarity.
    
    For each pair (i, j):
    - If they don't fit together: 1e-9
    - If they do fit: score = base * exp(-waste^2 / (2*sigma^2))
      where waste = capacity - (d[i] + d[j])
    
    Additionally, boost pairs that are close to being each other's ideal complement.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    pair_sum = d[:, None] + d[None, :]
    ok = pair_sum <= capacity
    waste = capacity - pair_sum
    
    # Base Gaussian on waste
    M = np.full((n, n), 1e-9)
    M[ok] = base * np.exp(-waste[ok] ** 2 / (2.0 * sigma ** 2))
    
    # Add complementarity boost: if j is near i's ideal partner, boost M[i,j]
    ideal_partner, ideal_waste = _two_pointer_complements(demands, capacity)
    
    # For each i, boost M[i, ideal_partner[i]] significantly
    valid = ideal_partner >= 0
    for i in np.where(valid)[0]:
        j = ideal_partner[i]
        if ok[i, j]:
            # Boost by a factor based on how tight the ideal pair is
            boost = 1.0 + 2.0 * (1.0 - ideal_waste[i] / capacity)
            M[i, j] *= boost
            # Also boost the reverse direction
            M[j, i] *= boost
    
    # Ensure non-negative and finite
    M[~np.isfinite(M)] = 1e-9
    M[M <= 0] = 1e-9
    
    return M


def _greedy_bins(demands, capacity, M):
    """
    Fast deterministic greedy: start each bin with largest remaining item,
    then greedily pick the item that maximizes M[current, j] among fitting items.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    used = np.zeros(n, dtype=bool)
    bins = 0
    
    while True:
        remaining = np.where(~used)[0]
        if len(remaining) == 0:
            break
        
        # Start new bin with largest remaining item
        start = remaining[np.argmax(d[remaining])]
        used[start] = True
        bin_load = d[start]
        bins += 1
        current = start
        
        while True:
            rem = np.where(~used)[0]
            if len(rem) == 0:
                break
            fitting = rem[d[rem] <= capacity - bin_load]
            if len(fitting) == 0:
                break
            
            scores = M[current, fitting]
            best_local = np.argmax(scores)
            best_j = fitting[best_local]
            
            used[best_j] = True
            bin_load += d[best_j]
            current = best_j
    
    return bins


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    capacity = float(capacity)
    n = len(demands)
    
    t0 = time.time()
    deadline = t0 + 280.0
    
    # Try several sigma values and pick the one that gives best greedy proxy
    sigmas = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
    
    best_M = None
    best_bins = float('inf')
    
    for sigma in sigmas:
        if time.time() > deadline:
            break
        M = _build_complementarity_matrix(demands, capacity, sigma=sigma)
        bins = _greedy_bins(demands, capacity, M)
        if bins < best_bins:
            best_bins = bins
            best_M = M.copy()
    
    # If we ran out of time before evaluating any, use a default
    if best_M is None:
        best_M = _build_complementarity_matrix(demands, capacity, sigma=25.0)
    
    # Final safety: ensure all values are positive and finite
    best_M[~np.isfinite(best_M)] = 1e-9
    best_M[best_M <= 0] = 1e-9
    
    return best_M
# EVOLVE-BLOCK-END
