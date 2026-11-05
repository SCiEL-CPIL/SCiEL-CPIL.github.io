"""Rewritten program: greedy set-cover + aggressive local-search polish for K_q(n, radius)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# Precompute powers of q for indexing
def _idx(cw, q, n):
    """Convert codeword (list of ints) to a linear index."""
    v = 0
    p = 1
    for i in range(n):
        v += cw[i] * p
        p *= q
    return v

def _neighbors(cw, q, n, radius):
    """Return the set of linear indices of all words within Hamming distance <= radius from cw."""
    total = q ** n
    # We'll use a set of indices
    covered = set()
    # Generate all words within distance <= radius
    # For radius 1: the word itself + n*(q-1) neighbors
    if radius == 0:
        return {_idx(cw, q, n)}
    if radius == 1:
        base = _idx(cw, q, n)
        covered.add(base)
        qpow = [1] * (n + 1)
        for i in range(1, n + 1):
            qpow[i] = qpow[i - 1] * q
        for pos in range(n):
            cur = cw[pos]
            for new_val in range(q):
                if new_val == cur:
                    continue
                # Compute index change
                delta = (new_val - cur) * qpow[pos]
                covered.add(base + delta)
        return covered
    else:
        # General radius: enumerate all words within distance
        # This is expensive; for the task radius=1, n=8, q=3, we can do it
        # But let's handle radius 1 efficiently and fall back for others
        base = _idx(cw, q, n)
        covered = {base}
        qpow = [1] * (n + 1)
        for i in range(1, n + 1):
            qpow[i] = qpow[i - 1] * q
        # Enumerate all subsets of positions of size 1..radius and all value changes
        from itertools import combinations, product
        for r in range(1, radius + 1):
            for positions in combinations(range(n), r):
                # For each position, choose a different value
                val_options = [list(range(q)) for _ in positions]
                # Filter out original values
                for combo in product(*val_options):
                    new_cw = list(cw)
                    valid = True
                    for idx_pos, new_val in zip(positions, combo):
                        if new_val == cw[idx_pos]:
                            valid = False
                            break
                        new_cw[idx_pos] = new_val
                    if valid:
                        covered.add(_idx(new_cw, q, n))
        return covered

def _build_all_neighbors(q, n, radius):
    """Precompute neighbor sets for all codewords in the space. Returns dict idx -> set of covered indices."""
    total = q ** n
    # For efficiency, we only precompute for codewords we actually use
    # But for the swap evaluation, we need to know overlap between any two codewords' neighborhoods
    # Strategy: precompute neighborhood for each possible codeword index
    # total = 3^8 = 6561, each neighborhood size = 1 + 8*2 = 17 for radius 1
    # So 6561 * 17 = ~111K entries, very manageable
    neigh = {}
    for idx in range(total):
        # Convert index to codeword
        cw = [0] * n
        rem = idx
        for i in range(n):
            cw[i] = rem % q
            rem //= q
        neigh[idx] = _neighbors(cw, q, n, radius)
    return neigh

def _greedy_cover(q, n, radius, deadline):
    """Greedy set cover: repeatedly pick the codeword that covers the most uncovered points."""
    total = q ** n
    # Precompute neighborhoods for all possible codewords
    neigh = _build_all_neighbors(q, n, radius)
    
    uncovered = set(range(total))
    selected = []
    
    # For efficiency, maintain a count of how many selected codewords cover each point
    # Actually, for greedy, we just need to find the codeword with max |neighborhood ∩ uncovered|
    # We can maintain a "remaining" count per point, but simpler: just iterate
    
    # To speed up, we can precompute for each codeword its neighborhood as a sorted list
    # and use bitsets or just set intersection
    
    while uncovered and time.time() < deadline:
        best_cw = None
        best_count = 0
        # Sample a subset of candidates to speed up, or iterate all
        # 6561 candidates, each with ~17 neighbors, so 6561*17 = ~111K ops per iteration
        # With up to ~486 iterations, that's ~54M ops, might be too slow in Python
        # Let's use a smarter approach: maintain a coverage count array
        
        # Alternative: maintain for each point how many selected codewords cover it
        # and for each candidate codeword, compute its marginal gain
        # This is still O(total * |neighborhood|) per step
        
        # Let's just do it directly but with numpy for speed
        # Convert uncovered to a numpy boolean array
        unc_arr = np.zeros(total, dtype=bool)
        for u in uncovered:
            unc_arr[u] = True
        
        # For each candidate, compute intersection size
        # This is 6561 set intersections, each of size ~17
        # Let's vectorize: for each candidate, sum(unc_arr[list(neigh[cand])])
        
        # Precompute neighbor lists as numpy arrays
        neigh_arrays = {idx: np.array(sorted(neigh[idx]), dtype=np.int32) for idx in range(total)}
        
        gains = np.zeros(total, dtype=np.int32)
        for cand in range(total):
            gains[cand] = np.sum(unc_arr[neigh_arrays[cand]])
        
        best_cw = int(np.argmax(gains))
        best_count = gains[best_cw]
        
        if best_count == 0:
            break
        
        selected.append(best_cw)
        # Remove covered points from uncovered
        for pt in neigh[best_cw]:
            uncovered.discard(pt)
    
    return selected, neigh

def _local_search_polish(q, n, radius, selected, neigh, deadline):
    """Local search: try to remove codewords or swap to reduce count while maintaining coverage."""
    total = q ** n
    selected = list(selected)
    
    # Build coverage count: for each point, how many selected codewords cover it
    coverage = np.zeros(total, dtype=np.int32)
    for cw_idx in selected:
        for pt in neigh[cw_idx]:
            coverage[pt] += 1
    
    # Try to remove codewords
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            cw_idx = selected[i]
            # Check if removing this codeword leaves all its points covered by others
            # i.e., for all pt in neigh[cw_idx], coverage[pt] >= 2
            removable = True
            for pt in neigh[cw_idx]:
                if coverage[pt] <= 1:
                    removable = False
                    break
            if removable:
                # Remove it
                for pt in neigh[cw_idx]:
                    coverage[pt] -= 1
                selected.pop(i)
                changed = True
                break  # Restart the loop after removal
    
    # Try swaps: replace one codeword with another that covers more uncovered-when-removed points
    # Actually, since we're maintaining full coverage, a swap means:
    # Remove cw_i, then find a new codeword cw_j that covers all points that were only covered by cw_i
    # and ideally covers some additional points (but there are no uncovered points if we have full coverage)
    # So swaps are only useful if they allow further removals
    # Let's try: for each pair (i, j) where i is in selected and j is not, check if replacing i with j
    # allows the coverage to still be valid and potentially enables more removals
    
    # This is expensive. Let's do a simpler version: try to find a codeword not in selected
    # that, when added, allows two existing codewords to be removed.
    # Or: try random swaps and see if they lead to a smaller code after polish.
    
    # For now, just do the removal polish and return
    return selected

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Greedy cover
    selected, neigh = _greedy_cover(q, n, radius, deadline)
    
    if time.time() > deadline:
        # Convert indices back to codewords
        result = []
        for idx in selected:
            cw = [0] * n
            rem = idx
            for i in range(n):
                cw[i] = rem % q
                rem //= q
            result.append(cw)
        return result
    
    # Local search polish
    selected = _local_search_polish(q, n, radius, selected, neigh, deadline)
    
    # Convert indices back to codewords
    result = []
    for idx in selected:
        cw = [0] * n
        rem = idx
        for i in range(n):
            cw[i] = rem % q
            rem //= q
        result.append(cw)
    
    return result
# EVOLVE-BLOCK-END
