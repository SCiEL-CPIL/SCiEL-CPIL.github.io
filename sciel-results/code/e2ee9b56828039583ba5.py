import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_MIN_V = P['minimum_vertices']

# Precompute adjacency masks
_adj = [0] * _N
for u, v in P['edges']:
    _adj[u] |= 1 << v
    _adj[v] |= 1 << u

def _get_vertices(mask):
    """Yield vertices set in mask."""
    while mask:
        lsb = mask & -mask
        yield lsb.bit_length() - 1
        mask ^= lsb

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    best_score = -1
    best_mask = 0
    
    n = _N
    min_v = _MIN_V
    
    # Iterate through all possible subsets
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            break
        
        # Check size constraint
        if mask.bit_count() < min_v:
            continue
            
        # Calculate degrees
        # We can early exit if the maximum possible distinct degrees is reached
        # Max distinct degrees is mask.bit_count()
        # But we just need to count them.
        
        degs = set()
        m = mask
        # Iterate through vertices in the subset
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (_adj[v] & mask).bit_count()
            degs.add(deg)
            m ^= lsb
            
        score = len(degs)
        
        if score > best_score:
            best_score = score
            best_mask = mask
            
            # Theoretical max for a subset of size k is k.
            # If we reach the size of the subset, we can't do better for this subset size,
            # but a larger subset might have more distinct degrees?
            # No, max distinct degrees is bounded by number of vertices.
            # However, a subset of size 8 can have at most 8 distinct degrees.
            # A subset of size 9 can have at most 9.
            # So if best_score is high, we can prune?
            # Actually, if best_score is 8, and we are checking a mask of size 8, we can't beat it.
            # But we don't know the size of the *next* mask.
            # We can skip masks of size k if best_score >= k.
            # Let's add this optimization.
            
            # Optimization: If we found a score of S, any subset of size < S cannot beat it.
            # So if mask.bit_count() < best_score, we could have skipped it?
            # We already checked bit_count < min_v.
            # We can add: if mask.bit_count() < best_score: continue
            # This is safe because max distinct degrees <= number of vertices.
            
    # Convert best_mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v)
        m ^= lsb
        
    if not result:
        # Fallback (should not happen)
        result = list(range(min_v))
        
    return result
