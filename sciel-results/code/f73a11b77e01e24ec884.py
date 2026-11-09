"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_min_v = int(P['minimum_vertices'])

# Precompute adjacency bitmasks
_adj_masks = [0] * _N
for _a, _b in P['edges']:
    _adj_masks[_a] |= (1 << _b)
    _adj_masks[_b] |= (1 << _a)

_start_time = time.time()
_deadline = _start_time + 80.0  # Safety margin

def construct():
    global _deadline
    
    best_score = 0
    best_mask = 0
    # Initialize with a valid subset (e.g., first 8 vertices) to ensure we have a fallback
    # Though the loop will find something better or equal quickly.
    init_mask = (1 << _min_v) - 1
    best_mask = init_mask
    
    # Iterate over all possible masks
    # 2^17 is 131072, which is very small.
    # We can iterate from 0 to 2^17 - 1.
    total_masks = 1 << _N
    
    # To allow early termination on best possible score, 
    # we can iterate masks in a way that prioritizes larger popcounts?
    # Or just check all. 131k iterations is trivial.
    
    for mask in range(total_masks):
        if time.time() > _deadline:
            break
        
        k = mask.bit_count()
        if k < _min_v:
            continue
        
        # Pruning: Max distinct degrees is k. 
        # If current best is >= k, this mask cannot improve.
        if best_score >= k:
            continue
            
        # Calculate distinct degrees
        degs = set()
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            v = lsb.bit_length() - 1
            # Degree of v in induced subgraph
            d = (mask & _adj_masks[v]).bit_count()
            degs.add(d)
            m ^= lsb
            
        score = len(degs)
        if score > best_score:
            best_score = score
            best_mask = mask
            # If we achieved the theoretical maximum for this size, 
            # we could potentially prune further, but since we iterate all masks,
            # the global best will just update.
            # Note: If best_score reaches _N (17), we can break immediately as it's the absolute max.
            if best_score == _N:
                break

    # Convert best_mask to list of vertices
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(int(v))
        m ^= lsb
    
    # Ensure at least _min_v vertices (should be guaranteed by logic, but safety check)
    if len(result) < _min_v:
        # Fallback to first _min_v vertices
        result = [int(i) for i in range(_min_v)]
        
    return result
# EVOLVE-BLOCK-END
