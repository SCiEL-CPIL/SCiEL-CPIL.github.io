"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj_masks():
    n = P['n']
    masks = [0] * n
    for u, v in P['edges']:
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin within 90s limit
    
    masks = _build_adj_masks()
    n = P['n']
    
    best_score = -1
    best_subset_mask = (1 << 8) - 1  # Initial guess: vertices 0..7
    
    # Precompute vertex lists for each size to speed up iteration
    # Search larger sizes first since they can have more distinct degrees
    # For size k, max possible distinct degrees is k
    # So size 17 can have up to 17 distinct degrees
    
    # Try sizes from largest to smallest, with early termination
    # if we find a subset with the theoretical maximum distinct degrees
    for size in range(n, 7, -1):
        theoretical_max = size  # degrees 0 through size-1
        for subset in itertools.combinations(range(n), size):
            # Build mask for this subset
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            degrees = set()
            # Iterate only over vertices in the subset
            m = mask
            while m:
                # Get lowest set bit
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                # Calculate degree of v in the induced subgraph
                deg = (masks[v] & mask).bit_count()
                degrees.add(deg)
                m ^= lsb
                
            score = len(degrees)
            if score > best_score:
                best_score = score
                best_subset_mask = mask
            
            # Early termination: if we hit theoretical max, can't do better
            if score == theoretical_max:
                # Found optimal for this size, but maybe larger size had better
                # Since we're going from large to small, this is global optimum
                # Convert and return immediately
                result = []
                mm = mask
                while mm:
                    lsb2 = mm & (-mm)
                    v2 = lsb2.bit_length() - 1
                    result.append(int(v2))
                    mm ^= lsb2
                return result
            
            # Time check
            if time.time() > deadline:
                break
        
        # Time check after each size
        if time.time() > deadline:
            break
    
    # Convert best_subset_mask back to list of vertices
    result = []
    m = best_subset_mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        result.append(int(v))
        m ^= lsb
        
    return result

# EVOLVE-BLOCK-END
