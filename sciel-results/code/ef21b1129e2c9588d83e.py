"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

D = 11
DEADLINE_MARGIN = 10  # seconds to spare before the 150s limit


def construct():
    """Construct integer vectors in R^11 with min pairwise distance >= max norm."""
    start_time = time.time()
    
    # Generate candidate vectors: all nonzero integer vectors with norm <= 10
    # We'll use a greedy approach to select a valid subset
    candidates = []
    
    # Generate vectors by coordinate patterns
    # Start with simple patterns and build up
    for norm_sq_max in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20]:
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        
        # Generate all vectors with squared norm <= norm_sq_max
        # Use recursive generation for small norms
        def gen_vectors(dim, max_norm_sq, current, remaining):
            if dim == 0:
                if current and sum(x*x for x in current) > 0:
                    candidates.append(tuple(current))
                return
            if time.time() - start_time > DEADLINE_MARGIN:
                return
            
            # Try values for this dimension
            max_val = int(np.sqrt(max_norm_sq - remaining)) + 1
            for v in range(-max_val, max_val + 1):
                new_remaining = remaining - v * v
                if new_remaining < 0:
                    continue
                gen_vectors(dim - 1, max_norm_sq, current + [v], new_remaining)
        
        gen_vectors(D, norm_sq_max, [], norm_sq_max)
    
    if not candidates:
        # Fallback: return the seed solution
        vectors = []
        for i, j in [(i, j) for i in range(D) for j in range(i+1, D)]:
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    # Convert to numpy array
    cand_array = np.array(candidates, dtype=np.int64)
    norms_sq = np.sum(cand_array ** 2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Greedy selection: pick vectors that satisfy the distance constraint
    # min pairwise distance >= max norm
    # i.e., ||vi - vj|| >= sqrt(max_norm_sq) for all i != j
    # i.e., ||vi - vj||^2 >= max_norm_sq
    
    selected = []
    selected_set = set()
    
    # Sort candidates by some heuristic (e.g., norm) to process
    # Let's try processing in order of increasing norm first
    order = np.argsort(norms_sq)
    
    for idx in order:
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        
        v = cand_array[idx]
        v_norm_sq = norms_sq[idx]
        
        # Check if this vector is compatible with all selected vectors
        compatible = True
        for sel_idx in selected:
            sel_v = cand_array[sel_idx]
            diff = v - sel_v
            dist_sq = np.sum(diff ** 2)
            if dist_sq < max_norm_sq:
                compatible = False
                break
        
        if compatible:
            selected.append(idx)
            selected_set.add(idx)
    
    if not selected:
        # Fallback: return the seed solution
        vectors = []
        for i, j in [(i, j) for i in range(D) for j in range(i+1, D)]:
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result = cand_array[selected]
    return result
# EVOLVE-BLOCK-END
