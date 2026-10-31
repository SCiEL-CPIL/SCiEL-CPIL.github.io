import numpy as np
import itertools

def construct():
    D = 11
    # Generate candidate integer vectors with norm squared <= 12
    # This is a search-based approach, distinct from the hardcoded algebraic construction.
    candidates = []
    
    # Norm squared 2: +/-1, +/-1 (D11 minimal)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Norm squared 4: +/-2, 0... (not minimal for D11, but valid integer)
    # Actually, let's stick to norm 8 and 10 to compete with the seed.
    # The seed uses norm 8.
    
    # Generate all vectors with norm squared 8
    # Type 1: 2, 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                candidates.append(v)
                
    # Type 2: 1, 1, 1, 1, 1, 1, 1, 1
    for mask in range(2**8):
        v = [0] * D
        minus_count = 0
        for k in range(8):
            if (mask >> k) & 1:
                v[k] = -1
                minus_count += 1
            else:
                v[k] = 1
        if minus_count % 2 == 0:
            candidates.append(v)
            
    # Type 3: 2, 1, 1, 1, 1
    for i in range(D):
        for j in itertools.combinations([k for k in range(D) if k != i], 4):
            for si in (1, -1):
                for sj in itertools.product((1, -1), repeat=4):
                    v = [0] * D
                    v[i] = 2 * si
                    for k, s in zip(j, sj):
                        v[k] = s
                    candidates.append(v)

    # Greedy selection
    selected = []
    max_norm_sq = 0
    
    # Sort by norm squared to prioritize smaller vectors?
    # Or just iterate.
    # To be fast, we only consider norm 8 vectors first, as they form the known 332 config.
    # If we include norm 10, we might block norm 8 or vice versa.
    # The seed score is 332. We aim to match or beat.
    
    # Filter candidates to norm 8 only to ensure we get the 332 base
    norm8_candidates = [v for v in candidates if sum(x*x for x in v) == 8]
    
    # Greedy add
    for v in norm8_candidates:
        v_sq = sum(x*x for x in v)
        if v_sq < max_norm_sq:
            continue
        # Check distance to all selected
        valid = True
        for u in selected:
            diff = [a - b for a, b in zip(v, u)]
            dist_sq = sum(x*x for x in diff)
            if dist_sq < max(max_norm_sq, v_sq):
                valid = False
                break
        if valid:
            selected.append(v)
            max_norm_sq = max(max_norm_sq, v_sq)
            
    return np.array(selected, dtype=np.int64)
