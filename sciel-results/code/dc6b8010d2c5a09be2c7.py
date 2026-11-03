import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11
    
    # Generate all 4-subsets (supports)
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    # Generate all triples
    triples = list(itertools.combinations(range(D), 3))
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # For each support, list the triples it contains
    sup_triples = []
    for s in supports:
        ts = [tuple(t) for t in itertools.combinations(s, 3)]
        sup_triples.append(ts)
        
    # Greedy packing
    # Track which triples are covered
    covered = [False] * len(triples)
    selected_supports = []
    
    # Precompute for each support, the list of triple indices
    sup_triple_indices = []
    for ts in sup_triples:
        idxs = [triple_to_idx[t] for t in ts]
        sup_triple_indices.append(idxs)
        
    # To speed up, sort supports by "potential" or just iterate
    # We want to maximize the number of supports.
    # Since all supports cover exactly 4 triples, any maximal packing will have the same size?
    # Not necessarily, but greedy usually works well.
    
    # A more efficient greedy:
    # Maintain a count of how many times each triple is covered.
    # A support is valid if all its 4 triples are currently covered 0 times.
    
    triple_count = np.zeros(len(triples), dtype=np.int8)
    
    while time.time() < deadline - 10.0:
        found = False
        # To make it deterministic and fast, we can iterate through supports
        # and pick the first one that is valid.
        # To get a better packing, we might want to pick the one that "blocks" the fewest others?
        # But with 330 supports, simple greedy is fast.
        
        # Optimization: Precompute conflicts?
        # Two supports conflict if they share a triple.
        # We can maintain a set of "blocked" supports.
        
        # Simpler: Just scan.
        for i in range(n_sup):
            if time.time() > deadline - 10.0:
                break
            valid = True
            for t_idx in sup_triple_indices[i]:
                if triple_count[t_idx] > 0:
                    valid = False
                    break
            if valid:
                selected_supports.append(i)
                for t_idx in sup_triple_indices[i]:
                    triple_count[t_idx] += 1
                found = True
                break
        
        if not found:
            break
            
    # Construct vectors
    all_vectors = []
    for s_idx in selected_supports:
        sup = supports[s_idx]
        # All 16 sign patterns
        for bits in range(16):
            signs = [1 if (bits >> b) & 1 else -1 for b in range(4)]
            v = np.zeros(D, dtype=np.int8)
            for b, idx in enumerate(sup):
                v[idx] = signs[b]
            all_vectors.append(v)
            
    vec_arr = np.array(all_vectors, dtype=np.int8)
    
    # Verify
    if len(vec_arr) > 1:
        norms_sq = np.sum(vec_arr**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        # Check pairwise distances
        n = len(vec_arr)
        ok = True
        # Batch verification
        for i in range(0, n, 50):
            if time.time() > deadline - 2.0:
                break
            batch = vec_arr[i:i+50]
            for j in range(i+1, n):
                if time.time() > deadline - 2.0:
                    ok = False
                    break
                d2 = np.sum((batch - vec_arr[j])**2, axis=1)
                if np.min(d2) < max_norm_sq:
                    ok = False
                    break
            if not ok:
                break
                
        if not ok:
            # Fallback to seed (weight 2)
            w2_vectors = []
            for i in range(D):
                for j in range(i+1, D):
                    for si in (-1, 1):
                        for sj in (-1, 1):
                            v = np.zeros(D, dtype=np.int8)
                            v[i] = si
                            v[j] = sj
                            w2_vectors.append(v)
            return np.array(w2_vectors, dtype=np.int8)
            
    return vec_arr
