"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Generate all D11 roots: weight-2 vectors with entries ±1
    # Total: C(11,2) * 4 = 55 * 4 = 220 vectors, each with norm^2 = 2
    all_roots = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                all_roots.append(v)
    all_roots = np.array(all_roots, dtype=np.int8)
    n_roots = len(all_roots)
    
    # Precompute pairwise dot product matrix (n_roots x n_roots)
    # For two roots u, v: dot(u,v) can be -2, -1, 0, 1, 2
    # Distance condition: ||u - v||^2 >= ||u||^2 = 2 (since all have same norm sqrt(2))
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2*dot(u,v) = 4 - 2*dot(u,v) >= 2
    # => dot(u,v) <= 1
    # So two roots are compatible iff dot(u,v) <= 1
    # They conflict iff dot(u,v) >= 2 (i.e., they are the same vector or negatives... wait)
    # Actually dot=2 means u=v, dot=-2 means u=-v
    # Since we include both +v and -v as separate roots, dot(u,v)=2 only if u=v
    # dot(u,v)=-2 if u=-v
    # The constraint is dot <= 1, so conflict when dot >= 2, i.e., dot == 2 (identical)
    # But wait - we're selecting a subset, so we won't pick the same one twice.
    # The real conflict: dot(u,v) = 2 means u and v are identical (same vector).
    # Since all vectors in all_roots are distinct, dot=2 never occurs between different entries.
    # So actually ALL pairs of distinct roots satisfy dot <= 1? Let me check.
    # u = e_i + e_j, v = e_i - e_j: dot = 1 - 1 = 0. OK.
    # u = e_i + e_j, v = e_i + e_k: dot = 1. OK (<=1).
    # u = e_i + e_j, v = -e_i - e_j: dot = -2. OK.
    # u = e_i + e_j, v = e_k + e_l (disjoint): dot = 0. OK.
    # u = e_i + e_j, v = -e_i + e_j: dot = -1 + 1 = 0. OK.
    # u = e_i + e_j, v = e_i + e_j: dot = 2. But these are the same vector, won't pick twice.
    # So for DISTINCT roots, max dot is 1. That means ALL 220 roots are mutually compatible!
    
    # Wait, that can't be right for a kissing number problem. Let me re-read the task.
    # "minimum pairwise distance is at least the maximum vector norm"
    # All vectors have norm sqrt(2), so max norm = sqrt(2).
    # Distance between u and v: ||u-v|| >= sqrt(2) => ||u-v||^2 >= 2
    # ||u-v||^2 = 4 - 2*dot(u,v) >= 2 => dot(u,v) <= 1
    # For distinct D11 roots, the maximum dot product between two different roots is 1
    # (e.g., e_i+e_j and e_i+e_k share one coordinate).
    # So indeed all 220 are mutually compatible!
    
    # But wait - the task says "nonzero integer vectors in R^11 whose minimum pairwise distance 
    # is at least the maximum vector norm". If all 220 roots work, the answer is at least 220.
    # But we could also add vectors of different norms. The constraint is:
    # min_{i != j} ||v_i - v_j|| >= max_k ||v_k||
    # If we add a vector w with norm > sqrt(2), then ALL pairs must be at distance >= ||w||.
    # This makes it harder. So the optimal strategy might be to use only norm-sqrt(2) vectors,
    # giving us 220, OR mix in some higher-norm vectors if they allow more total vectors.
    
    # Actually, let me reconsider. The kissing number in R^11 is known to be at least 220
    # (from the D11 lattice roots) and the exact value is unknown but bounded.
    # The D11 roots give 220. Can we do better by mixing norms?
    
    # Let's just return all 220 roots as a baseline, then try to augment.
    selected = all_roots.copy()
    
    # Now try to add weight-4 vectors (norm 2) to the configuration.
    # If we add a vector w with norm 2, then max norm becomes 2.
    # All existing roots must be at distance >= 2 from w:
    # ||root - w||^2 >= 4 => 2 + 4 - 2*dot(root, w) >= 4 => dot(root, w) <= 1
    # And w must be at distance >= 2 from all other added vectors.
    
    # Generate weight-4 vectors
    w4_vectors = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(support, signs):
                v[idx] = val
            w4_vectors.append(v)
    w4_vectors = np.array(w4_vectors, dtype=np.int8)
    n_w4 = len(w4_vectors)
    
    # Precompute dot products between roots and w4 vectors
    # roots: (220, 11), w4: (n_w4, 11)
    # dot_roots_w4 = roots @ w4.T  -> (220, n_w4)
    dot_roots_w4 = all_roots.astype(np.int16) @ w4_vectors.astype(np.int16).T  # (220, n_w4)
    
    # A w4 vector w is compatible with all roots iff max over roots of dot(root, w) <= 1
    # i.e., column max of dot_roots_w4 <= 1
    col_max = np.max(dot_roots_w4, axis=0)  # (n_w4,)
    compatible_w4_mask = col_max <= 1
    compatible_w4_indices = np.where(compatible_w4_mask)[0]
    
    # Among compatible w4 vectors, we need pairwise distance >= 2:
    # ||w_a - w_b||^2 >= 4 => 4 + 4 - 2*dot(w_a, w_b) >= 4 => dot(w_a, w_b) <= 2
    # So compatible w4 vectors must have dot <= 2 with each other.
    
    if len(compatible_w4_indices) > 0:
        comp_w4 = w4_vectors[compatible_w4_indices]
        n_comp = len(comp_w4)
        
        # Precompute pairwise dot among compatible w4
        # Use vectorized approach in batches
        w4_comp_dots = comp_w4.astype(np.int16) @ comp_w4.astype(np.int16).T  # (n_comp, n_comp)
        
        # Conflict matrix: conflict if dot > 2
        conflict = (w4_comp_dots > 2)
        np.fill_diagonal(conflict, False)
        
        # Greedy selection with local search
        best_w4_subset = []
        
        for restart in range(50):
            if time.time() > deadline:
                break
                
            np.random.seed(restart)
            perm = np.random.permutation(n_comp)
            
            current = set()
            for idx in perm:
                if time.time() > deadline:
                    break
                if not conflict[idx, list(current)].any():
                    current.add(idx)
            
            # Local search: try to swap one out for two in
            improved = True
            while improved and time.time() < deadline:
                improved = False
                current_list = list(current)
                for i, out_idx in enumerate(current_list):
                    if time.time() > deadline:
                        break
                    # Try removing out_idx and adding two new ones
                    remaining = [j for j in current_list if j != out_idx]
                    # Find candidates that conflict with out_idx but not with remaining
                    # Actually, find candidates not in current that are compatible with remaining
                    in_current_mask = np.zeros(n_comp, dtype=bool)
                    in_current_mask[list(current)] = True
                    # Candidates must be compatible with all of remaining
                    if remaining:
                        rem_arr = np.array(remaining)
                        compat_with_rem = ~conflict[rem_arr].any(axis=0)
                    else:
                        compat_with_rem = np.ones(n_comp, dtype=bool)
                    compat_with_rem[in_current_mask] = False
                    candidate_indices = np.where(compat_with_rem)[0]
                    
                    if len(candidate_indices) < 2:
                        continue
                    
                    # Try pairs of candidates
                    found_swap = False
                    for a in range(len(candidate_indices)):
                        if time.time() > deadline:
                            break
                        ca = candidate_indices[a]
                        for b in range(a+1, len(candidate_indices)):
                            cb = candidate_indices[b]
                            if not conflict[ca, cb]:
                                new_current = set(remaining)
                                new_current.add(ca)
                                new_current.add(cb)
                                if len(new_current) > len(current):
                                    current = new_current
                                    improved = True
                                    found_swap = True
                                    break
                        if found_swap:
                            break
                    if found_swap:
                        break
            
            if len(current) > len(best_w4_subset):
                best_w4_subset = list(current)
        
        # Add best w4 subset to selected
        if best_w4_subset:
            best_w4_vectors = comp_w4[best_w4_subset]
            selected = np.vstack([selected, best_w4_vectors])
    
    # Try to add weight-3 vectors (norm sqrt(3))
    # If max norm becomes sqrt(3), all pairs need distance >= sqrt(3)
    # ||u-v||^2 >= 3
    # For two roots: 4 - 2*dot >= 3 => dot <= 0.5 => dot <= 0
    # This is more restrictive. Let's check if it's worth it.
    # Actually, let's try adding weight-3 vectors with norm sqrt(3).
    # Then max norm = sqrt(3). All pairs need ||u-v|| >= sqrt(3).
    # root-root: 4 - 2*dot >= 3 => dot <= 0. So roots must have dot <= 0.
    # This means we can't use all 220 roots anymore.
    # This is a tradeoff. Let's skip for now and focus on what works.
    
    # Try to add weight-2 vectors with entry ±2 (norm 2)
    # v = 2*e_i, norm = 2. Then max norm = 2.
    # root to 2e_i: ||(e_j+e_k) - 2e_i||^2. If i not in {j,k}: 1+1+4=6 >= 4. OK.
    # If i=j: ||(-e_i + e_k)||^2 = 1+1=2 < 4. FAIL.
    # So 2e_i conflicts with roots that have e_i in their support.
    # 2e_i and 2e_j: ||2e_i - 2e_j||^2 = 4+4=8 >= 4. OK.
    # 2e_i and 2e_i: same, not an issue.
    # So we can add 2e_i vectors, but they conflict with roots containing e_i.
    
    w2_big = []
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s * 2
            w2_big.append(v)
    w2_big = np.array(w2_big, dtype=np.int8)
    
    # Check compatibility of each 2e_i with all current selected vectors
    # Current max norm: need to check
    current_max_norm_sq = np.max(np.sum(selected**2, axis=1)) if len(selected) > 0 else 0
    
    # For now, let's just return what we have
    if len(selected) == 0:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return selected.astype(np.int64)
# EVOLVE-BLOCK-END
