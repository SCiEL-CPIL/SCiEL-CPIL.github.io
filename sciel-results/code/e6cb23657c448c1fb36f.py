import itertools
import numpy as np
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Generate all weight-2 vectors (±1 in 2 positions)
    w2_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                w2_vectors.append(v)
    w2 = np.array(w2_vectors)
    
    # Generate all weight-3 vectors (±1 in 3 positions)
    w3_vectors = []
    for indices in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            w3_vectors.append(v)
    w3 = np.array(w3_vectors)
    
    # Combined candidate pool
    all_candidates = np.vstack([w2, w3])
    n_cand = len(all_candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(all_candidates * all_candidates, axis=1)
    
    # Precompute pairwise distance squared matrix
    # dist_sq[i,j] = ||v_i - v_j||^2
    # For efficiency, compute in chunks if needed, but n_cand is ~1100 so it's fine
    # Actually n_cand = 220 + 1320 = 1540
    # dist_sq matrix is 1540 x 1540 = ~2.4M entries, fine for numpy
    
    # dist^2 = ||a||^2 + ||b||^2 - 2*a.b
    # Compute dot product matrix
    dot_prod = all_candidates @ all_candidates.T
    dist_sq = norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - 2 * dot_prod
    np.fill_diagonal(dist_sq, 0)
    
    # We want minimum pairwise distance >= max norm in the set
    # Strategy: Greedy min-degree selection
    # Start with empty set, iteratively add the candidate that is compatible with the fewest remaining candidates
    # (i.e., least constraining)
    
    # Actually, let's use a different approach:
    # 1. Start with all weight-2 vectors as base
    # 2. Greedily add weight-3 vectors that are compatible with all current vectors
    # 3. Use local search to improve
    
    # First, let's try a clean greedy approach on the combined pool
    
    # Compatibility: vector a is compatible with vector b iff ||a-b||^2 >= max(||a||^2, ||b||^2)
    # In a set S, all pairs must satisfy this, AND the min pairwise dist >= max norm in S
    
    # Let's define: a set S is valid if min_{a,b in S, a!=b} ||a-b||^2 >= max_{a in S} ||a||^2
    
    # Greedy min-degree: at each step, among all candidates compatible with current set,
    # pick the one that is incompatible with the fewest other candidates (to leave options open)
    
    in_set = np.zeros(n_cand, dtype=bool)
    current_max_norm = 0
    
    # Start: add the smallest norm vectors first to establish a low max_norm baseline
    # Sort by norm
    order = np.argsort(norms_sq)
    
    # Add all weight-2 vectors first (they all have norm^2 = 2)
    # Check: two weight-2 vectors a, b. ||a-b||^2 can be 0, 2, 4, 8
    # If they share both positions with same sign: dist^2 = 0 -> not compatible
    # If they share one position: dist^2 = 2 + 2 - 2*(1 or -1) = 2 or 6
    #   Same sign on shared: dot=1, dist^2 = 4-2 = 2. Need >= 2. OK.
    #   Diff sign on shared: dot=-1, dist^2 = 4+2 = 6. Need >= 2. OK.
    # If they share no positions: dist^2 = 4. Need >= 2. OK.
    # So all distinct weight-2 vectors are compatible! (min dist^2 = 2 = max norm^2)
    
    # Add all weight-2 vectors
    w2_indices = list(range(len(w2_vectors)))
    for idx in w2_indices:
        in_set[idx] = True
        current_max_norm = max(current_max_norm, norms_sq[idx])
    
    # Now greedily add weight-3 vectors
    # A weight-3 vector v has norm^2 = 3
    # For v to be compatible with a weight-2 vector u (norm^2=2):
    #   Need ||v-u||^2 >= 3
    #   ||v-u||^2 = ||v||^2 + ||u||^2 - 2*v.u = 3 + 2 - 2*v.u = 5 - 2*v.u
    #   Need 5 - 2*v.u >= 3 => v.u <= 1
    #   v.u is the dot product, which is sum of products of coordinates
    #   Since v and u have ±1 entries, v.u can be -3,-1,1,3 if they overlap in 3 positions
    #   If overlap in 2 positions: v.u in {-2,0,2}
    #   If overlap in 1 position: v.u in {-1,1}
    #   If overlap in 0 positions: v.u = 0
    #   So v.u <= 1 is satisfied unless v.u = 3 (all 3 shared positions have same sign)
    
    # For two weight-3 vectors v, w (both norm^2=3):
    #   Need ||v-w||^2 >= 3
    #   ||v-w||^2 = 3 + 3 - 2*v.w = 6 - 2*v.w
    #   Need 6 - 2*v.w >= 3 => v.w <= 1.5 => v.w <= 1
    #   v.w is dot product of two weight-3 vectors
    
    # Greedy: for each weight-3 vector, count how many weight-2 vectors it's incompatible with
    # (i.e., where v.u > 1, meaning v.u = 3 for weight-2 case)
    # Wait, for weight-2 u and weight-3 v: incompatible if v.u > 1
    # v.u can be 3 only if all 3 positions of v are also in u... but u has only 2 positions.
    # So max overlap is 2, and v.u in {-2, 0, 2} when overlap is 2.
    # Actually if overlap is 2: v has ±1 in 3 positions, u has ±1 in 2 positions.
    # If the 2 positions of u are a subset of v's 3 positions, then v.u = sum over those 2 positions of v_i * u_i
    # This can be -2, 0, or 2.
    # So v.u > 1 means v.u = 2, which means both shared positions have same sign.
    
    # Let's just do the greedy with precomputed compatibility
    
    # Precompute for each candidate, which other candidates it's incompatible with
    # incompatible[i,j] = True if dist_sq[i,j] < max(norms_sq[i], norms_sq[j])
    # But this depends on the current set's max norm... 
    
    # Actually the condition is: for the final set S, min pairwise dist^2 >= max norm^2 in S
    # This is equivalent to: for all pairs a,b in S: dist^2(a,b) >= max(norm^2(a), norm^2(b)) >= min_{c in S} norm^2(c)
    # And also dist^2(a,b) >= norm^2(c) for all c in S, but since max norm is the binding constraint...
    # Actually: min_{a,b in S} dist^2(a,b) >= max_{c in S} norm^2(c)
    # This is equivalent to: for all a,b in S: dist^2(a,b) >= max_{c in S} norm^2(c)
    
    # So if we track the current max norm, a new vector v is compatible with set S iff:
    #   1. For all u in S: dist^2(v,u) >= current_max_norm (since current_max_norm <= new_max_norm)
    #   2. dist^2(v,u) >= norms_sq[v] for all u in S (if norms_sq[v] > current_max_norm)
    #   3. dist^2(v,u) >= norms_sq[u] for all u in S (already implied by 1 if current_max_norm >= norms_sq[u])
    
    # Let's simplify: a vector v is compatible with current set S iff for all u in S:
    #   dist^2(v,u) >= max(norms_sq[v], norms_sq[u], current_max_norm)
    # Since current_max_norm >= norms_sq[u] for all u in S, this becomes:
    #   dist^2(v,u) >= max(norms_sq[v], current_max_norm)
    
    # Precompute a "constraint" for each pair: the minimum norm threshold they'd impose
    # For a pair (i,j), they are compatible if current_max_norm <= dist_sq[i,j] AND norms_sq[i] <= dist_sq[i,j] AND norms_sq[j] <= dist_sq[i,j]
    # i.e., current_max_norm <= min(dist_sq[i,j], norms_sq[i], norms_sq[j])
    # Let threshold[i,j] = min(dist_sq[i,j], norms_sq[i], norms_sq[j])
    # Then i and j are compatible in a set with max norm M iff M <= threshold[i,j]
    
    threshold = np.minimum.reduce([dist_sq, norms_sq[:, np.newaxis], norms_sq[np.newaxis, :]])
    # threshold[i,j] = min(dist_sq[i,j], norms_sq[i], norms_sq[j])
    
    # Now greedy: maintain current_max_norm. A candidate j is compatible with set S iff
    #   for all i in S: current_max_norm <= threshold[i,j]
    #   AND norms_sq[j] can be added: new_max = max(current_max_norm, norms_sq[j])
    #   And for the new vector to be compatible with all existing: dist_sq[i,j] >= new_max for all i in S
    #   Which is: min_i dist_sq[i,j] >= new_max
    #   Since dist_sq[i,j] >= threshold[i,j] and threshold[i,j] <= dist_sq[i,j]... 
    #   Actually threshold[i,j] = min(dist_sq[i,j], norms_sq[i], norms_sq[j]) <= dist_sq[i,j]
    #   So threshold being >= M doesn't guarantee dist_sq >= M... wait, if threshold[i,j] >= M, then dist_sq[i,j] >= threshold[i,j] >= M. Yes.
    
    # So compatibility check: for candidate j with current set S and current_max_norm M:
    #   new_M = max(M, norms_sq[j])
    #   j is compatible iff min_{i in S} dist_sq[i,j] >= new_M
    #   Equivalently: min_{i in S} dist_sq[i,j] >= new_M
    
    # Let's do greedy min-degree: among compatible candidates, pick the one with fewest compatible partners
    # remaining (to preserve flexibility)
    
    # For efficiency, precompute for each candidate j, the sorted list of (dist_sq[i,j], i) 
    # Actually, let's just iterate
    
    added_any = True
    n_current = len(w2_indices)
    
    while added_any and time.time() < deadline:
        added_any = False
        new_M = current_max_norm
        
        # Find all candidates compatible with current set
        compatible = []
        for j in range(n_cand):
            if in_set[j]:
                continue
            # Check compatibility: min dist to any in-set vector >= max(norms_sq[j], current_max_norm)
            new_M_j = max(norms_sq[j], current_max_norm)
            # Need dist_sq[i,j] >= new_M_j for all i in set
            # This is equivalent to: min over in-set i of dist_sq[i,j] >= new_M_j
            # For speed, check against a few vectors first
            min_dist_to_set = np.inf
            for i in w2_indices[:50]:  # Check first 50 weight-2 vectors
                if dist_sq[i, j] < new_M_j:
                    break
            else:
                # Check all
                all_ok = True
                for i in range(len(w2_indices)):
                    if dist_sq[i, j] < new_M_j:
                        all_ok = False
                        break
                if all_ok:
                    # Also check non-w2 in set
                    for idx2 in range(len(w2_indices), n_cand):
                        if in_set[idx2] and dist_sq[idx2, j] < new_M_j:
                            all_ok = False
                            break
                    if all_ok:
                        compatible.append(j)
        
        if not compatible:
            break
        
        # Among compatible, pick the one with min degree (fewest compatible partners among remaining)
        # Approximate: count how many other compatible candidates it's incompatible with
        # For speed, just pick the first compatible one
        j_pick = compatible[0]
        
        in_set[j_pick] = True
        current_max_norm = max(current_max_norm, norms_sq[j_pick])
        added_any = True
    
    # Build result
    result_indices = np.where(in_set)[0]
    result = all_candidates[result_indices]
    
    # Validate
    if len(result) == 0:
        return w2
    
    norms_result = np.sum(result * result, axis=1)
    max_norm = np.max(norms_result)
    
    valid = True
    n_res = len(result)
    if n_res <= 500:
        for i in range(n_res):
            for j in range(i + 1, n_res):
                diff = result[i] - result[j]
                dist_sq_val = np.dot(diff, diff)
                if dist_sq_val < max_norm:
                    valid = False
                    break
            if not valid:
                break
    else:
        # Chunked validation
        for i in range(0, n_res, 100):
            if time.time() > deadline - 5:
                break
            chunk = result[i:i+100]
            diff = chunk[:, np.newaxis, :] - result[np.newaxis, :, :]
            dsq = np.sum(diff * diff, axis=2)
            # Check upper triangle of this chunk vs all
            for ci in range(len(chunk)):
                if np.min(dsq[ci, :]) < max_norm:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        return w2
    
    return result
