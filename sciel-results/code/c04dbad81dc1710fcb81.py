import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Generate E8 shortest vectors (norm^2 = 2)
    # E8 vectors: 
    # Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) with even number of minus signs
    # Type 2: (±1/2, ..., ±1/2) -> Not integer. 
    # Wait, E8 lattice is integral.
    # The standard integer representation of E8 shortest vectors are:
    # a) Permutations of (±1, ±1, 0, 0, 0, 0, 0, 0) with even number of -1s.
    # b) There are no other integer vectors of norm^2 2 in E8.
    
    e8_shortest = []
    # Generate permutations of (1, 1, 0, 0, 0, 0, 0, 0)
    base = [1, 1, 0, 0, 0, 0, 0, 0]
    seen = set()
    for perm in set(itertools.permutations(base)):
        # Apply signs
        for signs in itertools.product([1, -1], repeat=8):
            v = [p * s for p, s in zip(perm, signs)]
            # Check even number of -1s
            if sum(1 for x in v if x == -1) % 2 == 0:
                if tuple(v) not in seen:
                    seen.add(tuple(v))
                    e8_shortest.append(v)
                    
    # There should be 240 such vectors.
    # Embed in R^11 by padding with 0s
    S2 = []
    for v in e8_shortest:
        S2.append(v + [0, 0, 0])
        
    # 2. Generate A3 shortest vectors (norm^2 = 2)
    # A3 lattice: { (x1, x2, x3, x4) in Z^4 | sum xi = 0 }
    # Shortest vectors are permutations of (1, 1, -1, -1) ? No.
    # Norm^2 = 2.
    # (1, 1, 0, 0) sum=2 != 0.
    # (1, 0, 1, 0) sum=2.
    # In A3, the roots are e_i - e_j.
    # In the 4D hyperplane sum=0, the vectors are (1, -1, 0, 0) and perms.
    # Norm^2 = 1^2 + (-1)^2 = 2.
    # Number of such vectors: 4 * 3 * 2 (signs) ?
    # Pairs (i, j) with i != j. 12 pairs.
    # Vector has 1 at i, -1 at j.
    # So 12 vectors.
    
    a3_shortest = []
    for i in range(4):
        for j in range(4):
            if i != j:
                v = [0, 0, 0, 0]
                v[i] = 1
                v[j] = -1
                a3_shortest.append(v)
                
    # Embed in R^11 by padding with 0s in first 8 dims
    for v in a3_shortest:
        S2.append([0, 0, 0, 0, 0, 0, 0, 0] + v)
        
    S2 = np.array(S2, dtype=np.int64)
    
    # 3. Generate S4 candidates
    # A) E8 norm^2 = 4 vectors
    e8_norm4 = []
    seen4 = set()
    # Type 1: (±2, 0, ..., 0)
    for pos in range(8):
        for sign in [1, -1]:
            v = [0]*8
            v[pos] = 2*sign
            if tuple(v) not in seen4:
                seen4.add(tuple(v))
                e8_norm4.append(v)
                
    # Type 2: Permutations of (1, 1, 1, 1, 0, 0, 0, 0) with even number of -1s
    base4 = [1, 1, 1, 1, 0, 0, 0, 0]
    for perm in set(itertools.permutations(base4)):
        for signs in itertools.product([1, -1], repeat=8):
            v = [p * s for p, s in zip(perm, signs)]
            if sum(1 for x in v if x == -1) % 2 == 0:
                if tuple(v) not in seen4:
                    seen4.add(tuple(v))
                    e8_norm4.append(v)
                    
    for v in e8_norm4:
        S2.append(v + [0, 0, 0]) # These are S4 candidates, but we add to a separate pool
        
    # B) A3 norm^2 = 4 vectors
    a3_norm4 = []
    seen_a3_4 = set()
    # Type 1: Permutations of (1, 1, -1, -1)
    base_a3 = [1, 1, -1, -1]
    for perm in set(itertools.permutations(base_a3)):
        if tuple(perm) not in seen_a3_4:
            seen_a3_4.add(tuple(perm))
            a3_norm4.append(perm)
            
    # Type 2: Permutations of (2, -2, 0, 0)
    base_a3_2 = [2, -2, 0, 0]
    for perm in set(itertools.permutations(base_a3_2)):
        if tuple(perm) not in seen_a3_4:
            seen_a3_4.add(tuple(perm))
            a3_norm4.append(perm)
            
    for v in a3_norm4:
        S2.append([0, 0, 0, 0, 0, 0, 0, 0] + v)
        
    # C) Mixed (e, a) with ||e||^2=2, ||a||^2=2
    # e from e8_shortest, a from a3_shortest
    for e in e8_shortest:
        for a in a3_shortest:
            S2.append(e + a)
            
    # Now S2 contains:
    # 0..240: E8 norm2
    # 240..251: A3 norm2
    # 252..252+576-1: E8 norm4
    # ... A3 norm4
    # ... Mixed
    
    # We need to distinguish S2_base (norm2) and S4_cands (norm4)
    # Actually, we can just run a greedy independent set on the whole set S2.
    # But S2_base is known to be valid.
    # Let's identify indices of norm2 vectors.
    
    norms_sq = np.sum(S2**2, axis=1)
    is_norm2 = norms_sq == 2
    is_norm4 = norms_sq == 4
    
    # Start with all norm2 vectors
    current_set = np.where(is_norm2)[0]
    
    # Compute dot products for norm4 candidates against current_set and themselves
    # To save memory/time, we process norm4 candidates one by one or in batches.
    
    # Precompute dot products of norm4 candidates with norm2 vectors
    # N4 = number of norm4 candidates
    idx4 = np.where(is_norm4)[0]
    N4 = len(idx4)
    
    if N4 == 0:
        return S2[current_set]
        
    S2_4 = S2[idx4]
    S2_2 = S2[current_set]
    
    # Dot products S2_4 x S2_2
    # Condition for u in S2_4, v in S2_2:
    # 2 * u.dot(v) <= ||u|| * ||v|| = 2 * sqrt(2)
    # u.dot(v) <= sqrt(2) ~ 1.41
    # Since integer, u.dot(v) <= 1
    
    dots_4_2 = S2_4 @ S2_2.T
    conflicts_4_2 = dots_4_2 > 1
    
    # For each norm4 candidate, count how many norm2 vectors it conflicts with
    # We want to add candidates with 0 conflicts first.
    
    # Also need internal conflicts among norm4 candidates
    # Condition for u, v in S2_4:
    # 2 * u.dot(v) <= 2 * 2 = 4
    # u.dot(v) <= 2
    
    # We can't compute full N4 x N4 matrix if N4 is large (3474 -> 12M entries, ok).
    dots_4_4 = S2_4 @ S2_4.T
    conflicts_4_4 = dots_4_4 > 2
    np.fill_diagonal(conflicts_4_4, False)
    
    # Greedy selection
    # Sort norm4 candidates by number of conflicts with current set (norm2)
    # Then by number of internal conflicts?
    
    conflict_counts_2 = np.sum(conflicts_4_2, axis=1)
    
    # We will maintain a set of selected norm4 indices
    selected_4 = []
    # Map global index to local index in S2_4
    # local_idx -> global_idx = idx4[local_idx]
    
    # We need to track which norm2 vectors are "blocked" by selected norm4 vectors?
    # No, norm2 vectors are already in the set.
    # We just need to ensure new norm4 vectors don't conflict with existing norm2 OR existing norm4.
    
    # Efficient approach:
    # 1. Filter out norm4 candidates that conflict with ANY norm2 vector.
    #    If a candidate conflicts with a norm2 vector, it can NEVER be added (since norm2 vectors are fixed).
    #    So we only consider candidates with conflict_counts_2 == 0.
    
    valid_4_local = np.where(conflict_counts_2 == 0)[0]
    
    if len(valid_4_local) == 0:
        return S2[current_set]
        
    # Restrict S2_4 to valid_4_local
    S2_4_valid = S2_4[valid_4_local]
    N4v = len(S2_4_valid)
    
    # Compute internal conflicts for this subset
    dots_v = S2_4_valid @ S2_4_valid.T
    conflicts_v = dots_v > 2
    np.fill_diagonal(conflicts_v, False)
    
    # Greedy independent set on this subgraph
    # Use min-degree heuristic
    alive = np.ones(N4v, dtype=bool)
    degrees = np.sum(conflicts_v, axis=1)
    
    final_selected_local = []
    
    while np.any(alive):
        # Find min degree
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0: break
        
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[0] # Deterministic for stability, or random
        
        final_selected_local.append(chosen)
        
        # Remove chosen and its neighbors from alive
        nbrs = np.where(conflicts_v[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], nbrs)))
        alive[to_remove] = False
        
        # Update degrees for remaining alive nodes
        rem = np.where(alive)[0]
        if len(rem) > 0:
             degrees[rem] -= conflicts_v[rem][:, to_remove].sum(axis=1)
             
    # Map back to global indices
    selected_4_global = idx4[valid_4_local[final_selected_local]]
    
    # Combine
    final_indices = np.concatenate([current_set, selected_4_global])
    
    return S2[final_indices]

# EVOLVE-BLOCK-END
