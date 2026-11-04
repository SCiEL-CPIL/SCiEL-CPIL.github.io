"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate the seed set: All vectors with two non-zero entries of value +/-1
    # Total: C(11,2) * 4 = 55 * 4 = 220 vectors
    # Norm squared = 2
    v_seed = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                v_seed.append(v)
    
    vectors = np.array(v_seed, dtype=np.int64)
    n_vectors = len(vectors)
    
    # Compute norms squared
    norms_sq = np.sum(vectors**2, axis=1)
    max_norm_sq = np.max(norms_sq)  # Should be 2
    
    # Compute pairwise dot products for the seed set
    # dist_sq = norm_a^2 + norm_b^2 - 2 * dot(a,b)
    # Constraint: dist_sq >= max_norm_sq
    # norm_a^2 + norm_b^2 - 2 * dot(a,b) >= max_norm_sq
    # Since all seeds have norm_sq=2, max_norm_sq=2:
    # 2 + 2 - 2 * dot >= 2  =>  2 - 2*dot >= 0  => dot <= 1
    
    # We want to add more vectors.
    # Candidate types to try:
    # A. Vectors with 3 non-zero entries of value +/-1 (Norm squared = 3)
    #    Constraint against seed (norm_sq=2):
    #    2 + 3 - 2 * dot >= 3  =>  2 - 2*dot >= 0 => dot <= 1
    #    Constraint against other type A (norm_sq=3):
    #    3 + 3 - 2 * dot >= 3  =>  3 - 2*dot >= 0 => dot <= 1.5 => dot <= 1 (integer)
    
    # B. Vectors with 2 non-zero entries of value +/-2? No, let's stick to small integers.
    # C. Vectors with 1 non-zero entry? Norm 1.
    #    Constraint against seed (norm_sq=2):
    #    2 + 1 - 2 * dot >= 2 => 1 - 2*dot >= 0 => dot <= 0.5 => dot <= 0.
    #    If v has one non-zero at index k, and u has non-zeros at i,j.
    #    dot is non-zero only if k is in {i,j}.
    #    So we must ensure orthogonality with all existing seeds?
    #    If we add e_k (unit vector), it must be orthogonal to all seeds involving index k.
    #    There are 10 seeds involving k. So we cannot add any e_k if any seed involving k exists.
    #    Since all seeds involving k exist, we cannot add any unit vectors.

    # Let's focus on Type A: 3 non-zeros, +/-1.
    # Total candidates: C(11,3) * 2^3 = 165 * 8 = 1320.
    # We will generate all these candidates and greedily add them.
    
    # Pre-compute dot products of seeds with each other is not needed if we maintain a matrix.
    # Let's maintain a matrix of dot products with the current set of vectors.
    
    # Current set: 220 vectors.
    # Candidate set: 1320 vectors.
    
    # We need to check:
    # 1. dot(candidate, existing_vector) <= 1 for all existing vectors?
    #    Actually, the condition is:
    #    ||c - v||^2 >= max_norm_sq (current max)
    #    ||c||^2 + ||v||^2 - 2 c.v >= max_norm_sq
    #    
    #    If we add a vector of norm_sq=3, the new max_norm_sq becomes 3.
    #    Then we must re-check all pairs?
    #    Yes, if max_norm_sq increases, the constraint tightens for all existing pairs.
    #    
    #    Seed-Seed pairs: max norm was 2. Now max norm is 3.
    #    Check: ||u - w||^2 >= 3?
    #    ||u||=2, ||w||=2. ||u-w||^2 = 4 - 2 u.w.
    #    We need 4 - 2 u.w >= 3 => 1 >= 2 u.w => u.w <= 0.5 => u.w <= 0.
    #    
    #    Wait. The seed set has vectors like (1,1,0...) and (1,-1,0...).
    #    u=(1,1,0...), w=(1,1,0...). u.w = 2.
    #    dist^2 = 4 - 4 = 0.
    #    But u and w are distinct vectors in the set?
    #    No, the seed generation produces unique vectors.
    #    u=(1,1,0...), w=(-1,1,0...). u.w = -1+1 = 0. dist^2 = 4.
    #    u=(1,1,0...), w=(1,-1,0...). u.w = 1-1 = 0. dist^2 = 4.
    #    u=(1,1,0...), w=(-1,-1,0...). u.w = -1-1 = -2. dist^2 = 8.
    #    
    #    What about u=(1,1,0...) and w=(1,1,0...)? They are the same vector. We don't have duplicates.
    #    
    #    Let's check if any two distinct seed vectors have dot product > 0.
    #    Seeds have support of size 2.
    #    If supports are disjoint, dot=0.
    #    If shares 1 index: u=(1,1,0...), w=(1,2,0...) -> w not in seed. w=(1, -1, 0...) -> dot=0.
    #    If shares 2 indices (same support): u=(1,1), w=(1,-1) -> dot=0. u=(1,1), w=(-1,1) -> dot=0.
    #    So all distinct seed pairs have dot product 0?
    #    Let's verify.
    #    u = (si, sj) at (i,j). w = (sk, sl) at (k,l).
    #    If {i,j} != {k,l}, they share at most 1 index.
    #    If share i: u_i=si, w_i=sk. Others 0. Dot = si*sk.
    #    si, sk in {1,-1}. Dot is +/-1.
    #    Example: u=(1,1,0...), w=(1,-1,0...). Support {0,1}. Dot = 1*(-1) + 1*(-1)? No.
    #    u=(1,1) at 0,1. w=(1,-1) at 0,1.
    #    u = [1, 1, 0...]. w = [1, -1, 0...].
    #    Dot = 1*1 + 1*(-1) = 0.
    #    
    #    Example: u=(1,1) at 0,1. v=(1,1) at 1,2.
    #    u = [1, 1, 0...]. v = [0, 1, 1...].
    #    Dot = 1*0 + 1*1 + 0*1 = 1.
    #    
    #    Ah, so some seed pairs have dot product 1.
    #    If max_norm_sq becomes 3, we require ||u-v||^2 >= 3.
    #    4 - 2(1) = 2 < 3.
    #    So if we add a vector of norm 3, the existing seed set becomes INVALID because some pairs are too close.
    #    
    #    Therefore, we cannot simply add norm-3 vectors to the seed set.
    #    We must either:
    #    1. Replace the seed set with a better one.
    #    2. Add vectors of norm 2 that are far enough from each other.
    #    
    #    Are there more norm-2 vectors?
    #    Norm squared 2 implies two entries are +/-1, others 0.
    #    We already have all of them (220).
    #    
    #    So we are stuck at 220 if we require min_dist >= max_norm.
    #    
    #    Wait, the task says: "minimum pairwise distance is at least the maximum vector norm".
    #    If we use a different set where all vectors have the same norm, say R, then min_dist >= R.
    #    If we use vectors with different norms, let M = max norm. Then all pairs must be dist >= M.
    #    
    #    Can we find a set of vectors with norm > 2?
    #    If we use vectors of norm sqrt(4)=2 (e.g., two 1s and two -1s? No, norm^2=4 means sum of squares is 4).
    #    Example: (1,1,1,1,0...) norm^2=4.
    #    If we use a set of norm^2=4 vectors.
    #    Then min_dist >= 2.
    #    
    #    Let's check the Leech lattice vectors.
    #    In dimension 24, the kissing number is 196560.
    #    In dimension 11, the kissing number is 1408? 
    #    Actually, the kissing number for d=11 is known to be 1408?
    #    Let me check known values.
    #    d=1: 2
    #    d=2: 6
    #    d=3: 12
    #    d=4: 24
    #    d=5: 40
    #    d=6: 72
    #    d=7: 126
    #    d=8: 240
    #    d=9: 460
    #    d=10: 900 (lower bound), 1169? No.
    #    d=11: 1408?
    #    
    #    The seed gives 220.
    #    If the kissing number is indeed ~1408, we can do much better.
    #    
    #    How to construct them?
    #    The vectors can be generated from the E8 lattice or similar.
    #    But in dimension 11, we can use the fact that we can have vectors with more non-zero entries.
    #    
    #    Let's try to construct a set of vectors with norm^2 = 4.
    #    Candidates: 
    #    - 4 entries of 1: (1,1,1,1,0...). C(11,4) = 330. 330 vectors.
    #    - 2 entries of 2: (2,2,0...). C(11,2)*4 = 220.
    #    - 2 entries of 1, 1 entry of sqrt(2)? No, integers.
    #    - 2 entries of 1, 2 entries of 1? (1,1,1,1) is 4 entries.
    #    
    #    If we take all 330 vectors of type (1,1,1,1,0...):
    #    Norm^2 = 4. Max norm = 2.
    #    Check distances between them.
    #    u=(1,1,1,1) at 0,1,2,3.
    #    v=(1,1,1,1) at 0,1,2,4.
    #    Intersection: 3 indices.
    #    dot = 3.
    #    dist^2 = 4 + 4 - 2(3) = 2.
    #    2 < 4 (Max norm squared is 4, so max norm is 2. Constraint is dist >= 2, i.e., dist^2 >= 4).
    #    So 2 < 4. Invalid.
    #    
    #    We need dist^2 >= 4.
    #    8 - 2 dot >= 4 => 2 dot <= 4 => dot <= 2.
    #    So any two vectors in the set must have dot product <= 2.
    #    
    #    For (1,1,1,1) type vectors, if they share 3 indices, dot=3. Invalid.
    #    So we can only pick vectors that share at most 2 indices.
    #    This is a packing problem.
    #    
    #    Alternatively, mix types?
    #    
    #    Let's look at the "seed" again.
    #    Seed: 220 vectors, norm^2=2.
    #    
    #    Can we add vectors of norm^2=3 to the seed, IF we remove some seeds?
    #    Or just build a new set from scratch?
    #    
    #    Given the time limit, implementing a full lattice generator is risky.
    #    However, 220 is a very low bound for d=11.
    #    
    #    Let's try to generate vectors with norm^2=4 from the set of all integer vectors with small norm.
    #    And use a greedy algorithm to select a subset with min_dist^2 >= 4.
    #    
    #    Candidate Pool:
    #    1. Type (1,1,1,1): 330 vectors.
    #    2. Type (2,2,0...): 220 vectors.
    #    3. Type (2,1,1,0...): 11*10*9 * 2 * 2 * 2? 
    #       2 at i, 1 at j, 1 at k.
    #       C(11,1) * C(10,2) * 2 (sign of 2) * 2 (sign of j) * 2 (sign of k) = 11 * 45 * 8 = 3960.
    #    
    #    Total candidates ~ 4500.
    #    Greedy selection:
    #    Sort candidates by "centrality" or just iterate.
    #    For each candidate, check if it is compatible with all currently selected vectors.
    #    
    #    Compatibility check:
    #    For candidate c and selected s:
    #    ||c - s||^2 >= max_norm_sq_selected
    #    
    #    This is O(N_selected * N_candidates).
    #    4500 * 4500 = 20M ops. Fast enough in Python with NumPy vectorization?
    #    Maybe.
    
    # Let's implement this greedy approach.
    
    # Generate Candidates
    candidates = []
    
    # Type 1: Four 1s
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
            
    # Type 2: Two 2s
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = 2*si, 2*sj
                candidates.append(v)
                
    # Type 3: One 2, Two 1s
    for i in range(D):
        for j, k in itertools.combinations([x for x in range(D) if x != i], 2):
            for si in (1, -1):
                for sj in (1, -1):
                    for sk in (1, -1):
                        v = [0] * D
                        v[i] = 2*si
                        v[j] = sj
                        v[k] = sk
                        candidates.append(v)
                        
    candidates = np.array(candidates, dtype=np.int64)
    if len(candidates) == 0:
        return vectors # Fallback to seed
    
    # Remove duplicates just in case
    # candidates = np.unique(candidates, axis=0)
    
    # Greedy Selection
    selected = []
    selected_norms_sq = []
    max_norm_sq = 0
    
    # We can shuffle candidates to avoid bias, or sort by norm?
    # Let's try sorting by norm squared descending to place large vectors first?
    # Or maybe ascending to fill space?
    # Let's just iterate.
    
    # To speed up, we can vectorize the distance check.
    # But the set grows, so we need to check against all selected.
    
    # Optimization: 
    # If we select a vector of norm^2=4, max_norm_sq becomes 4.
    # All future vectors must be dist >= 2 from it.
    # 
    # Let's try to select a large set.
    
    # Shuffle candidates to get a random start? 
    # For reproducibility in this context, let's just use the order.
    
    for c in candidates:
        c_norm_sq = np.dot(c, c)
        
        # If adding this vector increases max_norm_sq, we must re-verify previous ones?
        # No, we only add if it's compatible with ALL existing ones.
        # But if c_norm_sq > max_norm_sq, the new max_norm_sq is c_norm_sq.
        # The condition is ||c - s||^2 >= NEW_max_norm_sq.
        # Since NEW_max_norm_sq >= OLD_max_norm_sq, the condition is stricter.
        # So we just check ||c - s||^2 >= c_norm_sq (if c is the new max) or max_norm_sq (if not).
        # Actually, the constraint is:
        # For all s in selected: ||c - s||^2 >= max(c_norm_sq, max_norm_sq) ?
        # No. The global max norm is M.
        # Constraint: dist(u,v) >= M for ALL pairs.
        # If we add c, new M' = max(M, ||c||).
        # We need:
        # 1. dist(c, s) >= M' for all s in selected.
        # 2. dist(s, t) >= M' for all s,t in selected (previously dist >= M, now need >= M').
        #    If M' > M, we might have violated the condition for existing pairs!
        #    Example: s,t had dist^2 = M^2. Now M' > M. dist(s,t) < M'. Invalid.
        
        # So, if we add a vector with larger norm, we might have to remove some existing vectors.
        # This makes greedy hard.
        
        # Strategy: 
        # Only add vectors with norm <= current max_norm_sq?
        # If we start with seeds (norm^2=2), we can only add vectors with norm^2 <= 2.
        # But we established there are no more norm^2=2 vectors.
        # So we are stuck at 220.
        
        # Alternative Strategy:
        # Start with an empty set.
        # Add vectors one by one.
        # If we add a vector with norm^2=4, max_norm_sq becomes 4.
        # Then we can only add vectors with norm^2 <= 4.
        # And they must be dist >= 2 from all existing.
        
        # So we should prioritize adding high-norm vectors first?
        # If we add a norm^2=4 vector first, max_norm_sq=4.
        # Then we add other norm^2=4 vectors that are dist>=2.
        # Then we can add norm^2=2 vectors?
        # If we add a norm^2=2 vector v, dist(v, u) >= 2 (max norm) => dist^2 >= 4.
        # ||v||^2 + ||u||^2 - 2 dot >= 4
        # 2 + 4 - 2 dot >= 4 => 2 - 2 dot >= 0 => dot <= 1.
        
        # This seems feasible.
        
        # Let's reorder candidates by norm squared descending.
        # This way, we establish the high max_norm early.
        
        norms_sq_candidates = np.sum(candidates**2, axis=1)
        indices = np.argsort(-norms_sq_candidates)
        candidates_sorted = candidates[indices]
        
        selected = []
        max_norm_sq = 0
        
        for c in candidates_sorted:
            c_norm_sq = np.dot(c, c)
            new_max_norm_sq = max(max_norm_sq, c_norm_sq)
            
            # Check if c is compatible with all selected, given the NEW max norm.
            # dist(c, s)^2 >= new_max_norm_sq
            # ||c||^2 + ||s||^2 - 2 c.s >= new_max_norm_sq
            
            if len(selected) == 0:
                selected.append(c)
                max_norm_sq = new_max_norm_sq
                continue
                
            # Vectorized check against all selected
            # selected is a list, convert to array? 
            # For efficiency, keep selected as array.
            
            if not hasattr(construct, 'sel_arr'):
                pass # Just a placeholder
                
            # To avoid creating a new array every time, we can maintain the array.
            # But for simplicity and correctness, let's use a list and convert.
            # With ~1000 selected and 4000 candidates, 4M checks. 
            # Python loops are slow.
            
            # Let's maintain selected as a NumPy array.
            if not isinstance(selected, np.ndarray):
                selected = np.array(selected, dtype=np.int64) if selected else np.array([], dtype=np.int64).reshape(0, D)
                
            if len(selected) > 0:
                # dist^2 = ||c||^2 + ||s||^2 - 2 c.s
                # We need dist^2 >= new_max_norm_sq for all s.
                # min_dist_sq = min( ||c||^2 + ||s||^2 - 2 c.s )
                
                # c is (D,), selected is (N, D)
                # dots = selected @ c  (N,)
                dots = selected @ c
                s_norms_sq = np.sum(selected**2, axis=1)
                
                dist_sq = c_norm_sq + s_norms_sq - 2 * dots
                min_dist_sq = np.min(dist_sq)
                
                if min_dist_sq >= new_max_norm_sq:
                    # Also need to check if existing pairs are still valid with new_max_norm_sq
                    # If new_max_norm_sq > max_norm_sq, we need to check all pairs in selected.
                    # This is expensive O(N^2).
                    # However, if we sort by norm descending, the first vector sets the max.
                    # Subsequent vectors have norm <= max.
                    # So new_max_norm_sq will NOT increase after the first few vectors.
                    # Once we add a vector with norm^2=4, max_norm_sq=4.
                    # All subsequent vectors have norm^2 <= 4.
                    # So new_max_norm_sq remains 4.
                    # Thus, we don't need to re-check existing pairs, as long as we don't increase max_norm_sq.
                    
                    selected = np.vstack([selected, c])
                    max_norm_sq = new_max_norm_sq
            else:
                selected = np.array([c], dtype=np.int64)
                max_norm_sq = new_max_norm_sq
                
            if time.time() > deadline:
                break
        
        if len(selected) > len(vectors):
            return selected
        else:
            return vectors

# EVOLVE-BLOCK-END
