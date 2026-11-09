import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    DEADLINE = 600.0
    
    # 1. Initialize with the known optimal set for M=2: all 220 weight-2 vectors
    #    This set is valid because for any u, v in this set, ||u-v||^2 >= 2 = max_norm.
    pts_list = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    pts_list.append(v)
    
    points = np.array(pts_list, dtype=np.int64)
    current_M = 2
    
    # Precompute Gram matrix for fast distance checks
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u.v
    # We maintain a matrix of dot products with new candidates?
    # Actually, for a new candidate c, we need to check:
    # ||c - p||^2 >= M_new for all p in points.
    # ||c||^2 + ||p||^2 - 2 c.p >= M_new
    # Since ||p||^2 <= current_M (and M_new is usually close to current_M or ||c||^2),
    # the tightest constraint is usually when ||p||^2 is small? No.
    # Let's just compute distances vectorized.
    
    def get_candidates(max_norm_sq):
        """
        Generate integer vectors with norm^2 <= max_norm_sq.
        We focus on norms that are likely to be compatible.
        For M=2, only norm 2 works (mostly).
        For M=4, norm 2 and 4 work.
        """
        cands = []
        seen = set()
        
        def add(v):
            key = tuple(v)
            if key not in seen:
                seen.add(key)
                cands.append(v)
        
        # We iterate by support size (number of non-zero entries)
        # To be efficient, we limit the search to small norms first.
        
        # Norm 1: (1, 0, ...) - 22 vectors
        for i in range(d):
            for s in [-1, 1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = s
                add(v)
                
        # Norm 2: (1, 1, 0...), (2, 0...), etc.
        # We already have the (1,1) ones in the initial set, but we might want to add them
        # to the candidate pool if we restart or if we want to consider them for expansion?
        # The initial set IS the norm 2 weight-2 vectors.
        # We should probably generate candidates with norm > current_M or similar.
        # But wait, we can mix norms.
        
        # Let's generate all vectors with norm^2 up to max_norm_sq.
        # To keep it fast, we limit the number of non-zero entries.
        max_nz = 4 # heuristic
        
        for nz in range(1, max_nz + 1):
            for indices in itertools.combinations(range(d), nz):
                # Generate all combinations of values
                # For simplicity, we iterate over possible squared norms
                # This is expensive if max_norm_sq is large.
                # Let's restrict values to small integers.
                
                # Optimization: Only generate if we haven't already.
                # Actually, for the greedy step, we want to try adding a vector.
                # We can generate a batch of candidates and try to add them one by one.
                
                pass # The loop above is just structure. We need actual generation.
                
        # Better generation strategy for the greedy loop:
        # We will try to increase M.
        # Start with M=2.
        # Try to add vectors with norm^2 = 3? No integer vectors have norm^2=3 in small dims?
        # 1^2+1^2+1^2 = 3. Yes.
        # Try to add vectors with norm^2 = 4? 2^2, 1^2+1^2+1^2+1^2.
        
        return cands

    # The previous generate function was too generic and slow.
    # Let's use a specific "expansion" strategy.
    
    # We know that mixing norms allows larger sets.
    # Example: Set of all vectors with norm^2 <= 4?
    # If we take ALL vectors with norm^2 <= 4, is it valid?
    # Max norm M=4.
    # Check pair (1,0,..) and (-1,0,..). Dist^2 = 4. OK.
    # Check pair (1,1,0..) and (1,-1,0..). Dist^2 = (0)^2 + (2)^2 = 4. OK.
    # Check pair (1,1,0..) and (1,1,0..). Dist^2 = 0. FAIL (must be distinct).
    # So we can't take duplicates.
    # Check pair (1,1,0..) and (2,0,0..). Dist^2 = (1-2)^2 + (1-0)^2 = 1+1=2.
    # 2 < M=4. FAIL.
    # So the set of ALL vectors with norm^2 <= 4 is NOT valid.
    
    # We need a greedy selection from the pool of vectors with norm^2 <= M_max.
    
    # Strategy:
    # 1. Start with the 220 weight-2 vectors (M=2).
    # 2. Generate a pool of "candidate" vectors with norm^2 in {3, 4, 5, ...}.
    # 3. Try to add them one by one, updating M as needed.
    # 4. If adding a vector with norm^2 = N increases M to N, we must check
    #    that it is at distance >= N from ALL existing points.
    
    # Let's implement this specific greedy.
    
    def validate(points):
        k = len(points)
        if k < 2:
            return False
        norms = np.sum(points**2, axis=1)
        max_norm = int(np.max(norms))
        for i in range(k):
            if np.all(points[i] == 0):
                return False
        # Vectorized distance check
        # dist^2_{ij} = ||p_i||^2 + ||p_j||^2 - 2 p_i . p_j
        # We need min(dist^2) >= max_norm
        # Compute Gram matrix G = P P^T
        G = points @ points.T
        norms_col = norms[:, np.newaxis]
        dist2 = norms_col + norms_col.T - 2 * G
        # np.fill_diagonal(dist2, np.inf)
        min_dist2 = int(np.min(dist2))
        return min_dist2 >= max_norm

    best_points = points.copy()
    best_k = len(points)
    
    # Candidate generation for higher norms
    # We will try to expand M from 2 upwards.
    # For each target M, we generate candidates with norm^2 <= M.
    # We prioritize candidates with norm^2 == M (to push M up) or norm^2 < M (to fill space).
    
    # To make this efficient, we pre-generate a large pool of low-norm vectors.
    # Norm 1: 22
    # Norm 2: 220 (already in)
    # Norm 3: 11 choose 3 * 8 = 165 * 8 = 1320
    # Norm 4: 11 choose 1 * 2 (val 2) + 11 choose 2 * 4 (val 1,1) + 11 choose 4 * 16 (val 1,1,1,1)
    #         22 + 44*4 + 330*16 = 22 + 176 + 5280 = 5478
    # Total pool ~ 7000. This is manageable.
    
    def generate_pool(max_norm_sq):
        pool = []
        seen = set()
        
        def add(v):
            key = tuple(v)
            if key not in seen:
                seen.add(key)
                pool.append(v)
                
        # Norm 1
        for i in range(d):
            for s in [-1, 1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = s
                add(v)
                
        # Norm 2
        for i in range(d):
            for j in range(i+1, d):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        add(v)
                        
        # Norm 3 (1,1,1)
        for i in range(d):
            for j in range(i+1, d):
                for k in range(j+1, d):
                    for s1 in [-1, 1]:
                        for s2 in [-1, 1]:
                            for s3 in [-1, 1]:
                                v = np.zeros(d, dtype=np.int64)
                                v[i] = s1
                                v[j] = s2
                                v[k] = s3
                                add(v)
                                
        # Norm 4
        # Type 1: (2, 0, ...) -> 22
        for i in range(d):
            for s in [-1, 1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = s * 2
                add(v)
        # Type 2: (1, 1, 0, ...) -> 11 choose 2 * 4 = 176
        for i in range(d):
            for j in range(i+1, d):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        add(v)
        # Type 3: (1, 1, 1, 1, 0, ...) -> 11 choose 4 * 16 = 5280
        for i in range(d):
            for j in range(i+1, d):
                for k in range(j+1, d):
                    for l in range(k+1, d):
                        for s1 in [-1, 1]:
                            for s2 in [-1, 1]:
                                for s3 in [-1, 1]:
                                    for s4 in [-1, 1]:
                                        v = np.zeros(d, dtype=np.int64)
                                        v[i] = s1
                                        v[j] = s2
                                        v[k] = s3
                                        v[l] = s4
                                        add(v)
                                        
        # Norm 5 (2, 1, 0...)
        # 11 choose 2 * 4 * 2 (swap positions of 2 and 1? No, values are fixed)
        # Indices i, j. v[i]=2, v[j]=1 OR v[i]=1, v[j]=2.
        # Signs: 4 combos.
        # Total: 55 * 2 * 4 = 440
        for i in range(d):
            for j in range(i+1, d):
                # Case 1: v[i]=2, v[j]=1
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1 * 2
                        v[j] = s2 * 1
                        add(v)
                # Case 2: v[i]=1, v[j]=2
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1 * 1
                        v[j] = s2 * 2
                        add(v)
                        
        return np.array(pool, dtype=np.int64)

    pool = generate_pool(5)
    pool_norms = np.sum(pool**2, axis=1)
    
    # We want to select a subset of `pool` to add to `points`.
    # Current points: 220 vectors, M=2.
    # We can add vectors from `pool`.
    # Note: `pool` contains the same 220 vectors as `points`. We should skip those.
    # Let's identify indices in pool that are NOT in points.
    
    # Create a set of tuples for points for fast lookup
    point_tuples = {tuple(p) for p in points}
    new_candidates = []
    for p in pool:
        if tuple(p) not in point_tuples:
            new_candidates.append(p)
            
    new_candidates = np.array(new_candidates, dtype=np.int64)
    cand_norms = np.sum(new_candidates**2, axis=1)
    
    # Greedy selection
    # We maintain `points` and `current_M`.
    # For each candidate c:
    #   new_M = max(current_M, ||c||^2)
    #   Check if ||c - p||^2 >= new_M for all p in points.
    #   If yes, add c to points, update current_M = new_M.
    
    # To speed up the check, we can precompute the Gram matrix of `points`?
    # No, `points` changes.
    # But we can compute distances vectorized:
    # dist2 = ||c||^2 + ||points||^2 - 2 * (points @ c)
    
    selected_indices = []
    current_points = points.copy()
    current_M = 2
    current_norms = np.sum(current_points**2, axis=1)
    
    # Sort candidates by norm? Or random?
    # Higher norm candidates increase M, making constraints tighter.
    # Lower norm candidates (like norm 3, 4) are easier to fit if M stays low?
    # But if we add a norm 4 vector, M becomes 4.
    # Then we must ensure all existing points (norm 2) are dist >= 4 from it.
    # dist^2 = 2 + 4 - 2 c.p = 6 - 2 c.p >= 4 => 2 c.p <= 2 => c.p <= 1.
    # This is a strong constraint.
    
    # Let's try adding candidates in order of increasing norm.
    # This allows us to potentially keep M low for longer?
    # No, if we add a norm 3 vector, M becomes 3.
    # Then dist^2 >= 3.
    # 2 + 3 - 2 c.p >= 3 => 2 - 2 c.p >= 0 => c.p <= 1.
    # Same constraint for norm 3 vs norm 2!
    # If we add norm 4 vector, M=4.
    # 2 + 4 - 2 c.p >= 4 => 2 c.p <= 2 => c.p <= 1.
    # Same constraint!
    # So the constraint on the dot product with existing norm-2 points is c.p <= 1 regardless of M (as long as M >= 3).
    
    # What about interactions between new points?
    # If we add c1 (norm 3) and c2 (norm 3).
    # M becomes 3.
    # dist^2(c1, c2) >= 3.
    # 3 + 3 - 2 c1.c2 >= 3 => 3 - 2 c1.c2 >= 0 => c1.c2 <= 1.5 => c1.c2 <= 1.
    
    # So the core constraint is: All pairs must have dot product <= 1?
    # Let's check.
    # If max norm is M.
    # ||u-v||^2 >= M.
    # ||u||^2 + ||v||^2 - 2 u.v >= M.
    # If ||u||^2 = M and ||v||^2 = M, then 2M - 2 u.v >= M => M >= 2 u.v => u.v <= M/2.
    # If M=2, u.v <= 1.
    # If M=4, u.v <= 2.
    # So higher M allows larger dot products between high-norm vectors.
    
    # However, if we mix norms:
    # u has norm M, v has norm m < M.
    # M + m - 2 u.v >= M => m >= 2 u.v => u.v <= m/2.
    # If m=2, u.v <= 1.
    # So high-norm vectors must have small dot product with low-norm vectors.
    
    # This suggests we should try to add many low-norm vectors first, then high-norm?
    # Or just greedy.
    
    # Let's run the greedy loop.
    # We will iterate through candidates multiple times? No, single pass greedy is fast.
    # We can try different orderings.
    
    # Order 1: Increasing norm
    order1 = np.argsort(cand_norms)
    
    # Order 2: Random
    np.random.seed(42)
    order2 = np.random.permutation(len(new_candidates))
    
    best_subset = []
    best_M = 2
    
    def try_greedy(order):
        nonlocal best_subset, best_M
        curr_pts = points.copy()
        curr_M = 2
        curr_norms = np.sum(curr_pts**2, axis=1)
        sel = []
        
        for idx in order:
            if time.time() - start_time > DEADLINE - 30:
                return
            
            c = new_candidates[idx]
            cn = int(cand_norms[idx])
            new_M = max(curr_M, cn)
            
            # Check compatibility
            # dist2 = ||c||^2 + ||p||^2 - 2 c.p
            # We need min(dist2) >= new_M
            # c.p = curr_pts @ c
            dots = curr_pts @ c
            dist2 = cn + curr_norms - 2 * dots
            min_d2 = int(np.min(dist2))
            
            if min_d2 >= new_M:
                sel.append(idx)
                curr_pts = np.vstack([curr_pts, c])
                curr_norms = np.append(curr_norms, cn)
                curr_M = new_M
        
        if len(sel) > len(best_subset):
            best_subset = sel
            best_M = curr_M
            best_points_candidate = np.vstack([points, new_candidates[sel]])
            # Validate to be safe
            if validate(best_points_candidate):
                best_points = best_points_candidate
                best_k = len(best_points)
            else:
                # Fallback to previous best
                pass
                
    # Run greedy with increasing norm
    try_greedy(order1)
    
    # Run greedy with random order
    try_greedy(order2)
    
    # Try a few more random orders if time permits
    for _ in range(10):
        if time.time() - start_time > DEADLINE - 30:
            break
        order = np.random.permutation(len(new_candidates))
        try_greedy(order)
        
    # Final validation
    if not validate(best_points):
        # Revert to initial
        best_points = points
        best_k = len(points)
        
    return best_points

# Run it
# Note: The function is defined. The evaluator calls it.
