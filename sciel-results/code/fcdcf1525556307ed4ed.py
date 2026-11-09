import time
import numpy as np
from itertools import product, combinations

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def time_remaining():
        return deadline - time.time()

    # 1. Generate candidate pool of integer vectors with norm^2 <= 4
    # This includes:
    # - Type 0: e_i (11 vectors, norm^2=1)
    # - Type 1: e_i - e_j (110 vectors, norm^2=2)
    # - Type 2: e_i + e_j (110 vectors, norm^2=2)
    # - Type 3: e_i + e_j - e_k (11*10*9 = 990 vectors, norm^2=3)
    # - Type 4: e_i + e_j + e_k (11*10*9/6 * 2? No, just sum of 3 distinct basis vectors: C(11,3)=165, norm^2=3)
    # Wait, let's be precise about "integer vectors".
    # Norm^2 = 1: (1,0,..) -> 11 * 2 signs? No, usually we can just take positive and handle signs or take all.
    # Let's generate all vectors v in Z^11 such that ||v||^2 <= 4.
    # Since we want to maximize count, we should include as many as possible.
    # However, the problem requires "nonzero integer vectors".
    # The constraint is min pairwise distance >= max norm.
    # If we mix norms, the max norm dominates the distance constraint.
    # It is generally better to have a uniform norm if possible, or at least ensure the "max norm" is not too large relative to the packing.
    # For kissing number, we usually normalize to unit sphere.
    # If we use vectors of different norms, the condition is dist(u,v) >= max(||u||, ||v||).
    # This is equivalent to: for all u,v, ||u-v||^2 >= max(||u||^2, ||v||^2).
    #
    # Strategy: Generate all vectors with small norm.
    # To maximize k, we want many vectors.
    # Let's generate all vectors with coordinates in {-1, 0, 1} and norm^2 <= 4.
    # Actually, coordinates can be larger if we want, but small norms allow denser packing.
    # Let's stick to coordinates in {-1, 0, 1} first.
    # Norm^2=1: 11 * 2 = 22 vectors.
    # Norm^2=2: C(11,2) * 2^2 = 55 * 4 = 220 vectors.
    # Norm^2=3: C(11,3) * 2^3 = 165 * 8 = 1320 vectors.
    # Norm^2=4: C(11,4) * 2^4 (all +/-1) + C(11,2) * 2 (one +2, one -2? No, coords in {-1,0,1} means max norm^2=11.
    # If we allow coordinate 2:
    # Norm^2=4: (2,0,..) -> 11*2=22. (1,1,1,1,..) -> C(11,4)*16 = 330*16=5280.
    # Total pool size is manageable.
    
    candidates = []
    
    # Helper to add vector
    def add_vec(v):
        # v is a tuple of ints
        if any(x != 0 for x in v):
            candidates.append(v)

    # Generate all vectors with coords in {-1, 0, 1}
    for bits in range(2**D):
        v = [0]*D
        norm_sq = 0
        for i in range(D):
            if (bits >> i) & 1:
                v[i] = 1
                norm_sq += 1
            # We need to handle signs. This loop only generates positive ones.
            # Let's use a more systematic generation for small sets.
        
    # Systematic generation for coords in {-1, 0, 1}
    # Norm^2 = 1
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D
            v[i] = s
            candidates.append(tuple(v))
            
    # Norm^2 = 2
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0]*D
                    v[i] = si
                    v[j] = sj
                    candidates.append(tuple(v))
                    
    # Norm^2 = 3
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        for sk in [1, -1]:
                            v = [0]*D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(tuple(v))
                            
    # Norm^2 = 4
    # Case 1: One coord is +/-2
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D
            v[i] = 2*s
            candidates.append(tuple(v))
    # Case 2: Four coords are +/-1
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for l in range(k+1, D):
                    for si in [1, -1]:
                        for sj in [1, -1]:
                            for sk in [1, -1]:
                                for sl in [1, -1]:
                                    v = [0]*D
                                    v[i] = si
                                    v[j] = sj
                                    v[k] = sk
                                    v[l] = sl
                                    candidates.append(tuple(v))

    # Convert to numpy array for fast math
    # Remove duplicates (though generation should be unique)
    unique_candidates = np.unique(np.array(candidates, dtype=np.int64), axis=0)
    
    # Calculate norms squared
    norms_sq = np.sum(unique_candidates**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # We want to select a subset S such that for all u,v in S:
    # ||u-v||^2 >= max_norm_sq
    # Note: If we include vectors with different norms, the condition is slightly different.
    # The problem statement says: "minimum pairwise distance is at least the maximum vector norm".
    # Let R = max_{v in S} ||v||.
    # Condition: min_{u!=v in S} ||u-v|| >= R.
    # Squared: min ||u-v||^2 >= R^2.
    
    # To simplify, we can try to find a large set with a FIXED norm first, or just use the global max norm of the selected set.
    # A greedy approach:
    # 1. Sort candidates by norm (ascending) to potentially keep R small? 
    #    Actually, if we pick a vector with large norm, R increases, making it harder for others to be close.
    #    So it's beneficial to prefer vectors with smaller norms if we can pack them densely.
    #    However, the "kissing number" usually implies equal radii. 
    #    The task says "Normalizing each vector to norm 2 certifies a kissing configuration".
    #    This implies we can normalize. 
    #    If we normalize, the distance between normalized vectors u/||u|| and v/||v|| is:
    #    d = ||u/||u|| - v/||v|||| = sqrt(2 - 2 (u dot v)/(||u|| ||v||))
    #    The condition "min pairwise distance >= max vector norm" in the ORIGINAL space is what's checked?
    #    "Construct nonzero integer vectors ... whose minimum pairwise distance is at least the maximum vector norm."
    #    Yes, the check is on the integer vectors.
    #    So if I pick vectors of norm 1 and norm 2, R=2. Distance must be >= 2.
    #    If I pick only vectors of norm 2, R=2. Distance must be >= 2.
    #    It is generally easier to satisfy dist >= R if R is small.
    #    So, let's prioritize smaller norms.
    
    # Sort candidates by norm squared ascending
    sort_idx = np.argsort(norms_sq)
    sorted_candidates = unique_candidates[sort_idx]
    sorted_norms_sq = norms_sq[sort_idx]
    
    # Greedy Maximum Independent Set on Conflict Graph
    # Two vectors u, v conflict if ||u-v||^2 < current_max_norm_sq (which is the max norm of the selected set so far, or potentially the global max of the pool if we assume we'll use the largest).
    # Wait, the constraint is dynamic. If I add a vector with large norm, the constraint for ALL previous vectors tightens.
    # This makes greedy tricky.
    # Heuristic: Iterate through candidates sorted by norm ASCENDING.
    # Maintain a list of selected vectors.
    # When considering a new vector v:
    #   Let R_new = max(R_old, ||v||).
    #   Check if for all u in Selected: ||u-v||^2 >= R_new^2.
    #   Also check if for all u,w in Selected: ||u-w||^2 >= R_new^2. (This is already true if they were selected under R_old <= R_new? No.
    #   If R increases, the required distance increases. Previously valid pairs might become invalid.
    #   This implies we should be very careful about increasing R.
    #   
    # Alternative Strategy: Fix a target norm R^2.
    # Find the largest set of vectors such that min pairwise dist^2 >= R^2 AND all norms <= R^2.
    # Then try different R^2 values.
    # Possible R^2 values in our pool: 1, 2, 3, 4.
    #
    # Let's try R^2 = 4 first (largest norm, allows most candidates).
    # If we find a set S with max norm^2 = 4 and min dist^2 >= 4, it's valid.
    # If we find a set S with max norm^2 = 1 and min dist^2 >= 1, it's valid.
    # Which gives higher k?
    # Usually, higher norm allows more distinct vectors in the pool, but tighter distance constraint.
    # For kissing number, the optimal configuration often has equal norms.
    # Let's assume the optimal set has equal norm.
    # We will run the greedy algorithm for each possible "max norm" constraint.
    
    best_set = []
    best_score = 0
    
    # Possible max norms squared present in pool
    unique_norms_sq = np.unique(sorted_norms_sq)
    
    for target_max_norm_sq in unique_norms_sq:
        if time.time() > deadline:
            break
            
        # Filter candidates with norm^2 <= target_max_norm_sq
        mask = sorted_norms_sq <= target_max_norm_sq
        pool = sorted_candidates[mask]
        pool_norms_sq = sorted_norms_sq[mask]
        
        if len(pool) == 0:
            continue
            
        # Greedy selection
        selected = []
        selected_norms_sq = []
        
        # To speed up, we can pre-compute conflicts? No, pool is large.
        # We iterate through pool.
        # Optimization: If we pick a vector, it conflicts with others close to it.
        # We can use a simple loop.
        
        # Sort pool by norm ascending (already done) then maybe by some other heuristic?
        # Let's just use the sorted order.
        
        current_selected = []
        current_max_norm_sq = 0
        
        for v in pool:
            v_norm_sq = np.dot(v, v)
            if v_norm_sq > target_max_norm_sq:
                break # Pool is sorted, so we can stop? No, pool is filtered, but sorted.
                
            new_max_norm_sq = max(current_max_norm_sq, v_norm_sq)
            
            # Check compatibility with all existing selected vectors
            compatible = True
            for u in current_selected:
                diff = v - u
                dist_sq = np.dot(diff, diff)
                if dist_sq < new_max_norm_sq:
                    compatible = False
                    break
            
            if compatible:
                # Also need to ensure existing pairs are still valid under new_max_norm_sq?
                # If new_max_norm_sq > current_max_norm_sq, then the required distance increased.
                # We must check ALL pairs in current_selected.
                # This is O(N^2) per insertion. Too slow if N is large.
                #
                # Refinement: Only allow adding vectors if their norm does NOT increase the max norm.
                # i.e., Process groups by norm.
                # Start with smallest norm. Add all possible vectors of that norm.
                # Then move to next norm. When moving to a higher norm, the constraint tightens.
                # We might need to REMOVE previously added vectors if they violate the new constraint.
                # This is complex.
                
                # Simpler Heuristic:
                # Assume we pick vectors of the SAME norm.
                # Find the largest clique (independence set) in the graph of vectors with norm^2 = K.
                # Do this for K=1,2,3,4.
                
                # Let's restart the strategy:
                # For each K in [1, 2, 3, 4]:
                #   Get all vectors with norm^2 == K.
                #   Build conflict graph where edge exists if dist^2 < K.
                #   Find Max Independent Set (Greedy).
                #   Keep the best result.
                
                # This is much faster and simpler.
                # The "max norm" of the set will be sqrt(K).
                # The condition is min dist >= sqrt(K) -> min dist^2 >= K.
                # This matches the edge definition.
                
                pass # This block was for the mixed-norm idea, which is dropped.
        
        # Let's implement the Fixed-Norm strategy properly outside this loop.
        pass

    # Fixed-Norm Strategy
    best_vectors = []
    
    for target_norm_sq in [1, 2, 3, 4]:
        if time.time() > deadline:
            break
            
        # Get all vectors with norm^2 == target_norm_sq
        mask = sorted_norms_sq == target_norm_sq
        pool = sorted_candidates[mask]
        
        if len(pool) == 0:
            continue
            
        # Greedy MIS
        selected = []
        # Sort by degree? We don't have the graph yet.
        # Just iterate.
        # To make it deterministic and potentially better, we can sort by "centrality" or just random shuffle?
        # Deterministic is better for reproducibility.
        
        # We can try a few different orderings:
        # 1. Original order
        # 2. Reverse order
        # 3. Random (fixed seed)
        
        # Let's just use the original order first.
        
        for v in pool:
            compatible = True
            for u in selected:
                diff = v - u
                dist_sq = np.dot(diff, diff)
                if dist_sq < target_norm_sq:
                    compatible = False
                    break
            if compatible:
                selected.append(v)
                
        if len(selected) > len(best_vectors):
            best_vectors = selected
            
    # Convert to array
    if len(best_vectors) == 0:
        # Fallback: return one vector
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    result = np.array(best_vectors, dtype=np.int64)
    
    # Final Validation
    if len(result) > 1:
        norms_sq = np.sum(result**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        min_dist_sq = np.inf
        n = len(result)
        for i in range(n):
            diffs = result[i+1:] - result[i]
            dists_sq = np.sum(diffs**2, axis=1)
            if len(dists_sq) > 0:
                min_d = np.min(dists_sq)
                if min_d < min_dist_sq:
                    min_dist_sq = min_d
        
        if min_dist_sq < max_norm_sq:
            # Invalid! Should not happen if logic is correct, but just in case.
            # Fallback to E8+A3 construction from seed?
            # Let's just return the first few vectors that are valid.
            # Or return empty? No, must be nonzero.
            # Let's return the single vector with smallest norm.
            return result[:1]
            
    return result
