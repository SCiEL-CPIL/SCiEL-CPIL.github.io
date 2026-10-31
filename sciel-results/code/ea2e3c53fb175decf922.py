import numpy as np
import time
import random
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build the full pool of candidate vectors:
    # Weight-2 vectors (norm^2=2): C(11,2)*4 = 220
    # Weight-4 vectors (norm^2=4): C(11,4)*16 = 330*16 = 5280
    # We'll use both, but the constraint differs by norm.
    # For a mixed set, we need: for all i != j, ||v_i - v_j|| >= max_k ||v_k||
    # If max norm is R, we need ||v_i - v_j||^2 >= R^2 for all pairs.
    # 
    # Key insight: if we restrict to vectors of the SAME norm squared, say n,
    # then R^2 = n, and we need ||v_i - v_j||^2 >= n
    # ||v_i - v_j||^2 = 2n - 2 v_i . v_j >= n  =>  v_i . v_j <= n/2
    #
    # For weight-2 (n=2): dot <= 1. As shown, all 220 satisfy this.
    # For weight-4 (n=4): dot <= 2. This is the constraint.
    #
    # Strategy: Use weight-2 vectors as a base (220 guaranteed), then try to ADD
    # weight-4 vectors that are compatible with all weight-2 vectors AND with each other.
    # A weight-4 vector v (norm^2=4) vs a weight-2 vector w (norm^2=2):
    #   ||v-w||^2 = 4 + 2 - 2 v.w = 6 - 2 v.w
    #   We need this >= max_norm^2. If max norm is 2 (from weight-4), need 6-2v.w >= 4 => v.w <= 1
    #   If max norm is sqrt(2) (only weight-2), need 6-2v.w >= 2 => v.w <= 2
    #
    # So if we include weight-4 vectors, the max norm becomes 2, and weight-2 vectors
    # must have dot <= 1 with weight-4 vectors.
    #
    # Alternative: use ONLY weight-4 vectors. Then max norm^2 = 4, need dot <= 2.
    # This gives a larger pool (5280) and potentially larger independent sets.
    #
    # BEST STRATEGY: Use local search on the weight-4 pool to find a large independent set,
    # then try to add weight-2 vectors that are compatible.
    #
    # Actually, let's think about what maximizes k:
    # - All 220 weight-2 vectors work together.
    # - Can we add weight-4 vectors? A weight-4 vector v must have dot <= 1 with every
    #   weight-2 vector in our set. Since we have all 220, v must have dot <= 1 with all of them.
    #   For weight-2 vector e_i + e_j (signs s_i, s_j), dot with v = s_i*v[i] + s_j*v[j].
    #   We need |v[i]| + |v[j]| <= 1 for all pairs (i,j) where the weight-2 vector has +1 at both.
    #   But we also have weight-2 vectors with -1, so we need |v[i]*s_i + v[j]*s_j| <= 1 for all sign combos.
    #   This means |v[i]| + |v[j]| <= 1 for ALL pairs (i,j). Since v has weight 4, it has 4 nonzero entries.
    #   Pick two nonzero entries i,j: |v[i]| + |v[j]| <= 1. With integer entries, this forces both to be 0 or one is 0.
    #   Contradiction with weight 4. So NO weight-4 vector is compatible with all 220 weight-2 vectors.
    #
    # So we can't mix. We must choose: all weight-2 (220) OR a subset of weight-4.
    # The weight-4 pool has 5280 vectors with constraint dot <= 2.
    # A large independent set in this graph could potentially exceed 220.
    #
    # Let's focus on finding a large independent set among weight-4 vectors using local search.
    
    # Generate all weight-4 vectors in {-1,0,1}^11
    pool = []
    pool_indices = []  # index into pool for each vector
    idx = 0
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(comb, signs):
                v[pos] = s
            pool.append(v)
            pool_indices.append(idx)
            idx += 1
    
    n_pool = len(pool)  # 5280
    pool_arr = np.array(pool, dtype=np.int64)  # shape (5280, 11)
    
    # Precompute dot products (5280 x 5280 matrix)
    # This is 5280^2 = 27.9M entries, each int64 = 223 MB. Might be tight.
    # Let's use int16 or compute in blocks.
    # Actually, dot products of weight-4 vectors range from -4 to 4, so int8 suffices.
    dot = np.zeros((n_pool, n_pool), dtype=np.int8)
    for i in range(n_pool):
        # Vectorized: dot[i] = pool_arr @ pool_arr[i]
        dot[i] = pool_arr @ pool_arr[i]
    
    # Constraint: for i != j, dot[i,j] <= 2. Conflict if dot > 2.
    # Since dot[i,i] = 4, we only care about i != j.
    conflict = (dot > 2)  # True where conflict exists (includes diagonal, but we won't select same twice)
    np.fill_diagonal(conflict, False)
    
    # Local search: start with a random feasible set, then try swaps to grow it.
    # Phase 1: Greedy seed
    # Phase 2: Local search with 1-for-1 and 1-for-0 (addition) moves
    
    best_set = set()
    best_size = 0
    
    # Greedy seed: repeatedly pick the vector with fewest conflicts with current set
    for trial in range(3):
        current_set = set()
        # Randomize order for diversity
        order = list(range(n_pool))
        random.shuffle(order)
        
        # Greedy: pick vectors that don't conflict with current set
        for v_idx in order:
            # Check if v_idx conflicts with any in current_set
            compatible = True
            for c in current_set:
                if conflict[v_idx, c]:
                    compatible = False
                    break
            if compatible:
                current_set.add(v_idx)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
        
        if time.time() > deadline - 10:
            break
    
    # Local search: try to improve by swapping out one vector for one that allows more additions
    # Or simpler: try to add vectors one at a time (greedy fill)
    # Then try 1-for-1 swaps to enable more additions
    
    current_set = set(best_set)
    
    # First, greedy fill: try to add any vector that's compatible
    improved = True
    while improved:
        improved = False
        for v_idx in range(n_pool):
            if v_idx in current_set:
                continue
            if all(not conflict[v_idx, c] for c in current_set):
                current_set.add(v_idx)
                improved = True
                break  # restart scan after addition
    
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = set(current_set)
    
    # Now local search: 1-for-1 swaps
    # Remove one vector, add another that was previously blocked
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        swapped = False
        # Try removing each vector in the set
        for rem in list(current_set):
            if time.time() > deadline - 5:
                break
            # What vectors become available if we remove 'rem'?
            # A vector v not in current_set can be added if it doesn't conflict with (current_set - {rem})
            # It conflicts with current_set only if it conflicts with rem (since it was blocked)
            # Actually, it might have been blocked by multiple vectors. Let's just check.
            
            # Find candidates: vectors not in current_set that don't conflict with current_set \ {rem}
            # This is expensive. Let's be smarter:
            # A vector v was blocked if it conflicts with at least one vector in current_set.
            # If we remove rem, v becomes addable if its ONLY conflict in current_set was rem.
            
            # Precompute: for each v not in current_set, count conflicts with current_set
            # This is O(n_pool * |current_set|) per removal, too slow.
            
            # Alternative: just try random swaps
            if random.random() < 0.5:
                # Try a random swap
                add_candidates = [v for v in range(n_pool) if v not in current_set]
                if not add_candidates:
                    continue
                add = random.choice(add_candidates)
                
                # Check if add is compatible with current_set - {rem}
                compatible = True
                for c in current_set:
                    if c == rem:
                        continue
                    if conflict[add, c]:
                        compatible = False
                        break
                
                if compatible:
                    current_set.discard(rem)
                    current_set.add(add)
                    swapped = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                    break
        
        if not swapped:
            # Try random perturbations: remove 2, add 2
            if len(current_set) >= 2:
                rem1, rem2 = random.sample(list(current_set), 2)
                add_candidates = [v for v in range(n_pool) if v not in current_set]
                if add_candidates:
                    add1 = random.choice(add_candidates)
                    add2 = random.choice([v for v in add_candidates if v != add1])
                    
                    # Check compatibility with current_set - {rem1, rem2}
                    base = current_set - {rem1, rem2}
                    ok1 = all(not conflict[add1, c] for c in base)
                    ok2 = all(not conflict[add2, c] for c in base)
                    ok12 = not conflict[add1, add2]
                    
                    if ok1 and ok2 and ok12:
                        current_set.discard(rem1)
                        current_set.discard(rem2)
                        current_set.add(add1)
                        current_set.add(add2)
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
    
    # Final: combine with all weight-2 vectors if the weight-4 set is smaller
    # Actually, we established we can't mix. So return the better of:
    # - All 220 weight-2 vectors
    # - The weight-4 independent set we found
    
    if best_size >= 220:
        # Return the weight-4 set
        result = pool_arr[list(best_set)]
        return result
    else:
        # Return all weight-2 vectors
        pool2 = []
        for comb in itertools.combinations(range(D), 2):
            for signs in itertools.product([1, -1], repeat=2):
                v = [0] * D
                for idx2, sign in zip(comb, signs):
                    v[idx2] = sign
                pool2.append(v)
        return np.array(pool2, dtype=np.int64)
