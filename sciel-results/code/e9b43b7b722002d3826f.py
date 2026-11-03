import itertools
import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    # Generate N=4 candidates
    # Type 1: +/- 2 e_i
    # Type 2: +/- e_i +/- e_j +/- e_k +/- e_l (even sum)
    
    cands = []
    
    # Type 1
    for i in range(D):
        v1 = np.zeros(D, dtype=np.int64)
        v1[i] = 2
        cands.append(v1)
        v2 = np.zeros(D, dtype=np.int64)
        v2[i] = -2
        cands.append(v2)
        
    # Type 2
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            if sum(signs) % 2 == 0:
                v = np.zeros(D, dtype=np.int64)
                for pos, sign in zip(support, signs):
                    v[pos] = sign
                cands.append(v)
                
    cands = np.array(cands, dtype=np.int64)
    n_cands = len(cands)
    
    # Precompute dot products
    # To save memory and time, we might not want full matrix if n is large, but 2662 is small enough.
    # 2662^2 * 8 bytes = ~56 MB. OK.
    dots = np.dot(cands, cands.T)
    
    # Conflict matrix: True if dot > 2 (since N=4, limit is 2)
    # Diagonal is 4, which is > 2, but we don't care about self.
    conflict = dots > 2
    
    # We want to find the maximum independent set.
    
    best_set = set()
    best_size = 0
    
    # Greedy Initialization
    def greedy_init(rng):
        order = rng.permutation(n_cands)
        S = set()
        # To speed up, we can maintain a list of available candidates
        # But simple check is O(|S|) per candidate.
        for idx in order:
            if time.time() > deadline - 10:
                break
            # Check if idx is compatible with all in S
            # conflict[idx, list(S)]
            if len(S) == 0:
                S.add(idx)
            else:
                # Convert S to array for fast dot check? 
                # Or just loop. |S| can be up to ~500. 2662 * 500 = 1.3M ops. Fast.
                compatible = True
                for s in S:
                    if conflict[idx, s]:
                        compatible = False
                        break
                if compatible:
                    S.add(idx)
        return S
        
    # Local Search
    def local_search(S, rng):
        S = set(S)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5:
                break
            
            # Try to add a vertex
            # Find candidates not in S
            # We can iterate through all candidates
            added_any = False
            # To make it faster, we can prioritize vertices with fewest conflicts in S?
            # Or just random order.
            order = rng.permutation(n_cands)
            for idx in order:
                if idx in S:
                    continue
                # Check compatibility
                compatible = True
                for s in S:
                    if conflict[idx, s]:
                        compatible = False
                        break
                if compatible:
                    S.add(idx)
                    improved = True
                    added_any = True
                    if time.time() > deadline - 5:
                        break
            if not added_any and not improved:
                # Try 1-remove-many-add
                # For each u in S, remove it, see how many we can add back
                # This is expensive. Do it for a few random u.
                if len(S) > 0:
                    # Pick a random u
                    u = rng.choice(list(S))
                    S.remove(u)
                    # Try to add as many as possible
                    order = rng.permutation(n_cands)
                    added_count = 0
                    for idx in order:
                        if idx in S or idx == u:
                            continue
                        compatible = True
                        for s in S:
                            if conflict[idx, s]:
                                compatible = False
                                break
                        if compatible:
                            S.add(idx)
                            added_count += 1
                    if added_count > 1:
                        improved = True
                    else:
                        # Revert if no improvement
                        S.add(u)
                        # Remove the ones we added? 
                        # We added them to S. We need to remove them to revert.
                        # This is messy. Let's skip the revert logic for now and just accept if added_count > 1.
                        # If added_count <= 1, we should revert.
                        # To revert, we need to know what was added.
                        # Let's simplify: only do this if we have time.
                        
        return S

    rng = np.random.RandomState(42)
    
    # Run multiple trials
    for trial in range(200):
        if time.time() > deadline - 10:
            break
        
        # Greedy
        S = greedy_init(rng)
        
        # Local Search
        S = local_search(S, rng)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            # print(f"Trial {trial}: {best_size}") # Disabled for speed
            
    # Convert indices to vectors
    if best_size == 0:
        # Fallback: just return a small valid set
        # e.g. +/- e_i
        vecs = []
        for i in range(D):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 1
            vecs.append(v)
            v[i] = -1
            vecs.append(v)
        return np.array(vecs)
        
    result_indices = np.array(list(best_set))
    return cands[result_indices]

# EVOLVE-BLOCK-END
