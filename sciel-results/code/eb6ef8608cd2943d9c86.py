import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 17
    MIN_D = 6
    max_val = 1 << N
    
    # Precompute weights
    weights = np.zeros(max_val, dtype=np.uint8)
    for i in range(1, max_val):
        weights[i] = weights[i >> 1] + (i & 1)
        
    best_code = []
    best_size = 0
    
    # Helper to build a linear code of size 2^k
    def build_linear_code(k):
        basis = []
        current_lin = [0]
        for g in range(1, max_val):
            if len(basis) >= k:
                break
            valid = True
            for c in current_lin:
                if weights[g ^ c] < MIN_D:
                    valid = False
                    break
            if valid:
                basis.append(g)
                old_code = current_lin
                current_lin = []
                for c in old_code:
                    current_lin.append(c)
                    current_lin.append(c ^ g)
        return current_lin

    # Helper for randomized greedy expansion
    def expand_greedy(seed, time_limit):
        current_code = list(seed)
        code_set = set(seed)
        
        # Compute min_dist to seed for all x
        # We only care about x not in seed
        # Vectorized min dist calculation
        seed_arr = np.array(seed, dtype=np.uint32)
        
        # Process in chunks to avoid memory blowup if seed is large, but seed is <= 128 here mostly
        # Actually, seed is small (32, 64, 128). 
        # 131072 * 128 is ~16M ops.
        
        valid_candidates = []
        
        # We can compute min_dist for all x
        # But we only need those with min_dist >= MIN_D
        # And we want to shuffle them.
        
        # To speed up, we can just iterate x and check against seed?
        # 131072 * 128 checks.
        # In Python, this is slow.
        # Use NumPy.
        
        # We'll compute min_dist for all x in chunks
        chunk_size = 10000
        all_valid = []
        
        for i in range(0, max_val, chunk_size):
            chunk_idx = np.arange(i, min(i + chunk_size, max_val), dtype=np.uint32)
            
            # Skip if all in set? Hard to vectorize set check easily.
            # Just compute dist.
            xor_matrix = chunk_idx[:, np.newaxis] ^ seed_arr[np.newaxis, :]
            dist_matrix = weights[xor_matrix]
            min_dists = np.min(dist_matrix, axis=1)
            
            valid_mask = min_dists >= MIN_D
            valid_idx = chunk_idx[valid_mask]
            
            # Filter out those already in code_set
            # This is a Python loop, but valid_idx is smallish?
            # If min_dist >= 6, it's likely not in seed (distance 0).
            # So valid_idx should be disjoint from seed mostly.
            # But to be safe:
            filtered_idx = [x for x in valid_idx if x not in code_set]
            all_valid.extend(filtered_idx)
            
            if time.time() > time_limit:
                break
        
        if not all_valid:
            return current_code
            
        # Shuffle candidates
        random.shuffle(all_valid)
        
        # Greedy Add
        code_arr = np.array(current_code, dtype=np.uint32)
        
        for x in all_valid:
            if time.time() > time_limit:
                break
            
            # Check x against current_code
            # x is int, code_arr is numpy array
            xor_vals = x ^ code_arr
            dists = weights[xor_vals]
            
            if np.min(dists) >= MIN_D:
                current_code.append(x)
                code_set.add(x)
                code_arr = np.array(current_code, dtype=np.uint32)
                
        return current_code

    # Try different k
    # Prioritize smaller k which might allow more nonlinear expansion
    # k=7, 6, 5
    
    for k in [7, 6, 5]:
        if time.time() > deadline - 5:
            break
            
        # Build seed
        seed = build_linear_code(k)
        if not seed:
            continue
            
        # Try to expand
        # Allocate remaining time / 2 for this k? 
        # Let's give it a fixed budget, e.g., 15s for k=7, 15s for k=6, rest for k=5
        # Or just use deadline - 2.
        
        # We can try multiple random restarts if time permits
        # For now, one pass with shuffle.
        
        # To improve, we can try multiple shuffles if time is left.
        # Let's try 2 passes for k=7 and k=6 if fast.
        
        attempts = 3 if k == 7 else 5 if k == 6 else 10
        
        for attempt in range(attempts):
            if time.time() > deadline - 2:
                break
                
            # Re-shuffle for each attempt? 
            # The shuffle is inside expand_greedy.
            # But expand_greedy recomputes min_dists every time.
            # That's expensive.
            # We should compute min_dists once, and then just shuffle and add.
            
            pass # Logic needs restructuring to avoid recomputing min_dists

    # Restructure: 
    # 1. Build seed
    # 2. Compute valid candidates (min_dist >= 6)
    # 3. Loop:
    #    a. Shuffle candidates
    #    b. Greedy add
    #    c. Update best
    
    # Let's implement this properly.

    for k in [7, 6, 5]:
        if time.time() > deadline - 5:
            break
            
        seed = build_linear_code(k)
        if len(seed) < (1 << k):
            continue # Failed to build full linear code? Shouldn't happen.
            
        seed_set = set(seed)
        seed_arr = np.array(seed, dtype=np.uint32)
        
        # Compute valid candidates
        valid_candidates = []
        chunk_size = 10000
        for i in range(0, max_val, chunk_size):
            if time.time() > deadline - 5:
                break
            chunk_idx = np.arange(i, min(i + chunk_size, max_val), dtype=np.uint32)
            xor_matrix = chunk_idx[:, np.newaxis] ^ seed_arr[np.newaxis, :]
            dist_matrix = weights[xor_matrix]
            min_dists = np.min(dist_matrix, axis=1)
            valid_mask = min_dists >= MIN_D
            valid_idx = chunk_idx[valid_mask]
            
            # Filter seed
            filtered_idx = [int(x) for x in valid_idx if x not in seed_set]
            valid_candidates.extend(filtered_idx)
            
        if not valid_candidates:
            continue
            
        # Try multiple greedy expansions
        # How many? Depends on time.
        # Est time per expansion: 
        #   Sorting/Shuffling: O(M log M)
        #   Greedy Add: O(M * S) where S is final size.
        #   M ~ 100k. S ~ 200.
        #   20M ops. ~1s in NumPy/Python mix?
        
        # Let's try 5 restarts for k=7, 10 for k=6, 20 for k=5?
        # But we must check time.
        
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break
                
            random.shuffle(valid_candidates)
            
            current_code = list(seed)
            code_set = seed_set.copy()
            code_arr = seed_arr.copy()
            
            for x in valid_candidates:
                if time.time() > deadline - 2:
                    break
                
                xor_vals = x ^ code_arr
                dists = weights[xor_vals]
                
                if np.min(dists) >= MIN_D:
                    current_code.append(x)
                    code_set.add(x)
                    code_arr = np.array(current_code, dtype=np.uint32)
                    
            if len(current_code) > best_size:
                best_size = len(current_code)
                best_code = current_code
                
    if not best_code:
        # Fallback: just return a small valid code
        # k=5 seed should have worked.
        best_code = build_linear_code(5)
        
    return best_code
