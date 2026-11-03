import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    total_words = q ** n
    ball_size = 1 + n * (q - 1)
    
    # 1. Precompute the ball matrix: ball_matrix[i, j] = 1 if j is in ball(i)
    # This is a 2187 x 2187 matrix of 0s and 1s.
    # We can build this efficiently.
    ball_matrix = np.zeros((total_words, total_words), dtype=np.uint8)
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
        
    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx
            
    for idx in range(total_words):
        w = idx_to_word(idx)
        ball_matrix[idx, idx] = 1
        for pos in range(n):
            original = w[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                w[pos] = new_val
                ball_matrix[idx, word_to_idx(w)] = 1
            w[pos] = original

    # 2. Greedy Construction (Vectorized)
    def greedy_cover(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(total_words, dtype=np.uint8)
        code = []
        
        while not np.all(covered):
            # Gain for each candidate c is sum(ball_matrix[c] * (1 - covered))
            # Since covered is 0/1, 1-covered is 1 for uncovered, 0 for covered.
            # gain = ball_matrix @ (1 - covered)
            gains = ball_matrix.dot(1 - covered)
            
            # Find max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Random choice among candidates
            chosen = candidates[np.random.randint(len(candidates))]
            
            code.append(chosen)
            covered |= ball_matrix[chosen]
            
            if time.time() > deadline:
                return None
                
        return code

    # 3. Local Search (Vectorized)
    def local_search(code):
        if not code:
            return code
            
        in_code = np.zeros(total_words, dtype=bool)
        for c in code:
            in_code[c] = True
            
        cover_count = ball_matrix[in_code].sum(axis=0)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # --- Phase 1: Remove Redundant Codewords ---
            # A codeword c is redundant if for all w in ball(c), cover_count[w] > 1
            # i.e., min(cover_count[ball(c)]) > 1
            # We can check this for all c in code at once.
            
            if len(code) > 0:
                # cover_counts_for_codewords[i, j] = cover_count[ball(code[i])[j]]
                # We want to know if min(cover_counts_for_codewords[i]) > 1
                # min > 1 is equivalent to min >= 2
                # Let's compute min_cover_for_each_codeword
                # This is a reduction operation.
                
                # To be safe and fast, we can just check one by one in Python loop
                # but we can vectorize the "is redundant" check.
                
                # Let's try a vectorized check:
                # min_cover = np.min(ball_matrix[in_code] * cover_count[None, :], axis=1)
                # But we only care about indices where in_code is True.
                
                # Actually, simpler:
                # For each c in code, check if np.min(cover_count[ball_matrix[c] == 1]) > 1
                # This is still a loop.
                
                # Let's stick to a Python loop for removal but use vectorized logic inside?
                # No, let's try to vectorize the redundancy check.
                
                # min_cover_for_code[i] = min(cover_count[j] for j where ball_matrix[code[i], j] == 1)
                # We can do:
                # masked_cover = np.where(ball_matrix[in_code], cover_count[None, :], np.inf)
                # min_covers = np.min(masked_cover, axis=1)
                # redundant_indices = np.where(min_covers > 1)[0]
                
                masked_cover = np.where(ball_matrix[in_code], cover_count[None, :], np.inf)
                min_covers = np.min(masked_cover, axis=1)
                redundant_indices = np.where(min_covers > 1)[0]
                
                if len(redundant_indices) > 0:
                    # Remove one redundant codeword (or all? Let's do one to be safe with state)
                    # Actually, removing all at once might be risky if they overlap in unique coverage?
                    # No, if min_cover > 1, removing it keeps min_cover >= 1 for those balls.
                    # But removing multiple might drop a shared word to 0?
                    # If word w is in ball(c1) and ball(c2), and cover_count[w] == 2.
                    # If we remove c1, cover_count[w] becomes 1.
                    # If we remove c2, cover_count[w] becomes 0 (if we update sequentially).
                    # So we should remove one by one.
                    
                    # Pick the first one
                    idx_to_remove = redundant_indices[0]
                    c = code[idx_to_remove]
                    
                    # Update state
                    cover_count -= ball_matrix[c]
                    in_code[c] = False
                    code.pop(idx_to_remove)
                    improved = True
                    continue # Restart loop to recalculate min_covers
                
            # --- Phase 2: Swap ---
            # For each codeword c in code:
            #   Find unique words U(c) = {w in ball(c) | cover_count[w] == 1}
            #   If U(c) is not empty, find c' not in code such that U(c) subset of ball(c')
            #   If found, swap c with c'.
            
            # We can vectorize the search for c'
            # For a specific c, we need a c' such that ball_matrix[c'] covers all words in U(c).
            # This is equivalent to: for all w in U(c), ball_matrix[c', w] == 1.
            # i.e., min(ball_matrix[not_in_code, U(c)]) == 1
            
            # Let's iterate through codewords in code.
            # To speed up, we can precompute "unique words" for all codewords?
            # Unique words change as cover_count changes.
            
            # We will loop through code indices.
            # If we find a swap, we apply it and restart.
            
            # Optimization: Only consider codewords that have unique words.
            # A codeword has unique words if min_cover_for_code[i] == 1.
            # We already computed min_covers in Phase 1 (before removal).
            # But cover_count might have changed if we entered Phase 2 after Phase 1 found nothing.
            # If Phase 1 found nothing, min_covers <= 1 for all.
            # So candidates for swap are those with min_covers == 1.
            
            # Recompute min_covers to be sure (since we might have skipped Phase 1 if no removals)
            # Actually, if we didn't remove anything, cover_count is unchanged.
            
            masked_cover = np.where(ball_matrix[in_code], cover_count[None, :], np.inf)
            min_covers = np.min(masked_cover, axis=1)
            swap_candidates = np.where(min_covers == 1)[0]
            
            # Shuffle swap candidates to avoid bias
            np.random.shuffle(swap_candidates)
            
            for idx in swap_candidates:
                if time.time() > deadline:
                    return code
                
                c = code[idx]
                
                # Get unique words for c
                unique_mask = ball_matrix[c] & (cover_count == 1)
                unique_words = np.where(unique_mask)[0]
                
                if len(unique_words) == 0:
                    continue
                    
                # Find c' not in code that covers all unique_words
                # c' must satisfy: ball_matrix[c', unique_words] are all 1.
                # i.e., np.all(ball_matrix[not_in_code, unique_words] == 1, axis=1)
                
                not_in_code = np.where(~in_code)[0]
                if len(not_in_code) == 0:
                    continue
                
                # Check candidates
                # This is a sub-matrix access
                # comp_matrix = ball_matrix[not_in_code, unique_words]
                # valid_c_primes = np.where(np.all(comp_matrix == 1, axis=1))[0]
                
                # To avoid creating a huge submatrix if unique_words is large, 
                # we can do:
                # sum of (ball_matrix[not_in_code, unique_words] == 0) should be 0
                
                # Let's just do the check. 2187 * 15 is small.
                comp = ball_matrix[not_in_code, unique_words]
                valid_indices = np.where(np.all(comp == 1, axis=1))[0]
                
                if len(valid_indices) > 0:
                    # Pick a random valid c'
                    # Prefer one that adds minimal extra coverage? 
                    # For now, random.
                    c_prime_idx = valid_indices[np.random.randint(len(valid_indices))]
                    c_prime = not_in_code[c_prime_idx]
                    
                    # Perform Swap
                    # Remove c
                    cover_count -= ball_matrix[c]
                    in_code[c] = False
                    
                    # Add c_prime
                    cover_count += ball_matrix[c_prime]
                    in_code[c_prime] = True
                    
                    code[idx] = c_prime
                    improved = True
                    break # Restart main while loop
                    
        return code

    # 4. Main Loop
    best_solution = None
    best_size = float('inf')
    
    # Initial Greedy
    initial_code = greedy_cover(0)
    if initial_code:
        # Reduce
        # We can do a quick reduction pass
        in_code = np.zeros(total_words, dtype=bool)
        for c in initial_code:
            in_code[c] = True
        cover_count = ball_matrix[in_code].sum(axis=0)
        
        # Simple reduction loop
        changed = True
        while changed:
            changed = False
            for i in range(len(initial_code) - 1, -1, -1):
                c = initial_code[i]
                if np.all(cover_count[ball_matrix[c] == 1] > 1):
                    cover_count -= ball_matrix[c]
                    in_code[c] = False
                    initial_code.pop(i)
                    changed = True
        
        local_searched_code = local_search(initial_code)
        best_solution = local_searched_code
        best_size = len(local_searched_code)
        
    # Iterate with different seeds
    seed = 1
    while time.time() < deadline:
        code = greedy_cover(seed)
        if code is None:
            break
            
        # Reduce
        in_code = np.zeros(total_words, dtype=bool)
        for c in code:
            in_code[c] = True
        cover_count = ball_matrix[in_code].sum(axis=0)
        
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                if np.all(cover_count[ball_matrix[c] == 1] > 1):
                    cover_count -= ball_matrix[c]
                    in_code[c] = False
                    code.pop(i)
                    changed = True
        
        local_searched_code = local_search(code)
        
        if len(local_searched_code) < best_size:
            best_solution = local_searched_code
            best_size = len(local_searched_code)
            
        seed += 1
        
    if best_solution is None:
        # Fallback
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    result = []
    for idx in best_solution:
        result.append(idx_to_word(idx))
        
    return result

