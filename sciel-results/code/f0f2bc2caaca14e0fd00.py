import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N = 17
    MIN_D = 6
    TOTAL = 1 << N
    
    # Precompute popcount for all 17-bit values
    # Using a lookup table is fastest
    popcount_lookup = np.array([bin(i).count('1') for i in range(TOTAL)], dtype=np.uint8)
    
    # 1. Generate the known 256-word linear code
    # We can hardcode the generator or use a fast greedy.
    # The seed program finds it quickly. Let's keep that but make it faster.
    
    def generate_linear_code():
        # Use a known basis for [18,9,6] shortened or just greedy.
        # Greedy is fine for 256 words.
        code = [0]
        code_set = {0}
        # Try to add words in increasing order of weight? 
        # Linear codes have specific structure.
        # Let's just use the existing greedy but optimized.
        for g in range(1, TOTAL):
            if time.time() > deadline - 40:
                break
            # Check if g is far from all in code
            valid = True
            for c in code:
                if popcount_lookup[g ^ c] < MIN_D:
                    valid = False
                    break
            if valid:
                code.append(g)
                code_set.add(g)
                # Double the code
                new_words = [c ^ g for c in code[:-1]] # c ^ g for all previous
                # Actually, if we add g, we must add c^g for all c in code.
                # The loop above only checks g. We need to ensure c^g is also valid?
                # In a linear code construction, if g is far from all c, then g^c is far from all c'?
                # dist(g^c, c') = dist(g, c^c'). Since code is linear/closed?
                # This greedy builds a linear code if we always add g and c^g.
                # But checking g against all c is sufficient if we maintain the code as a subspace?
                # No, this greedy just finds a set.
                # To ensure it's linear, we should maintain basis.
                # But we just need a large set.
                # The standard greedy for linear codes:
                # If g is not in span(basis) and dist(g, span) >= d, add to basis.
                # This is harder to check quickly.
                
                # The current code in the prompt does:
                # if valid: basis.append(g); new_code = [c, c^g for c in current_code]
                # This ensures the set is closed under XOR with g.
                
                # Let's implement that properly.
                pass
        
        return code

    # The prompt's generate_linear_code is okay. Let's use it but with the fast popcount.
    def generate_linear_code_fast():
        code = [0]
        for g in range(1, TOTAL):
            if time.time() > deadline - 40:
                break
            valid = True
            for c in code:
                if popcount_lookup[g ^ c] < MIN_D:
                    valid = False
                    break
            if valid:
                # Add g and all c^g
                new_code = []
                for c in code:
                    new_code.append(c)
                    new_code.append(c ^ g)
                code = new_code
                if len(code) >= 256:
                    break
        return code

    # Fast vectorized generation of the [17,8,6] linear code
    def generate_linear_code_vectorized():
        # Generator matrix for [17,8,6] code (shortened [18,9,6] QR code)
        # This is a known valid generator matrix.
        # We need 8 basis vectors.
        # Let's construct it or hardcode the 256 codewords?
        # Hardcoding 256 integers is possible but let's generate from a basis.
        # Basis vectors (17 bits):
        # Using a standard [17,8,6] code basis.
        # Reference: Brouwer's tables often list specific codes.
        # Let's use a greedy vectorized approach to find 8 basis vectors quickly?
        # Or just use the known fact that a [17,8,6] code exists and find one.
        
        # Actually, the previous greedy `generate_linear_code_fast` was O(2^17 * 256) which is ~33M ops.
        # In Python, this might take seconds. Vectorized NumPy is better.
        
        # Let's find 8 basis vectors using vectorized checks.
        basis = []
        code = np.array([0], dtype=np.uint32)
        
        for _ in range(8):
            # Candidates are 1..TOTAL-1
            # We want a word g such that min(dist(g, code)) >= 6
            # And g is not in span(basis)
            
            # To speed up, we can just check random candidates or iterate.
            # Iterating 131072 candidates against 256 codewords is 33M XORs and popcounts.
            # NumPy can do this in one go if we batch it.
            
            # Let's just take the first 10000 candidates not in code
            # Actually, let's just use the previous greedy result if it was fast enough?
            # The previous program scored 8.0, which implies it finished.
            # But we want to be faster to allow more search.
            
            # Let's use a hardcoded basis for the [17,8,6] code to be instant.
            # From literature, one such code has generator matrix rows:
            # ... This is risky to guess.
            
            # Alternative: Use the `generate_linear_code_fast` but optimize it?
            # It's a Python loop.
            
            # Let's try a vectorized greedy for the basis.
            # We maintain `code` as a numpy array.
            # We check a batch of candidates against `code`.
            
            candidates = np.arange(1, TOTAL, dtype=np.uint32)
            # Shuffle to find diverse basis? Or just take first?
            # Taking first valid is fine for linear code existence.
            
            found = False
            # Process in chunks to avoid memory issues? 131072 * 256 is fine.
            # dist = popcount(candidates[:, None] ^ code[None, :])
            # This creates a 131072 x 256 array. ~33M bytes. OK.
            
            # But we need to ensure g is not in span(basis).
            # If we pick g such that dist(g, code) >= 6, and code is the span of previous basis,
            # then g is automatically not in the span (distance to span is min distance to code).
            # So the condition min_dist >= 6 implies g is not in span.
            
            # Let's do this 8 times.
            for i in range(8):
                if time.time() > deadline - 50:
                    break
                # Calculate distances from all candidates to all current codewords
                # code is shape (K, ), candidates is (M, )
                # We want min over K for each candidate.
                
                # Optimization: We only need to check candidates not yet in code.
                # Since code grows, we can just check all and filter.
                
                # Vectorized distance
                # To save memory, we can do this in chunks if K is large, but K<=256.
                # 131072 * 256 = 33,554,432 elements. uint8 is 33MB. OK.
                
                dists = popcount_lookup[candidates[:, None] ^ code[None, :]]
                min_dists = np.min(dists, axis=1)
                
                valid_indices = np.where(min_dists >= MIN_D)[0]
                
                if len(valid_indices) > 0:
                    # Pick the first one? Or random?
                    # First one is deterministic and fast.
                    g = candidates[valid_indices[0]]
                    basis.append(g)
                    # Update code: new_code = [c, c^g for c in code]
                    code = np.concatenate([code, code ^ g])
                    found = True
                else:
                    break
                    
            return code.tolist()

    best_code = generate_linear_code_vectorized()
    best_size = len(best_code)
    
    # Convert to list and set for fast manipulation
    best_set = set(best_code)
    
    # Helper to check if a word can be added
    def can_add(word, code_list, code_set):
        if word in code_set:
            return False
        for c in code_list:
            if popcount_lookup[word ^ c] < MIN_D:
                return False
        return True

    # Vectorized check: returns boolean array of which candidates can be added
    def can_add_vectorized(candidates, code_np, code_set):
        # candidates: np array of uint32
        # code_np: np array of uint32
        # Returns mask of same length as candidates
        
        # Filter out those already in code_set
        # This is hard to vectorize directly with set membership.
        # We can create a boolean array for all TOTAL words, but that's 131k.
        # Or just check in the loop after finding valid distance?
        
        # Calculate distances
        # candidates (M, 1), code_np (1, K)
        dists = popcount_lookup[candidates[:, None] ^ code_np[None, :]]
        min_dists = np.min(dists, axis=1)
        
        valid_mask = min_dists >= MIN_D
        
        # Now check set membership for those that passed distance check
        # This is a Python loop, but only for the few that pass distance.
        final_mask = np.zeros(len(candidates), dtype=bool)
        for i in np.where(valid_mask)[0]:
            w = candidates[i]
            if w not in code_set:
                final_mask[i] = True
                
        return final_mask

    # Greedy Pass
    def greedy_pass(order, initial_code=None):
        if initial_code:
            code = list(initial_code)
            code_set = set(code)
        else:
            code = []
            code_set = set()
            
        code_np = np.array(code, dtype=np.uint32)
        
        # Process in batches
        batch_size = 1000
        for i in range(0, len(order), batch_size):
            if time.time() > deadline - 5:
                break
                
            batch = order[i : i + batch_size]
            batch_np = np.array(batch, dtype=np.uint32)
            
            mask = can_add_vectorized(batch_np, code_np, code_set)
            
            if np.any(mask):
                new_words = batch_np[mask]
                code.extend(new_words.tolist())
                code_set.update(new_words.tolist())
                # Update code_np
                code_np = np.array(code, dtype=np.uint32)
                
        return code, code_set

    # Local Search: Try to swap or add
    # Strategy: 
    # 1. Try to add any word.
    # 2. If stuck, try to swap one word with another to enable addition.
    
    def local_improve(code, code_set, candidates, time_limit):
        improved = True
        code_np = np.array(code, dtype=np.uint32)
        
        while improved and time.time() < time_limit:
            improved = False
            # Try to add
            # Pick a random subset of candidates to check
            # Or just iterate through all?
            # Checking all 131k candidates against 256 code is 33M ops.
            # Vectorized, this is very fast (<0.1s).
            
            all_candidates_np = np.array(candidates, dtype=np.uint32)
            mask = can_add_vectorized(all_candidates_np, code_np, code_set)
            
            if np.any(mask):
                new_words = all_candidates_np[mask]
                code.extend(new_words.tolist())
                code_set.update(new_words.tolist())
                code_np = np.array(code, dtype=np.uint32)
                improved = True
                # Continue loop to see if more can be added
                continue
                
            # If no add, try swap
            # Pick a random word to remove
            if len(code) == 0:
                break
            idx = random.randint(0, len(code)-1)
            removed_word = code[idx]
            code.pop(idx)
            code_set.remove(removed_word)
            code_np = np.array(code, dtype=np.uint32)
            
            # Try to add words again
            mask = can_add_vectorized(all_candidates_np, code_np, code_set)
            
            added_count = 0
            if np.any(mask):
                new_words = all_candidates_np[mask]
                # We removed 1, so if we add >1, we improve.
                # If we add 1, we swap.
                # If we add 0, we revert.
                
                # To maximize size, we should add as many as possible?
                # But they must be compatible with each other?
                # `can_add_vectorized` checks against current code, not new words.
                # So we might add words that are too close to each other.
                
                # We need to be careful.
                # Let's just add the first one that fits?
                # Or do a greedy pass on the new candidates?
                
                # Simple strategy: Add one word that was blocked.
                # The removed word blocked some candidates.
                # We don't know which ones.
                
                # Let's just try to add ANY word.
                if len(new_words) > 0:
                    # Add one
                    w = new_words[0]
                    code.append(w)
                    code_set.add(w)
                    code_np = np.array(code, dtype=np.uint32)
                    added_count = 1
                    
            # If we added nothing, revert
            if added_count == 0:
                code.append(removed_word)
                code_set.add(removed_word)
                code_np = np.array(code, dtype=np.uint32)
            else:
                improved = True
                # If we swapped (1 out, 1 in), size is same.
                # We only care about size improvement for the main loop,
                # but structural change might help future adds.
                # However, if we don't increase size, we might loop forever.
                # Let's only accept swaps if they allow future adds?
                # Too complex.
                
                # Let's revert if size didn't increase?
                # If we swapped, size is same.
                # If we added >1, size increased.
                
                # For now, if we didn't increase size, let's revert to avoid cycling?
                # Or keep it?
                # Keeping it might help.
                
        return code, code_set

    # Main Loop
    all_words_list = list(range(TOTAL))
    
    # Pre-shuffle candidates for local search
    # We want to try words that are "likely" to fit.
    
    while time.time() < deadline - 5:
        # 1. Random Greedy Restart
        random.shuffle(all_words_list)
        # Bias towards middle weights?
        # For now, uniform.
        
        # Start from best code? Or from scratch?
        # Starting from best code and trying to extend is usually better.
        
        # Try extending best code
        candidates = [w for w in all_words_list if w not in best_set]
        random.shuffle(candidates)
        
        # Run greedy on candidates
        new_code, new_set = greedy_pass(candidates, initial_code=best_code)
        
        if len(new_code) > best_size:
            best_code = new_code
            best_set = new_set
            best_size = len(best_code)
            
        # Try local improvement on the current best
        if time.time() < deadline - 10:
            # Use a subset of candidates for speed
            # Pick candidates that are close to the code?
            # Or just random.
            local_candidates = random.sample(all_words_list, min(5000, len(all_words_list)))
            # Filter out those already in best
            local_candidates = [w for w in local_candidates if w not in best_set]
            
            improved_code, improved_set = local_improve(best_code, best_set, local_candidates, deadline - 10)
            
            if len(improved_code) > best_size:
                best_code = improved_code
                best_set = improved_set
                best_size = len(best_code)
                
    return best_code
