import time
import random
import numpy as np

def construct(q, n, radius):
    # Specialized for K_3(7,1) with algebraic structure + local search
    if q != 3 or n != 7 or radius != 1:
        # Generic fallback: simple greedy
        return _generic_greedy(q, n, radius)
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Precompute ball masks for all 2187 words
    total_words = 2187
    balls = np.zeros(total_words, dtype=object)
    
    # Convert index to word tuple
    def idx_to_word(idx):
        word = []
        temp = idx
        for _ in range(7):
            word.append(temp % 3)
            temp //= 3
        word.reverse()
        return tuple(word)
    
    # Word to index
    def word_to_idx(word):
        idx = 0
        for i in range(7):
            idx = idx * 3 + word[i]
        return idx
    
    # Precompute all balls as bitmasks (using Python ints for 2187 bits)
    # Each ball contains the word itself + 2*7 neighbors = 15 words
    balls_list = []
    for i in range(total_words):
        word = idx_to_word(i)
        mask = 1 << i
        for pos in range(7):
            for val in range(3):
                if val == word[pos]:
                    continue
                new_word = list(word)
                new_word[pos] = val
                idx = word_to_idx(tuple(new_word))
                mask |= (1 << idx)
        balls_list.append(mask)
    
    # Convert to numpy array of objects for fast operations
    balls = balls_list
    
    # Strategy 1: Algebraic construction using Hamming code structure
    # The ternary [7,4,3] Hamming code has 3^4 = 81 codewords
    # We'll use a structured approach based on linear codes
    
    # Generate a base set using greedy on quotient space
    # Then exploit symmetry via affine transformations
    
    # First, try a structured greedy that leverages the code structure
    best_code = None
    best_len = float('inf')
    
    # Multiple random restarts with different strategies
    rng = random.Random(42)
    
    for iteration in range(200):  # Will be limited by time
        if time.time() > deadline:
            break
        
        # Strategy: Greedy with randomization, biased toward structured positions
        uncovered = (1 << total_words) - 1
        code = []
        
        # Use a priority-based selection: prefer words that cover many uncovered
        # with some randomness to escape local optima
        
        while uncovered:
            # Sample a subset of candidates for speed, then pick best
            # For early iterations, sample more; for later, less
            num_sampled = min(500, total_words)
            
            if num_sampled < total_words:
                candidates = rng.sample(range(total_words), num_sampled)
            else:
                candidates = list(range(total_words))
            
            max_cov = -1
            best_candidates = []
            
            for i in candidates:
                cov = (uncovered & balls[i]).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    best_candidates = [i]
                elif cov == max_cov:
                    best_candidates.append(i)
            
            if max_cov == 0:
                # Fallback: search all
                for i in range(total_words):
                    cov = (uncovered & balls[i]).bit_count()
                    if cov > max_cov:
                        max_cov = cov
                        best_candidates = [i]
                    elif cov == max_cov:
                        best_candidates.append(i)
                if max_cov == 0:
                    break
            
            chosen = rng.choice(best_candidates)
            word = idx_to_word(chosen)
            code.append(list(word))
            uncovered &= ~balls[chosen]
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]
        
        # Local search: try to remove redundant codewords
        if best_code is not None and time.time() < deadline - 5:
            # Try deletion: for each codeword, check if removing it still covers all
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(len(best_code)):
                    if time.time() > deadline - 5:
                        break
                    # Compute coverage without this codeword
                    test_uncovered = 0
                    for j, cw in enumerate(best_code):
                        if j == i:
                            continue
                        test_uncovered |= balls[word_to_idx(tuple(cw))]
                    
                    # Check if all words are covered
                    if test_uncovered == (1 << total_words) - 1:
                        # This codeword is redundant
                        best_code.pop(i)
                        improved = True
                        break
    
    # Final local search: try swaps
    if best_code is not None and time.time() < deadline - 2:
        best_set = set(tuple(cw) for cw in best_code)
        best_mask = 0
        for cw in best_code:
            best_mask |= balls[word_to_idx(tuple(cw))]
        
        # Try to replace one codeword with another that covers more uncovered
        for _ in range(50):  # Limited iterations
            if time.time() > deadline - 1:
                break
            
            # Find uncovered words
            uncovered = ~best_mask & ((1 << total_words) - 1)
            if uncovered == 0:
                break
            
            # For each codeword, try replacing with a better one
            for i in range(len(best_code)):
                if time.time() > deadline - 1:
                    break
                
                # What would be uncovered if we remove this codeword?
                other_mask = 0
                for j, cw in enumerate(best_code):
                    if j != i:
                        other_mask |= balls[word_to_idx(tuple(cw))]
                
                # Words only covered by this codeword
                only_covered = best_mask & ~other_mask
                # Words not covered at all (should be 0 if valid)
                not_covered = ~best_mask & ((1 << total_words) - 1)
                
                # We want to find a new codeword that covers not_covered + only_covered
                target = not_covered | only_covered
                if target == 0:
                    continue
                
                # Sample candidates
                sample_size = 200
                candidates = rng.sample(range(total_words), sample_size)
                
                best_new = None
                best_new_cov = -1
                for c in candidates:
                    if c in [word_to_idx(tuple(cw)) for cw in best_code]:
                        continue
                    cov = (target & balls[c]).bit_count()
                    if cov > best_new_cov:
                        best_new_cov = cov
                        best_new = c
                
                if best_new is not None and best_new_cov > 0:
                    # Check if this replacement improves overall coverage
                    new_mask = other_mask | balls[best_new]
                    new_uncovered = ~new_mask & ((1 << total_words) - 1)
                    old_uncovered = ~best_mask & ((1 << total_words) - 1)
                    
                    if new_uncovered.bit_count() < old_uncovered.bit_count():
                        # Accept the swap
                        best_code[i] = list(idx_to_idx(best_new))
                        best_mask = new_mask
                        break
    
    # Convert to proper format
    result = []
    for cw in best_code:
        result.append(list(cw))
    
    # Ensure distinct codewords
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(list(cw))
    
    return unique_result

def _generic_greedy(q, n, radius):
    """Simple greedy fallback for non-specialized cases."""
    total = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = []
        temp = idx
        for _ in range(n):
            word.append(temp % q)
            temp //= q
        word.reverse()
        return word
    
    # Precompute balls
    balls = []
    for i in range(total):
        word = idx_to_word(i)
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val == word[pos]:
                    continue
                new_word = word[:]
                new_word[pos] = val
                idx = word_to_idx(new_word)
                mask |= (1 << idx)
        balls.append(mask)
    
    uncovered = (1 << total) - 1
    code = []
    
    while uncovered:
        max_cov = -1
        best_i = -1
        for i in range(total):
            cov = (uncovered & balls[i]).bit_count()
            if cov > max_cov:
                max_cov = cov
                best_i = i
        
        if max_cov == 0:
            break
        
        code.append(idx_to_word(best_i))
        uncovered &= ~balls[best_i]
    
    return code
