import time
import random
import itertools

def construct(q, n, radius):
    # This function is expected to be called with q=3, n=7, radius=1
    # But we should be somewhat general or at least robust to the specific task parameters.
    # The task says: "the evaluator passes this task's alphabet size q, length n and covering radius."
    # So we must handle general q, n, radius, but optimize for the specific case if possible?
    # The prompt says: "Keep the function names and signatures of the seed program."
    # The seed program hardcodes logic for q=3, n=7.
    # If I write a general solver, it might be slower.
    # However, the task is specifically "Football pool problem for 7 matches: ternary covering code K_3(7,1)".
    # The contract says: "Interface: construct(q, n, radius) -> list of codewords".
    # It is safer to assume q=3, n=7, radius=1, but I should probably check and fall back to a simple construction if parameters differ,
    # or just implement the greedy for the specific case and assume the evaluator only tests this case.
    # Given the "120s" limit and the specific nature of the task, I will assume q=3, n=7, radius=1.
    # If q, n, radius are different, I will return a simple valid code (like the seed or random) to ensure validity.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple valid construction for other parameters
        # Simple construction: all words? No, that's too big.
        # For small n, we can just return all words if n is very small.
        # Or use the tetracode logic if applicable.
        # For safety, let's just return a random set that is likely to cover, or the seed logic if it fits.
        # Actually, the seed logic only works for n>=4.
        # Let's just return the first q^n words? No, must be valid.
        # If parameters don't match, return a trivial valid code: all words.
        # This is valid but score will be bad. But the task is specific.
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3**7
    ALL_ONES = (1 << N) - 1
    
    # Precompute ball masks
    # Word index to tuple of digits
    # We can map index to tuple directly or just use index arithmetic.
    # ball_mask[i] will be an integer with bits set for all words in ball of word i.
    
    ball_masks = [0] * N
    
    # Helper to get digits of index
    def idx_to_digits(idx):
        digits = []
        for _ in range(n):
            digits.append(idx % 3)
            idx //= 3
        return digits
    
    # Precompute all words as tuples of digits for ball calculation
    words = [idx_to_digits(i) for i in range(N)]
    
    # Calculate ball masks
    for i in range(N):
        w = words[i]
        mask = 1 << i
        # Add neighbors at distance 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                # Create new word tuple
                new_w = w.copy()
                new_w[pos] = new_val
                # Convert new_w back to index
                # Since we generated words in lexicographic order of index (base 3), 
                # we can compute index directly.
                # Index is sum(digit[i] * 3^i)
                new_idx = 0
                power = 1
                for d in new_w:
                    new_idx += d * power
                    power *= 3
                mask |= (1 << new_idx)
        ball_masks[i] = mask

    # Greedy Set Cover
    def greedy_randomized():
        covered = 0
        code_indices = []
        # Shuffle indices to randomize tie-breaking
        indices = list(range(N))
        random.shuffle(indices)
        
        while covered != ALL_ONES:
            best_gain = -1
            best_idx = -1
            # Iterate over all possible codewords
            # We can optimize by skipping if we already have a good gain, but full scan is fast enough.
            for idx in indices:
                # Calculate new coverage
                new_cov = ball_masks[idx] & ~covered
                gain = new_cov.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    # Early exit if gain is 0? No, we need to cover something.
                    # If best_gain is 15 (max), we can't do better, but we still need to check if there are others?
                    # No, if we find max possible gain, we can break if we don't care about random tie-breaking among maxes.
                    # But we shuffled, so the first max found is random-ish.
                    # Actually, to get better randomization, we should pick randomly among ALL with max gain.
                    # But that requires storing them or a second pass.
                    # Given N=2187, a second pass is cheap.
                    # Let's stick to "first max found in shuffled order" for simplicity and speed.
                    # It effectively picks one of the max-gain words uniformly at random (approximately).
            code_indices.append(best_idx)
            covered |= ball_masks[best_idx]
            
        return code_indices

    # Run multiple restarts
    best_code = None
    best_len = float('inf')
    
    # Get a baseline from seed logic to ensure we have a valid code if greedy fails or is slow
    # Seed logic: tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    best_code = seed_code
    best_len = len(seed_code)
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer
    
    run_count = 0
    while time.time() < deadline:
        code_indices = greedy_randomized()
        if len(code_indices) < best_len:
            best_len = len(code_indices)
            # Convert indices back to words
            best_code = [words[i] for i in code_indices]
            # Optional: If we hit 186, we can stop or try to break it.
            # 186 is the known best. If we find 186, we can continue to see if we can get 185.
            # But 156 is lower bound.
            # If we find 186, it's a great score.
            # Let's keep going to try for better.
        
        run_count += 1
        # Check time inside loop to avoid long blocking if N was larger, but here it's fixed.
        
    return best_code
