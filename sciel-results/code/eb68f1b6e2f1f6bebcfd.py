import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code using a greedy set-cover algorithm with multiple restarts 
    and redundancy removal.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)
    else:
        return _generic_greedy(q, n, radius, start_time, deadline)

def _solve_ternary_7_1(start_time, deadline):
    q = 3
    n = 7
    total_words = q ** n
    
    # Precompute powers of 3 for fast index conversion
    p3 = [1] * n
    for i in range(1, n):
        p3[i] = p3[i - 1] * q

    # Precompute ball_indices: for each word index, list of indices within Hamming distance 1
    ball_indices = []
    for i in range(total_words):
        digits = [0] * n
        tmp = i
        for pos in range(n):
            digits[pos] = tmp % q
            tmp //= q
        
        ball = [i]
        for pos in range(n):
            orig = digits[pos]
            for val in range(q):
                if val != orig:
                    new_i = i + (val - orig) * p3[pos]
                    ball.append(new_i)
        ball_indices.append(ball)
    
    # Convert to numpy arrays for vectorized operations in local search
    # Ball size is always 1 + 7*2 = 15.
    ball_indices_np = np.array(ball_indices, dtype=np.int32)
    
    # Precompute covers as bitmasks for fast greedy
    covers = [0] * total_words
    for i in range(total_words):
        mask = 0
        for j in ball_indices[i]:
            mask |= (1 << j)
        covers[i] = mask

    # Generate the [7,4,3] ternary code
    # x0=a, x1=b, x2=c, x3=d
    # x5 = a + d - c
    # x6 = 2a + 2b + c + d
    # x4 = b + 2c + d
    hamming_code_indices = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    x0 = a
                    x1 = b
                    x2 = c
                    x3 = d
                    x5 = (a + d - c) % 3
                    x6 = (2*a + 2*b + c + d) % 3
                    x4 = (b + 2*c + d) % 3
                    idx = 0
                    for digit in [x0, x1, x2, x3, x4, x5, x6]:
                        idx = idx * 3 + digit
                    hamming_code_indices.append(idx)

    best_code_size = total_words
    best_codewords = None

    num_restarts = 100 # Will be limited by time

    for run_id in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(run_id)
        
        # Two initialization strategies
        if run_id % 2 == 0:
            # Strategy A: Start with Hamming Code
            initial_code = list(hamming_code_indices)
        else:
            # Strategy B: Start Empty
            initial_code = []

        # Greedy Fill
        code = initial_code.copy()
        code_set = set(code)
        
        # Calculate uncovered
        uncovered = (1 << total_words) - 1
        for c in code:
            uncovered &= ~covers[c]
            
        while uncovered:
            if time.time() > deadline - 2.0:
                break
                
            best_idx = -1
            best_count = 0
            # To add randomness, we can shuffle the scan order or just pick random ties
            # For speed, we scan all. 2187 is small.
            # Optimization: We can skip indices already in code_set? 
            # Yes, adding a duplicate doesn't help.
            
            # We want to maximize (uncovered & covers[i]).bit_count()
            # We don't need to check i in code_set if we just want to find a good new word.
            # If the best word is already in the set, its count contribution is 0 to 'uncovered' 
            # because 'uncovered' excludes covered words. 
            # Wait, if i is in code, covers[i] is already subtracted from uncovered.
            # So (uncovered & covers[i]) will be 0 for i in code.
            # So we can just scan all i.
            
            candidates = list(range(total_words))
            rng.shuffle(candidates)
            
            for i in candidates:
                cnt = (uncovered & covers[i]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
                # We don't track all ties to save time, the shuffle handles randomization of the "first best" found
            
            if best_idx == -1 or best_count == 0:
                break # Should not happen if uncovered is not empty and code is not complete
                
            code.append(best_idx)
            code_set.add(best_idx)
            uncovered &= ~covers[best_idx]
            
        # Local Search
        if time.time() > deadline - 2.0:
            continue
            
        # Initialize coverage counts
        cnt = np.zeros(total_words, dtype=np.int32)
        for c in code:
            cnt[ball_indices_np[c]] += 1
            
        best_size = len(code)
        best_code = code.copy()
        
        no_improve = 0
        max_no_improve = 20
        
        while time.time() < deadline - 1.0 and no_improve < max_no_improve:
            changed = False
            
            # Phase 1: Deletion
            # Try to remove random codewords
            indices_to_try = list(range(len(code)))
            rng.shuffle(indices_to_try)
            
            for idx in indices_to_try:
                if idx >= len(code): continue
                c = code[idx]
                # Check if all neighbors are covered by >= 2
                if cnt[ball_indices_np[c]].min() >= 2:
                    # Safe to remove
                    cnt[ball_indices_np[c]] -= 1
                    code.pop(idx)
                    code_set.remove(c)
                    changed = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code.copy()
            
            if not changed:
                no_improve += 1
            else:
                no_improve = 0
            
            # Phase 2: Swap
            if time.time() > deadline - 1.0:
                break
                
            swaps_attempted = 0
            max_swaps = 50
            while swaps_attempted < max_swaps and time.time() < deadline - 1.0:
                swaps_attempted += 1
                if not code:
                    break
                
                # Pick a random codeword to remove
                c = rng.choice(code)
                
                # Find words that would become uncovered
                # U = {w in ball(c) | cnt[w] == 1}
                ball_c = ball_indices_np[c]
                U = ball_c[cnt[ball_c] == 1]
                
                if len(U) == 0:
                    # c is redundant
                    cnt[ball_c] -= 1
                    code.remove(c)
                    code_set.remove(c)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code.copy()
                    continue
                
                # Find candidates c' that cover all words in U
                # c' must be in ball(w) for all w in U.
                # Intersection of balls of U.
                
                # Get set of candidates from first word in U
                candidates = set(ball_indices_np[U[0]].tolist())
                
                for w in U[1:]:
                    candidates &= set(ball_indices_np[w].tolist())
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                    
                # Filter out candidates already in code
                valid_candidates = [c_prime for c_prime in candidates if c_prime not in code_set]
                
                if not valid_candidates:
                    continue
                    
                # Pick a random valid candidate
                c_prime = rng.choice(valid_candidates)
                
                # Perform swap
                # Remove c
                cnt[ball_indices_np[c]] -= 1
                code.remove(c)
                code_set.remove(c)
                
                # Add c_prime
                cnt[ball_indices_np[c_prime]] += 1
                code.append(c_prime)
                code_set.add(c_prime)
                
                # Note: We don't immediately check if this swap is "good" in terms of size (it's same size)
                # but it might enable future deletions.
                # We accept the swap to explore the landscape.
                
            if len(code) < best_size:
                best_size = len(code)
                best_code = code.copy()

        # Update global best
        if len(best_code) < best_code_size:
            best_code_size = len(best_code)
            best_codewords = best_code.copy()
            
            # Early exit if we reach the known lower bound or very close
            if best_code_size <= 156:
                break

    if best_codewords is None:
        # Fallback
        return [list(all_codewords[i]) for i in range(total_words)] # This will be invalid if not covering, but should not happen

    # Convert indices to lists of digits
    result = []
    for idx in best_codewords:
        digits = [0] * n
        tmp = idx
        for pos in range(n):
            digits[pos] = tmp % q
            tmp //= q
        result.append(digits)
        
    return result

def _generic_greedy(q, n, radius, start_time, deadline):
    total_words = q ** n
    all_codewords = list(itertools.product(range(q), repeat=n))
    
    covers = [set() for _ in range(total_words)]
    for i, c in enumerate(all_codewords):
        c_list = list(c)
        for delta in range(radius + 1):
            for positions in itertools.combinations(range(n), delta):
                for changes in itertools.product(range(1, q), repeat=delta):
                    new_c = list(c_list)
                    for pos, ch in zip(positions, changes):
                        new_c[pos] = (new_c[pos] + ch) % q
                    idx = 0
                    for digit in new_c:
                        idx = idx * q + digit
                    covers[i].add(idx)
                    
    best_code_size = total_words
    best_codewords = None
    
    for run_id in range(50):
        if time.time() > deadline:
            break
        random.seed(run_id)
        uncovered = set(range(total_words))
        chosen_indices = []
        
        while uncovered:
            max_count = -1
            candidates = []
            for c in range(total_words):
                count = len(covers[c] & uncovered)
                if count > max_count:
                    max_count = count
                    candidates = [c]
                elif count == max_count:
                    candidates.append(c)
            if not candidates:
                break
            c_star = random.choice(candidates)
            chosen_indices.append(c_star)
            uncovered -= covers[c_star]
            
        if len(chosen_indices) < best_code_size:
            best_code_size = len(chosen_indices)
            best_codewords = [all_codewords[i] for i in chosen_indices]
            
    if best_codewords is None:
        return [list(all_codewords[i]) for i in range(total_words)]
        
    return [list(c) for c in best_codewords]
