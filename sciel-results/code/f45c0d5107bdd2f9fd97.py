import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline, start_time)

    # Fallbacks
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline, start_time):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n  # 17

    # Precompute weights for digit extraction
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    # Precompute balls: balls[w] contains w and all words at distance 1 from w
    # Use a vectorized approach to build the balls array
    all_indices = np.arange(total, dtype=np.int32)
    
    # For each position and each possible digit change, precompute the offset
    # offset[pos][delta] where delta is the change in digit
    # Actually, let's just build it directly but efficiently
    
    balls = np.zeros((total, ball_size), dtype=np.int32)
    balls[:, 0] = all_indices
    
    # For each position, for each of the 2 other digits, compute the new index
    for pos in range(n):
        weight = weights[pos]
        for delta in [-1, 1]:
            # For each word, compute what the new word would be
            # digit = (w // weight) % 3
            # new_digit = (digit + delta) % 3
            # new_w = w + (new_digit - digit) * weight
            
            # Vectorized:
            digits = (all_indices // weight) % q
            new_digits = (digits + delta + q) % q
            new_indices = all_indices + (new_digits - digits) * weight
            balls[:, 1 + 2 * pos + (1 if delta > 0 else 0)] = new_indices
    
    # Precompute a lookup structure for fast "is word in ball" checks
    # For each codeword c, we need to check if a set of critical words are in balls[c]
    # We can use a boolean mask approach or a set
    # For speed, let's precompute balls as a list of sets for the swap phase
    # But that's memory heavy. Instead, we'll use numpy for the critical check.
    
    # Alternative: for the swap phase, we can precompute for each pair of codewords
    # whether one's ball contains the other's critical words. But that's O(total^2).
    # Instead, we'll do it on the fly with numpy.
    
    best_size = total
    best_set = None
    rng = np.random.default_rng(42)
    
    iteration = 0
    while time.time() < deadline - 8.0:
        iteration += 1
        
        # === PHASE 1: Randomized Greedy Construction ===
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=np.int32)
        
        # Greedy loop with randomization
        max_greedy_iters = 5000
        for _ in range(max_greedy_iters):
            if time.time() > deadline - 10.0:
                break
            if np.max(uncovered) == 0:
                break
            
            # Count how many uncovered words each candidate covers
            coverage = uncovered[balls].sum(axis=1)
            
            if np.max(coverage) == 0:
                break
            
            # Among candidates with max coverage, pick one with random tie-breaking
            max_cov = np.max(coverage)
            candidates = np.where(coverage == max_cov)[0]
            
            # Add small random perturbation for diversity
            scores = coverage.astype(np.float64)
            scores += rng.random(total) * 0.1 * max_cov
            
            best_idx = int(np.argmax(scores))
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered = np.minimum(uncovered, cover_count)
        
        # If we didn't complete the cover, try to fix
        while np.any(uncovered > 0):
            if time.time() > deadline - 10.0:
                break
            uncov_idx = int(np.where(uncovered > 0)[0][0])
            if uncov_idx not in selected:
                selected.append(uncov_idx)
                cover_count[balls[uncov_idx]] += 1
                uncovered = np.minimum(uncovered, cover_count)
        
        if np.any(uncovered > 0):
            continue
        
        # === PHASE 2: Local Search ===
        
        # 2a: Remove all removable codewords
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        
        # 2b: Swap-based improvement
        improved = True
        max_swap_rounds = 50
        for round_num in range(max_swap_rounds):
            if time.time() > deadline - 5.0:
                break
            if not improved:
                break
            improved = False
            
            # Shuffle order for diversity
            order = rng.permutation(len(selected))
            
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                
                c = selected[idx]
                
                # Try to remove
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    del selected[idx]
                    improved = True
                    # Adjust order
                    if idx < len(order):
                        for j in range(idx, len(order) - 1):
                            order[j] = order[j + 1]
                        order = order[:-1]
                    continue
                
                # Try to swap with a randomly chosen other codeword
                # First, identify critical words if we remove c
                cover_count[balls[c]] -= 1
                critical_mask = (cover_count[balls[c]] == 1)
                critical = balls[c][critical_mask]
                
                if len(critical) == 0:
                    # c was removable
                    cover_count[balls[c]] += 1
                    continue
                
                # Try a few random swaps
                swap_found = False
                for _ in range(20):
                    if time.time() > deadline - 5.0:
                        break
                    
                    c_prime = int(all_indices[rng.integers(total)])
                    if c_prime == c:
                        continue
                    
                    # Check if c_prime covers all critical words
                    # Use numpy: check if all critical words are in balls[c_prime]
                    cp_ball = balls[c_prime]
                    # Create a set for fast lookup
                    cp_set = set(cp_ball.tolist())
                    
                    valid = True
                    for w in critical:
                        if int(w) not in cp_set:
                            valid = False
                            break
                    
                    if valid:
                        # Perform the swap
                        cover_count[balls[c_prime]] += 1
                        selected[idx] = c_prime
                        swap_found = True
                        improved = True
                        break
                
                if not swap_found:
                    # Revert the removal of c
                    cover_count[balls[c]] += 1
        
        # 2c: Final cleanup
        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 486:
                break
    
    # Convert to output format
    if best_set is None:
        # Fallback
        return [list(c) for c in itertools.product(range(q), repeat=n)]
    
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
