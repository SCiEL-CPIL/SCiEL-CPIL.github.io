import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_optimized(deadline)
    
    # Fallbacks for other cases
    if q == 3 and n >= 4:
        # Direct sum with tetracode (Radius 2 for n>4, but valid format)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode
        
    # Generic fallback: return all words (invalid for score but valid format)
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_optimized(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Weights for positions
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    # Precompute balls for all words
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    # 1. Greedy Construction with deterministic tie-breaking
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    current_uncovered = np.ones(total, dtype=bool)
    
    # Precompute coverage matrix: cover_matrix[i,j] = 1 if word j is in ball of word i
    # This is the incidence matrix for the set cover problem
    
    while np.any(current_uncovered):
        # Count uncovered neighbors for each word
        counts = np.sum(current_uncovered[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
        
        # Deterministic: pick smallest index among max-count candidates
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[0]
        
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
        
        if time.time() > deadline - 30.0:
            break
    
    # If timed out during greedy, complete it
    while np.any(current_uncovered) and time.time() < deadline - 5.0:
        counts = np.sum(current_uncovered[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[0]
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
    
    # 2. Aggressive Pruning with early termination
    # Use a priority-based approach: try to remove codewords in order of "least critical"
    # A codeword is removable if all its covered words have cover_count > 1
    
    changed = True
    while changed and time.time() < deadline - 15.0:
        changed = False
        new_selected = []
        for c in selected:
            # Check if removing c leaves all its neighbors covered by others
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
    
    # 3. Local Search: try to swap codewords to reduce size
    # For each codeword, try replacing it with an uncovered neighbor that covers more
    if time.time() < deadline - 10.0:
        selected_set = set(selected)
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            new_selected = []
            for c in selected:
                # Try to find a better replacement for c
                # Look at words in ball(c) that are not in selected
                # If replacing c with w allows removing another codeword, do it
                best_swap = None
                for w in balls[c]:
                    if w in selected_set:
                        continue
                    # Check if adding w and removing c still covers everything
                    # This is expensive, so we use a heuristic:
                    # Only consider w if it covers at least as many uncovered words as c
                    # Actually, let's just try simple removal first
                    pass
                
                new_selected.append(c)
            selected = new_selected
    
    # 4. Final verification and conversion
    # Verify coverage
    final_cover = np.zeros(total, dtype=np.int32)
    for c in selected:
        final_cover[balls[c]] += 1
    
    if np.any(final_cover == 0):
        # If invalid, add missing words
        missing = np.where(final_cover == 0)[0]
        for m in missing:
            selected.append(m)
            final_cover[balls[m]] += 1
    
    # Convert to digit lists
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
