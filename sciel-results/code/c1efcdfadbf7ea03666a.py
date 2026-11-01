import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n # 17
    
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Precompute balls
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
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1
                
    # Greedy Initialization
    selected = np.zeros(total, dtype=bool)
    cover_count = np.zeros(total, dtype=np.int32)
    uncovered = np.ones(total, dtype=bool)
    
    while np.any(uncovered):
        # Count how many uncovered words each candidate covers
        # candidates = all words
        # counts[c] = sum(uncovered[balls[c]])
        counts = uncovered[balls].sum(axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
        candidates = np.where(counts == max_count)[0]
        # Pick random among best
        best_idx = candidates[np.random.randint(len(candidates))]
        
        selected[best_idx] = True
        cover_count[balls[best_idx]] += 1
        uncovered[balls[best_idx]] = False
        
        if time.time() > deadline - 30.0:
            break
            
    # If greedy didn't finish, force finish or use SA to fix?
    # Assuming greedy finishes quickly for 6561 words.
    
    # Simulated Annealing
    # Objective: Minimize Size + Lambda * Uncovered
    # We want to minimize Size. Uncovered must be 0 for validity.
    # Lambda should be large to penalize uncovered, but allow escape.
    
    lambda_penalty = 10.0
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    
    best_size = np.sum(selected)
    best_selected = selected.copy()
    
    current_size = best_size
    current_uncovered = np.sum(uncovered)
    current_cost = current_size + lambda_penalty * current_uncovered
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 5.0 and T > T_min:
        # Pick a random codeword
        c = rng.integers(total)
        
        if selected[c]:
            # Try to remove
            # Words that will become uncovered:
            # w in balls[c] and cover_count[w] == 1
            c_words = balls[c]
            unique_mask = cover_count[c_words] == 1
            num_uncover = np.sum(unique_mask)
            
            delta_size = -1
            delta_uncovered = num_uncover
            delta_cost = delta_size + lambda_penalty * delta_uncover
            
            accept = False
            if delta_cost < 0:
                accept = True
            else:
                if delta_cost < 100: # Avoid overflow in exp
                    accept = (rng.random() < np.exp(-delta_cost / T))
            
            if accept:
                selected[c] = False
                cover_count[c_words] -= 1
                # Update uncovered status for words that became 0
                newly_uncovered = c_words[unique_mask]
                if len(newly_uncovered) > 0:
                    uncovered[newly_uncovered] = True
                    current_uncovered += len(newly_uncovered)
                
                current_size += delta_size
                current_cost += delta_cost
                
                if current_uncovered == 0 and current_size < best_size:
                    best_size = current_size
                    best_selected = selected.copy()
                    # Optional: restart with lower T? No, just continue.
        
        else:
            # Try to add
            # Words that will become covered:
            # w in balls[c] and cover_count[w] == 0
            c_words = balls[c]
            uncovered_mask = cover_count[c_words] == 0
            num_cover = np.sum(uncovered_mask)
            
            delta_size = 1
            delta_uncovered = -num_cover
            delta_cost = delta_size + lambda_penalty * delta_uncovered
            
            accept = False
            if delta_cost < 0:
                accept = True
            else:
                if delta_cost < 100:
                    accept = (rng.random() < np.exp(-delta_cost / T))
            
            if accept:
                selected[c] = True
                cover_count[c_words] += 1
                newly_covered = c_words[uncovered_mask]
                if len(newly_covered) > 0:
                    uncovered[newly_covered] = False
                    current_uncovered -= len(newly_covered)
                
                current_size += delta_size
                current_cost += delta_cost
                
                if current_uncovered == 0 and current_size < best_size:
                    best_size = current_size
                    best_selected = selected.copy()
        
        T *= cooling
        
        # If we have a valid solution, we can try to cool faster or restart?
        # Just continue cooling.
        
    # Final Pruning: Remove any redundant codewords
    # Iterate until no more can be removed
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        for i in range(total):
            if selected[i]:
                c_words = balls[i]
                if np.all(cover_count[c_words] > 1):
                    # Can remove
                    selected[i] = False
                    cover_count[c_words] -= 1
                    current_size -= 1
                    changed = True
                    if current_size < best_size:
                        best_size = current_size
                        best_selected = selected.copy()
                    
    # Return best found
    result_indices = np.where(best_selected)[0]
    result = []
    for w in result_indices:
        digits = []
        temp = int(w)
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
