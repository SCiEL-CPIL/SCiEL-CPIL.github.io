import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n
    
    # Precompute balls
    weights = [3 ** (n - 1 - i) for i in range(n)]
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
    
    # Initial Greedy Solution
    selected_mask = np.zeros(total, dtype=bool)
    cover_count = np.zeros(total, dtype=np.int32)
    
    # Simple Greedy to get a starting point
    uncovered_mask = np.ones(total, dtype=bool)
    while np.any(uncovered_mask):
        counts = np.sum(uncovered_mask[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0: break
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[np.random.randint(len(candidates))]
        selected_mask[best_idx] = True
        cover_count[balls[best_idx]] += 1
        uncovered_mask[balls[best_idx]] = False
        
    # Initial Pruning
    changed = True
    while changed:
        changed = False
        for i in range(total):
            if selected_mask[i]:
                if np.min(cover_count[balls[i]]) > 1:
                    selected_mask[i] = False
                    cover_count[balls[i]] -= 1
                    changed = True
                    
    current_size = np.sum(selected_mask)
    current_uncovered = np.sum(cover_count == 0)
    best_size = current_size
    best_mask = selected_mask.copy()
    
    # Simulated Annealing
    T = 50.0
    cooling = 0.9995
    min_T = 0.01
    lambda_penalty = 1000
    
    def get_cost(size, uncovered):
        if uncovered == 0:
            return size
        return size + lambda_penalty * uncovered

    current_cost = get_cost(current_size, current_uncovered)
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 5.0 and T > min_T:
        move_type = rng.integers(3)
        
        if move_type == 0: # Add
            candidates = np.where(~selected_mask)[0]
            if len(candidates) == 0: continue
            c = candidates[rng.integers(len(candidates))]
            
            ball = balls[c]
            was_zero = cover_count[ball] == 0
            cover_count[ball] += 1
            new_uncovered = current_uncovered - np.sum(was_zero)
            new_size = current_size + 1
            
            new_cost = get_cost(new_size, new_uncovered)
            delta = new_cost - current_cost
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                selected_mask[c] = True
                current_size = new_size
                current_uncovered = new_uncovered
                current_cost = new_cost
                if current_uncovered == 0 and current_size < best_size:
                    best_size = current_size
                    best_mask = selected_mask.copy()
            else:
                # Reject: revert cover_count
                cover_count[ball] -= 1
                
        elif move_type == 1: # Remove
            candidates = np.where(selected_mask)[0]
            if len(candidates) == 0: continue
            c = candidates[rng.integers(len(candidates))]
            
            ball = balls[c]
            cover_count[ball] -= 1
            is_zero = cover_count[ball] == 0
            new_uncovered = current_uncovered + np.sum(is_zero)
            new_size = current_size - 1
            
            new_cost = get_cost(new_size, new_uncovered)
            delta = new_cost - current_cost
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                selected_mask[c] = False
                current_size = new_size
                current_uncovered = new_uncovered
                current_cost = new_cost
                if current_uncovered == 0 and current_size < best_size:
                    best_size = current_size
                    best_mask = selected_mask.copy()
            else:
                # Reject: revert cover_count
                cover_count[ball] += 1
                
        else: # Swap
            c_in = np.where(selected_mask)[0]
            c_out = np.where(~selected_mask)[0]
            if len(c_in) == 0 or len(c_out) == 0: continue
            c_rm = c_in[rng.integers(len(c_in))]
            c_add = c_out[rng.integers(len(c_out))]
            
            ball_rm = balls[c_rm]
            ball_add = balls[c_add]
            
            # Apply removal
            cover_count[ball_rm] -= 1
            # Apply addition
            cover_count[ball_add] += 1
            
            new_uncovered = np.sum(cover_count == 0)
            new_size = current_size
            
            new_cost = get_cost(new_size, new_uncovered)
            delta = new_cost - current_cost
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                selected_mask[c_rm] = False
                selected_mask[c_add] = True
                current_size = new_size
                current_uncovered = new_uncovered
                current_cost = new_cost
                if current_uncovered == 0 and current_size < best_size:
                    best_size = current_size
                    best_mask = selected_mask.copy()
            else:
                # Reject: revert
                cover_count[ball_rm] += 1
                cover_count[ball_add] -= 1
                # selected_mask unchanged
                
        T *= cooling
        
    result = []
    for w in np.where(best_mask)[0]:
        digits = []
        temp = int(w)
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
