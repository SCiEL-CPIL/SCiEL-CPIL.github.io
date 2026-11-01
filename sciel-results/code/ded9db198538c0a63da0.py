import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks
    import itertools
    if q == 3 and n >= 4:
        # Tetracode based construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    
    # Precompute neighbors (balls) for all 6561 words
    # weights[i] is the value of the i-th digit
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # balls[word] is a list of words within distance 1
    balls = []
    for w in range(total):
        ball = [w]
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                ball.append(new_w)
        balls.append(ball)
    
    # Convert balls to numpy arrays for faster indexing
    balls_np = np.array(balls, dtype=np.int32)
    
    best_size = total
    best_set = None
    
    # Run multiple iterations of Simulated Annealing
    # Each iteration starts from a greedy solution
    while time.time() < deadline - 5.0:
        # 1. Greedy Initialization
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        # Standard Greedy Set Cover
        while np.any(uncovered_mask):
            if time.time() > deadline - 5.0:
                break
            # Count how many uncovered words each candidate covers
            # balls_np is (total, 17). uncovered_mask is (total,)
            # We want sum of uncovered_mask[balls_np[i]] for each i
            # This is a matrix-vector product-like operation but with boolean mask
            # Efficient way: 
            # counts = balls_np.T @ uncovered_mask  (if uncovered_mask is 0/1)
            # But balls_np.T is 17 x 6561. uncovered_mask is 6561.
            # Result is 17? No.
            # We need for each codeword c (0..6560), the number of words in balls[c] that are currently uncovered.
            # Let U be the set of uncovered words.
            # count[c] = | balls[c] \cap U |
            
            # Using numpy:
            # mask_uncovered = uncovered_mask.astype(np.int8)
            # counts = balls_np.dot(mask_uncovered) # This works if balls_np is 0/1 matrix? No, balls_np contains indices.
            
            # We need to gather.
            # counts = np.bincount(balls_np[uncovered_mask_indices]... ) No.
            
            # Let's stick to the previous method which was fast enough:
            # counts = uncovered_mask[balls_np].sum(axis=1)
            counts = uncovered_mask[balls_np].sum(axis=1)
            
            max_count = np.max(counts)
            if max_count == 0:
                break # Should not happen if uncovered_mask is true
                
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            cover_count[balls_np[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue # Invalid initial solution
            
        # 2. Simulated Annealing
        current_set = set(selected)
        # cover_count is already correct for current_set
        
        # Objective: Minimize size, keep valid.
        # We allow temporary invalid states to escape minima.
        # Cost = size + M * (number of uncovered)
        M = 1000
        T = 5.0 # Initial temperature
        T_min = 0.01
        cooling_rate = 0.999
        
        current_cost = len(current_set) + M * np.sum(cover_count == 0)
        best_local_size = len(current_set)
        best_local_set = set(current_set)
        best_local_cover = cover_count.copy()
        
        steps = 0
        max_steps = 50000 # Cap steps to ensure time check
        
        while T > T_min and steps < max_steps and time.time() < deadline - 2.0:
            steps += 1
            move_type = random.random()
            
            if move_type < 0.3:
                # Remove a random codeword
                if not current_set:
                    continue
                c = random.choice(list(current_set))
                # Tentative remove
                cover_count[balls_np[c]] -= 1
                new_size = len(current_set) - 1
                new_uncovered = np.sum(cover_count == 0)
                new_cost = new_size + M * new_uncovered
                delta = new_cost - current_cost
                
                if delta < 0 or (random.random() < np.exp(-delta / T)):
                    current_set.remove(c)
                    current_cost = new_cost
                    if new_uncovered == 0 and new_size < best_local_size:
                        best_local_size = new_size
                        best_local_set = set(current_set)
                        best_local_cover = cover_count.copy()
                else:
                    # Revert
                    cover_count[balls_np[c]] += 1
                    
            elif move_type < 0.6:
                # Add a random codeword
                c = np.random.randint(total)
                if c in current_set:
                    continue
                # Tentative add
                cover_count[balls_np[c]] += 1
                new_size = len(current_set) + 1
                new_uncovered = np.sum(cover_count == 0)
                new_cost = new_size + M * new_uncovered
                delta = new_cost - current_cost
                
                if delta < 0 or (random.random() < np.exp(-delta / T)):
                    current_set.add(c)
                    current_cost = new_cost
                    if new_uncovered == 0 and new_size < best_local_size:
                        best_local_size = new_size
                        best_local_set = set(current_set)
                        best_local_cover = cover_count.copy()
                else:
                    # Revert
                    cover_count[balls_np[c]] -= 1
                    
            else:
                # Swap: remove c1, add c2
                if len(current_set) < 2:
                    continue
                c1 = random.choice(list(current_set))
                c2 = np.random.randint(total)
                if c1 == c2:
                    continue
                
                # Tentative swap
                cover_count[balls_np[c1]] -= 1
                cover_count[balls_np[c2]] += 1
                
                new_size = len(current_set) # Size unchanged
                new_uncovered = np.sum(cover_count == 0)
                new_cost = new_size + M * new_uncovered
                delta = new_cost - current_cost
                
                if delta < 0 or (random.random() < np.exp(-delta / T)):
                    current_set.remove(c1)
                    current_set.add(c2)
                    current_cost = new_cost
                    if new_uncovered == 0 and new_size < best_local_size:
                        best_local_size = new_size
                        best_local_set = set(current_set)
                        best_local_cover = cover_count.copy()
                else:
                    # Revert
                    cover_count[balls_np[c1]] += 1
                    cover_count[balls_np[c2]] -= 1
            
            T *= cooling_rate
            
        # After SA, try to clean up (remove redundant)
        # Reconstruct cover_count from best_local_set to be safe? 
        # Or just use best_local_cover if it was valid.
        if best_local_size < best_size:
            best_size = best_local_size
            best_set = list(best_local_set)
            
    # Fallback if no valid set found (should not happen)
    if best_set is None:
        return [list(np.base_repr(w, 3).zfill(n)) for w in range(total)]
        
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
