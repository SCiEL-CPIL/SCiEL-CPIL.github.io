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
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    N = q ** n # 6561
    
    # Precompute balls
    # balls[i] is a list of neighbors of i (including i)
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = []
    balls_matrix = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    
    for w in range(N):
        neighbors = [w]
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors.append(new_w)
        balls.append(neighbors)
        balls_matrix[w] = neighbors
        
    best_size = N
    best_set = None
    
    # Helper to convert index to list of digits
    def idx_to_list(idx):
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        return digits[::-1]
        
    # Main Loop
    while time.time() < deadline - 10.0:
        # 1. Greedy Construction
        selected = []
        cover_count = np.zeros(N, dtype=np.int32)
        uncovered = np.ones(N, dtype=bool)
        
        while np.any(uncovered):
            if time.time() > deadline - 10.0:
                break
            
            # Find word covering most uncovered
            # Vectorized: sum uncovered mask over balls
            gains = uncovered[balls_matrix].sum(axis=1)
            
            if np.max(gains) == 0:
                break # Should not happen if logic correct
                
            # Random tie-breaking
            max_gain = np.max(gains)
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            # Update cover_count and uncovered
            # Only update uncovered if count was 0
            ball = balls[best_idx]
            # We need to check which were 0
            # cover_count[ball] += 1
            # uncovered[ball] = False  <-- WRONG if count was already > 0? No, if count > 0, uncovered is already False.
            # So setting to False is safe.
            cover_count[ball] += 1
            uncovered[ball] = False
            
        if np.any(uncovered):
            continue # Invalid
            
        # 2. Prune
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            # Check each codeword
            # Vectorized check:
            # For each c in selected, min(cover_count[balls[c]])
            # If > 1, remove.
            
            # To do this efficiently in a loop (since selected changes size):
            # We can do it in batches or one by one.
            # One by one is O(K * BallSize). K~500, BallSize~17. ~8500 ops. Fast.
            
            i = 0
            while i < len(selected):
                c = selected[i]
                if np.min(cover_count[balls[c]]) > 1:
                    # Remove
                    cover_count[balls[c]] -= 1
                    selected.pop(i)
                    changed = True
                else:
                    i += 1
            
        # 3. Local Search (Swaps)
        # Try to swap codewords to enable further pruning
        # We will run many swap attempts
        # Pre-allocate masks for speed?
        mask_u = np.zeros(N, dtype=bool)
        mask_v = np.zeros(N, dtype=bool)
        
        swap_attempts = 0
        while time.time() < deadline - 5.0:
            swap_attempts += 1
            if not selected:
                break
                
            # Pick random index
            i = np.random.randint(len(selected))
            u = selected[i]
            
            # Pick random replacement
            v = np.random.randint(N)
            if v == u:
                continue
                
            # Check safety
            # Risk = N[u] \ N[v]
            mask_u.fill(False)
            mask_u[balls[u]] = True
            mask_v.fill(False)
            mask_v[balls[v]] = True
            
            risk = mask_u & ~mask_v
            
            safe = True
            if np.any(risk):
                if np.min(cover_count[risk]) <= 1:
                    safe = False
            
            if safe:
                # Perform swap
                cover_count[balls[u]] -= 1
                cover_count[balls[v]] += 1
                selected[i] = v
                
                # Try to prune
                pruned = False
                changed = True
                while changed:
                    changed = False
                    new_selected = []
                    for c in selected:
                        if np.min(cover_count[balls[c]]) > 1:
                            cover_count[balls[c]] -= 1
                            changed = True
                            pruned = True
                        else:
                            new_selected.append(c)
                    selected = new_selected
                
                # If we pruned, we improved. If not, we are at same size but different state.
                # We keep the state to explore further.
                
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()

    if best_set is None:
        # Fallback
        return [idx_to_list(w) for w in range(N)]
        
    return [idx_to_list(w) for w in best_set]
