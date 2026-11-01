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
    ball_size = 1 + 2 * n
    
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
    
    # Precompute ball membership as a flat set for O(1) lookup
    ball_sets = [set(row.tolist()) for row in balls]
    
    best_size = total
    best_set = None
    
    np.random.seed(42)
    
    while time.time() < deadline - 8.0:
        # 1. Greedy with batch selection
        selected = []
        uncovered = np.ones(total, dtype=bool)
        
        while np.any(uncovered) and time.time() < deadline - 12.0:
            counts = uncovered[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            
            # Try to select multiple non-overlapping candidates in this batch
            batch = []
            batch_uncovered = uncovered.copy()
            for _ in range(5):  # try up to 5 per batch
                if len(batch) == len(candidates):
                    break
                cands = np.where(batch_uncovered[balls[candidates]].sum(axis=1) > 0)[0]
                if len(cands) == 0:
                    break
                c_idx = cands[np.random.randint(len(cands))]
                c = candidates[c_idx]
                batch.append(c)
                batch_uncovered[balls[c]] = False
            
            for c in batch:
                selected.append(c)
                uncovered[balls[c]] = False
        
        if np.any(uncovered):
            continue
            
        selected = np.array(selected, dtype=np.int32)
        
        # 2. Pruning
        cover_count = np.zeros(total, dtype=np.int32)
        cover_count[balls[selected]] += 1
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            mins = np.min(cover_count[balls[selected]], axis=1)
            removable_mask = mins > 1
            
            if not np.any(removable_mask):
                break
                
            remove_idx = selected[removable_mask]
            selected = selected[~removable_mask]
            
            cover_count[balls[remove_idx]] -= 1
            changed = True
            
        # 3. Local Search (Swap) - more aggressive
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            if len(selected) == 0:
                break
                
            c_idx = np.random.randint(len(selected))
            c = selected[c_idx]
            
            # Get words uniquely covered by c
            c_ball = balls[c]
            unique_mask = cover_count[c_ball] == 1
            unique_words = c_ball[unique_mask]
            
            # Find candidate c_prime that covers all unique words
            # Strategy: pick a word from unique_words, find codewords whose ball contains it
            # and check if it covers ALL unique words
            if len(unique_words) == 0:
                # c is fully redundant (shouldn't happen after pruning, but handle it)
                # Just try to remove it
                cover_count[c_ball] -= 1
                selected = np.delete(selected, c_idx)
                # Re-prune
                changed = True
                while changed and time.time() < deadline - 5.0:
                    if len(selected) == 0:
                        break
                    mins = np.min(cover_count[balls[selected]], axis=1)
                    removable_mask = mins > 1
                    if not np.any(removable_mask):
                        break
                    remove_idx = selected[removable_mask]
                    selected = selected[~removable_mask]
                    cover_count[balls[remove_idx]] -= 1
                    changed = True
                continue
            
            # Pick a random unique word and find candidates that cover it
            target_word = unique_words[np.random.randint(len(unique_words))]
            
            # Find all codewords whose ball contains target_word
            # We need an inverse mapping: word -> list of codewords covering it
            # This is expensive to precompute, so use a heuristic:
            # c_prime must be within distance 1 of target_word
            # Compute neighbors of target_word
            tw = target_word
            t_neighbors = set()
            t_neighbors.add(tw)
            for pos in range(n):
                weight = weights[pos]
                digit = (tw // weight) % 3
                for new_digit in range(q):
                    if new_digit != digit:
                        t_neighbors.add(tw + (new_digit - digit) * weight)
            
            # Check each neighbor as candidate
            found_swap = False
            for c_prime in t_neighbors:
                c_prime = int(c_prime)
                if c_prime == c:
                    continue
                if c_prime in set(selected.tolist()):
                    continue
                
                # Check if c_prime covers all unique_words
                cp_set = ball_sets[c_prime]
                valid = all(w in cp_set for w in unique_words)
                if not valid:
                    continue
                
                # Perform swap
                cover_count[c_ball] -= 1
                cover_count[balls[c_prime]] += 1
                selected[c_idx] = c_prime
                found_swap = True
                
                # Re-prune
                changed = True
                while changed and time.time() < deadline - 5.0:
                    if len(selected) == 0:
                        break
                    mins = np.min(cover_count[balls[selected]], axis=1)
                    removable_mask = mins > 1
                    if not np.any(removable_mask):
                        break
                    remove_idx = selected[removable_mask]
                    selected = selected[~removable_mask]
                    cover_count[balls[remove_idx]] -= 1
                    changed = True
                break
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 403:
                break

    if best_set is None:
        best_set = np.arange(total, dtype=np.int32)
        
    result = []
    for w in best_set:
        digits = []
        temp = int(w)
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
