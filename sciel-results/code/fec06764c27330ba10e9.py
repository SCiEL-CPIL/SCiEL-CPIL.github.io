import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases (not expected in this specific task instance but good for robustness)
    import itertools
    if q == 3 and n >= 4:
        # Direct sum with tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
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
    
    best_size = total
    best_set = None
    
    rng = np.random.RandomState(42)
    
    def greedy_construct(seed_offset=0):
        """Randomized greedy set cover with incremental updates."""
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered = np.ones(total, dtype=bool)
        
        while np.any(uncovered):
            # Count how many uncovered words each candidate covers
            counts = uncovered[balls].sum(axis=1)
            
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            
            selected.append(best_idx)
            
            # Incrementally update: only words in ball(best_idx) change
            ball_words = balls[best_idx]
            was_uncovered = uncovered[ball_words]
            cover_count[ball_words] += 1
            # Only words that were uncovered and now have count >= 1 become covered
            newly_covered = ball_words[was_uncovered]
            if len(newly_covered) > 0:
                uncovered[newly_covered] = False
            
            if time.time() > deadline - 8.0:
                return None  # timeout
        
        return selected
    
    def prune_set(selected, cover_count):
        """Remove redundant codewords."""
        changed = True
        while changed and time.time() < deadline - 4.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected
    
    def local_search(selected, cover_count, max_iterations=50):
        """Try to replace codewords to reduce size or maintain coverage with fewer."""
        selected_set = set(selected)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 4.0:
                break
            
            # Try to remove a codeword and see if we can add one that covers more
            improved = False
            
            for c in list(selected):
                if np.min(cover_count[balls[c]]) > 1:
                    # Can remove c
                    cover_count[balls[c]] -= 1
                    selected.remove(c)
                    selected_set.discard(c)
                    improved = True
                    break
            
            if not improved:
                break
        
        return selected, cover_count
    
    # Multiple restarts with different seeds
    num_restarts = 0
    while time.time() < deadline - 10.0:
        num_restarts += 1
        rng_seed = 42 + num_restarts * 1000
        rng = np.random.RandomState(rng_seed)
        
        selected = greedy_construct()
        if selected is None:
            continue
        
        # Compute cover counts for pruning
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        
        # Prune
        selected = prune_set(selected, cover_count)
        
        # Local search: try exchanges
        # For each codeword, try replacing it with another that covers the same or more
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            for i, c in enumerate(selected):
                # Find words that are only covered by c (critical words)
                only_c = (cover_count == 1) & (np.isin(np.arange(total), balls[c]))
                # Actually, let's check which words in ball(c) have cover_count == 1
                ball_words = balls[c]
                critical = ball_words[cover_count[ball_words] == 1]
                
                if len(critical) == 0:
                    # c is redundant, remove it
                    cover_count[balls[c]] -= 1
                    selected.pop(i)
                    improved = True
                    break
                
                # Try to find a replacement that covers all critical words
                # This is expensive, so only do it for a few candidates
                # Skip for now - just do removal-based pruning
            
            if not improved:
                break
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        if best_size <= 486:
            # Keep searching a bit more
            pass
    
    # Final polish: try to find a smaller solution from the best found
    if best_set is not None and time.time() < deadline - 3.0:
        cover_count = np.zeros(total, dtype=np.int32)
        for c in best_set:
            cover_count[balls[c]] += 1
        
        # Aggressive pruning
        for _ in range(5):
            changed = False
            new_best = []
            for c in best_set:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_best.append(c)
            best_set = new_best
            if not changed:
                break
        
        best_size = len(best_set)
    
    # Convert best_set to list of digit lists
    if best_set is None:
        # Fallback: return all words
        result = []
        for w in range(total):
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % 3)
                temp //= 3
            digits.reverse()
            result.append(digits)
        return result
    
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
