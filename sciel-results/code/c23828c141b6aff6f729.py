import time
import itertools
import numpy as np
import random

_DEADLINE = 140.0

def _direct_sum(q, n, radius):
    """Always-valid baseline: direct sum construction."""
    if n >= 4 and radius >= 1:
        base = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        return [h + list(tail) for h in base
                for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        return [list(w) for w in itertools.product(range(q), repeat=n)]

def _idx(w, q):
    idx = 0
    for x in w:
        idx = idx * q + x
    return idx

def _build_all_words(q, n):
    """Build all words as a numpy array of shape (total, n)."""
    total = q ** n
    if total <= 10**7:
        return np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    return None

def _compute_balls_vectorized(all_words, q, n, radius):
    """Compute balls for all centers using vectorized operations.
    Returns a list of arrays, each containing indices of words in the ball."""
    total = all_words.shape[0]
    balls = []
    # Precompute single-position changes
    # For each position p and value v (v != c[p]), compute the new index
    # Ball of radius 1: center + all words differing in exactly 1 position
    
    for i in range(total):
        c = all_words[i]
        ball_set = set()
        ball_set.add(i)
        for p in range(n):
            for v in range(q):
                if v == c[p]:
                    continue
                w = list(c)
                w[p] = v
                ball_set.add(_idx(w, q))
        balls.append(np.array(list(ball_set), dtype=np.int32))
    return balls

def _greedy_cover(q, n, radius, all_words, balls, time_budget, seed=0):
    """Greedy set-cover: repeatedly pick the center covering most uncovered words."""
    total = all_words.shape[0]
    rng = random.Random(seed)
    start = time.time()
    
    covered = np.zeros(total, dtype=bool)
    code = []
    
    while not covered.all():
        if time.time() - start > time_budget:
            break
        
        # Compute gain for each candidate
        best_gain = 0
        best_idx = -1
        
        # Sample-based approach for speed when total is large
        if total > 50000:
            # Sample a subset of candidates
            sample_size = min(5000, total)
            candidates = rng.sample(range(total), sample_size)
            for i in candidates:
                b = balls[i]
                gain = int(np.sum(~covered[b]))
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
        else:
            for i in range(total):
                b = balls[i]
                gain = int(np.sum(~covered[b]))
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
        
        if best_idx == -1 or best_gain == 0:
            break
        
        code.append(all_words[best_idx].tolist())
        covered[balls[best_idx]] = True
    
    return code

def _verify_coverage(code, q, n, radius, all_words, balls):
    """Verify that the code covers all words."""
    total = all_words.shape[0]
    covered = np.zeros(total, dtype=bool)
    for c in code:
        idx = _idx(c, q)
        covered[balls[idx]] = True
    return bool(covered.all())

def _local_search(code, q, n, radius, all_words, balls, time_budget, seed=0):
    """Local search to reduce code size."""
    total = all_words.shape[0]
    rng = random.Random(seed)
    start = time.time()
    
    # Build coverage counts
    counts = np.zeros(total, dtype=np.int32)
    for c in code:
        idx = _idx(c, q)
        counts[balls[idx]] += 1
    
    best_code = [list(c) for c in code]
    best_len = len(code)
    current_code = [list(c) for c in code]
    current_counts = counts.copy()
    
    def try_remove(i):
        """Try removing codeword i. Return True if successful."""
        nonlocal current_code, current_counts, best_code, best_len
        c = current_code[i]
        idx = _idx(c, q)
        ball = balls[idx]
        
        # Temporarily remove
        current_counts[ball] -= 1
        if np.all(current_counts >= 1):
            # Can remove
            new_code = current_code[:i] + current_code[i+1:]
            if len(new_code) < best_len:
                best_code = [list(c) for c in new_code]
                best_len = len(new_code)
            current_code = new_code
            return True
        else:
            # Restore
            current_counts[ball] += 1
            return False
    
    def try_swap(i, new_idx):
        """Try swapping codeword i with word at new_idx."""
        nonlocal current_code, current_counts, best_code, best_len
        c_old = current_code[i]
        old_idx = _idx(c_old, q)
        
        new_word = all_words[new_idx].tolist()
        if new_word == c_old:
            return False
        
        old_ball = balls[old_idx]
        new_ball = balls[new_idx]
        
        # Temporarily swap
        current_counts[old_ball] -= 1
        current_counts[new_ball] += 1
        if np.all(current_counts >= 1):
            # Accept swap (same size, may enable future removals)
            current_code[i] = new_word
            return True
        else:
            # Revert
            current_counts[new_ball] -= 1
            current_counts[old_ball] += 1
            return False
    
    def try_replace(i, new_idx):
        """Try replacing codeword i with a new one, allowing size increase."""
        nonlocal current_code, current_counts, best_code, best_len
        c_old = current_code[i]
        old_idx = _idx(c_old, q)
        
        new_word = all_words[new_idx].tolist()
        if new_word == c_old:
            return False
        
        old_ball = balls[old_idx]
        new_ball = balls[new_idx]
        
        # Check if new word is already in code
        for j, c in enumerate(current_code):
            if j != i and _idx(c, q) == new_idx:
                return False
        
        current_counts[old_ball] -= 1
        current_counts[new_ball] += 1
        if np.all(current_counts >= 1):
            current_code[i] = new_word
            return True
        else:
            current_counts[new_ball] -= 1
            current_counts[old_ball] += 1
            return False
    
    # Local search loop
    no_improve = 0
    while time.time() - start < time_budget:
        improved = False
        
        # Phase 1: Try removing each codeword
        for i in range(len(current_code)):
            if time.time() - start > time_budget:
                break
            if try_remove(i):
                improved = True
                no_improve = 0
                break
            no_improve += 1
        
        if improved:
            continue
        
        # Phase 2: Try random swaps
        if len(current_code) > 1:
            for _ in range(50):
                if time.time() - start > time_budget:
                    break
                i = rng.randrange(len(current_code))
                new_idx = rng.randrange(total)
                if try_swap(i, new_idx):
                    improved = True
                    no_improve = 0
                    break
            if improved:
                continue
        
        # Phase 3: Try to find a better replacement for a random codeword
        if len(current_code) > 1:
            for _ in range(20):
                if time.time() - start > time_budget:
                    break
                i = rng.randrange(len(current_code))
                # Try to find a word that covers more uncovered area
                c = current_code[i]
                idx = _idx(c, q)
                ball = balls[idx]
                
                # Find words that are not in the current ball
                # Sample some candidates
                for _ in range(100):
                    new_idx = rng.randrange(total)
                    if new_idx == idx:
                        continue
                    if try_replace(i, new_idx):
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
            if improved:
                continue
        
        # No improvement
        no_improve += 1
        if no_improve > 100:
            break
    
    # Verify best_code
    if _verify_coverage(best_code, q, n, radius, all_words, balls):
        return best_code
    
    # Fallback: verify current_code
    if _verify_coverage(current_code, q, n, radius, all_words, balls):
        return current_code
    
    return best_code

def construct(q, n, radius):
    start = time.time()
    
    # Always-valid baseline
    baseline = _direct_sum(q, n, radius)
    best = baseline
    best_len = len(baseline)
    
    total = q ** n
    if total > 10**7:
        # Too large for our approach, return baseline
        return baseline
    
    # Build all words and balls
    all_words = _build_all_words(q, n)
    if all_words is None:
        return baseline
    
    balls = _compute_balls_vectorized(all_words, q, n, radius)
    
    elapsed = time.time() - start
    remaining = _DEADLINE - elapsed - 5.0
    
    # Try multiple greedy restarts
    num_restarts = max(1, min(5, int(remaining / 10)))
    
    for restart in range(num_restarts):
        if time.time() - start > _DEADLINE - 10:
            break
        
        time_budget = min(15.0, remaining / (num_restarts - restart))
        seed = restart * 1000 + 42
        
        code = _greedy_cover(q, n, radius, all_words, balls, time_budget, seed)
        
        if code and _verify_coverage(code, q, n, radius, all_words, balls):
            if len(code) < best_len:
                best = code
                best_len = len(code)
        
        remaining = _DEADLINE - (time.time() - start) - 5.0
        if remaining < 5:
            break
    
    # Local search polish
    remaining = _DEADLINE - (time.time() - start) - 5.0
    if remaining > 3 and best_len > 1:
        polished = _local_search(best, q, n, radius, all_words, balls, remaining, seed=42)
        if polished and _verify_coverage(polished, q, n, radius, all_words, balls):
            if len(polished) < best_len:
                best = polished
                best_len = len(polished)
        
        # Second polish pass
        remaining = _DEADLINE - (time.time() - start) - 5.0
        if remaining > 3 and best_len > 1:
            polished2 = _local_search(best, q, n, radius, all_words, balls, remaining, seed=123)
            if polished2 and _verify_coverage(polished2, q, n, radius, all_words, balls):
                if len(polished2) < best_len:
                    best = polished2
                    best_len = len(polished2)
    
    # Final safety check
    if time.time() - start > _DEADLINE - 3:
        return baseline
    
    return best
