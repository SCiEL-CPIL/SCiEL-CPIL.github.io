"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _max_sidon_size_fast(s):
    """Fast backtracking to find max Sidon subset size."""
    s = sorted(s)
    n = len(s)
    best = 0
    
    def backtrack(idx, current_subset, used_sums):
        nonlocal best
        # Pruning
        if len(current_subset) + (n - idx) <= best:
            return
        
        if idx == n:
            if len(current_subset) > best:
                best = len(current_subset)
            return
            
        # Option 1: Exclude s[idx]
        backtrack(idx + 1, current_subset, used_sums)
        
        # Option 2: Include s[idx]
        can_add = True
        new_sums = []
        for j in current_subset:
            sm = s[idx] + s[j]
            if sm in used_sums:
                can_add = False
                break
            new_sums.append(sm)
            
        if can_add:
            sm_diag = 2 * s[idx]
            if sm_diag in used_sums:
                can_add = False
            else:
                new_sums.append(sm_diag)
                
        if can_add:
            new_used_sums = set(used_sums)
            new_used_sums.update(new_sums)
            backtrack(idx + 1, current_subset + [idx], new_used_sums)
            
    backtrack(0, [], set())
    return best

def _try_candidate(cands):
    """Return (valid, max_size) for a candidate list of 12 distinct ints in [0,128]."""
    if len(cands) != 12:
        return False, 99
    if len(set(cands)) != 12:
        return False, 99
    if any(x < 0 or x > 128 for x in cands):
        return False, 99
    return True, _max_sidon_size_fast(cands)

def _build_candidates():
    """Generate candidate ambient sets designed to have small max Sidon subset."""
    candidates = []
    
    # Base 1: geometric + small steps, linear map
    base1 = [0, 1, 2, 4, 8, 16, 3, 5, 6, 10, 12, 20]
    for a in range(1, 129):
        for b in range(129):
            c = [(x * a + b) % 129 for x in base1]
            if len(set(c)) == 12:
                candidates.append(c)
    
    # Base 2: small cluster
    base2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for a in range(1, 129):
        for b in range(129):
            c = [(x * a + b) % 129 for x in base2]
            if len(set(c)) == 12:
                candidates.append(c)
    
    # Base 3: mixed
    base3 = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66]
    for a in range(1, 129):
        for b in range(129):
            c = [(x * a + b) % 129 for x in base3]
            if len(set(c)) == 12:
                candidates.append(c)
    
    # Base 4: near-arithmetic with perturbation
    base4 = [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21]
    for a in range(1, 129):
        for b in range(129):
            c = [(x * a + b) % 129 for x in base4]
            if len(set(c)) == 12:
                candidates.append(c)
    
    # Base 5: two clusters
    base5 = [0, 1, 2, 3, 4, 5, 32, 33, 34, 35, 36, 37]
    for a in range(1, 129):
        for b in range(129):
            c = [(x * a + b) % 129 for x in base5]
            if len(set(c)) == 12:
                candidates.append(c)
    
    # Base 6: quadratic
    base6 = [i*i for i in range(12)]
    for a in range(1, 129):
        for b in range(129):
            c = [(x * a + b) % 129 for x in base6]
            if len(set(c)) == 12:
                candidates.append(c)
    
    # Base 7: Fibonacci-like
    base7 = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    # Deduplicate
    base7 = sorted(set(base7))
    if len(base7) == 12:
        for a in range(1, 129):
            for b in range(129):
                c = [(x * a + b) % 129 for x in base7]
                if len(set(c)) == 12:
                    candidates.append(c)
    
    return candidates

def construct():
    start = time.time()
    deadline = start + 85  # 85s safety margin
    
    # Generate candidates and evaluate
    candidates = _build_candidates()
    
    # Shuffle to try diverse ones first
    import random
    random.seed(42)
    random.shuffle(candidates)
    
    best_size = 99
    best_cand = None
    
    for cand in candidates:
        if time.time() > deadline:
            break
        valid, size = _try_candidate(cand)
        if valid and size < best_size:
            best_size = size
            best_cand = cand
            if best_size <= 4:
                # 4 is a very low bound for 12 elements in [0, 128]
                # If we find 4, it's likely optimal or close.
                # We can break early if we are confident, but let's run a bit longer to see if 3 is possible.
                # Actually, 3 is very hard. Let's break if we find 4 to save time for other candidates?
                # No, we want the MINIMUM. If we find 4, we should look for 3.
                # But 3 is unlikely. Let's just continue if time permits.
                pass
    
    if best_cand is None:
        # Fallback: return a simple set
        return list(range(12))
    
    return best_cand
# EVOLVE-BLOCK-END
