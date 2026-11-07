import time
import random

def _validate(S):
    """Check if S is a valid set: all non-empty subset sums are distinct
    and no two distinct subset sums divide one another."""
    n = len(S)
    if n == 0:
        return True
    # Generate all non-empty subset sums
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask >> i & 1:
                s += S[i]
        if s in sums:
            return False
        sums.add(s)
    
    sorted_sums = sorted(sums)
    m = len(sorted_sums)
    for i in range(m):
        a = sorted_sums[i]
        for j in range(i + 1, m):
            b = sorted_sums[j]
            if b % a == 0:
                return False
            if b > a * 2:  # optimization: if b > 2a, no larger multiple possible within range
                # Actually we need to check all, but we can break early if b > a * max_possible
                # For now just continue
                pass
    return True

def _validate_fast(S):
    """Faster validation using incremental subset sum tracking."""
    n = len(S)
    if n == 0:
        return True
    if n > 20:
        return _validate(S)
    
    sums = set()
    # Build subset sums incrementally
    sums.add(0)
    for val in S:
        new_sums = set()
        for s in sums:
            ns = s + val
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
    sums.discard(0)
    
    sorted_sums = sorted(sums)
    m = len(sorted_sums)
    for i in range(m):
        a = sorted_sums[i]
        for j in range(i + 1, m):
            b = sorted_sums[j]
            if b % a == 0:
                return False
    return True

def _greedy_color_bound(candidates, current_set, max_size):
    """Upper bound on additional elements we can add using greedy coloring."""
    # Simple bound: just len(candidates)
    return len(candidates)

def _branch_and_bound(candidates, current_set, best, deadline, current_size):
    """Branch and bound search for maximum valid set."""
    if current_size >= len(best):
        pass  # continue to try to improve
    if len(candidates) + current_size <= len(best):
        return best
    if time.time() > deadline:
        return best
    
    # Try adding each candidate
    for idx, cand in enumerate(candidates):
        if time.time() > deadline:
            return best
        
        new_set = current_set + [cand]
        if _validate_fast(new_set):
            if len(new_set) > len(best):
                best = new_set[:]
            
            # Recurse with remaining candidates
            remaining = candidates[idx + 1:]
            # Filter remaining that don't conflict with cand
            # Actually we need to check the full set, so just pass remaining
            result = _branch_and_bound(remaining, new_set, best, deadline, current_size + 1)
            if len(result) > len(best):
                best = result
    
    return best

def _try_random_search(deadline, best):
    """Try random searches to find valid sets of various sizes."""
    random.seed(42)
    pool = list(range(100, 257))  # Focus on larger numbers
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        for size in range(len(best) + 1, 15):
            if size > 20:
                break
            S = random.sample(pool, size)
            S.sort()
            if _validate_fast(S):
                if len(S) > len(best):
                    best = S[:]
                break
    
    return best

def _try_heuristic_construct(deadline, best):
    """Try heuristic construction starting from large numbers."""
    # Start with a known good pattern and try to extend
    base_patterns = [
        [128, 200, 232],
        [130, 196, 240],
        [126, 198, 234],
    ]
    
    for base in base_patterns:
        if time.time() > deadline:
            break
        if len(base) <= len(best):
            continue
        if _validate_fast(base):
            best = base[:]
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin
    
    # Start with seed solution
    best = [31, 32, 34]
    
    # Try heuristic constructions
    best = _try_heuristic_construct(deadline, best)
    
    # Try random search
    best = _try_random_search(deadline, best)
    
    # Try branch and bound from best found
    if time.time() < deadline:
        # Use best as starting point, try to add more
        remaining = [x for x in range(100, 257) if x not in best]
        # Sort remaining by some heuristic
        remaining.sort(reverse=True)
        result = _branch_and_bound(remaining, best, best, deadline, len(best))
        if len(result) > len(best):
            best = result
    
    # Final validation
    if not _validate_fast(best):
        # Fallback to seed
        best = [31, 32, 34]
    
    return list(best)
