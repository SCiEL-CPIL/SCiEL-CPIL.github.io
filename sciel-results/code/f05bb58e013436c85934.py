"""Optimized 3-AP maximizer with no 4-AP, n=256."""
import time
import random

N = 256
DEADLINE_MARGIN = 5.0  # seconds before the 60s wall-clock limit

def _count_3aps(A):
    """Count distinct 3-term APs (a, a+d, a+2d) with d>0 in set A."""
    A_set = set(A)
    count = 0
    # For each pair (a, a+2d) in A, check if a+d in A
    # More efficient: for each center c and difference d, check c-d and c+d
    for c in A:
        for d in range(1, N + 1):
            if (c - d) in A_set and (c + d) in A_set:
                count += 1
    return count

def _has_4ap(A):
    """Check if set A contains a nonconstant 4-term AP."""
    A_set = set(A)
    for a in A:
        for d in range(1, N + 1):
            if (a + d) in A_set and (a + 2*d) in A_set and (a + 3*d) in A_set:
                return True
    return False

def _add_3ap_gain(x, A_set):
    """Count how many new 3-APs are created by adding x to A_set."""
    gain = 0
    # x as first term: x, x+d, x+2d
    for d in range(1, N + 1):
        if (x + d) in A_set and (x + 2*d) in A_set:
            gain += 1
    # x as middle term: x-d, x, x+d
    for d in range(1, x):
        if (x - d) in A_set and (x + d) in A_set:
            gain += 1
    # x as last term: x-2d, x-d, x
    for d in range(1, (x + 1) // 2):
        if (x - 2*d) in A_set and (x - d) in A_set:
            gain += 1
    return gain

def _creates_4ap(x, A_set):
    """Check if adding x to A_set creates a 4-term AP."""
    # x as first: x, x+d, x+2d, x+3d
    for d in range(1, N + 1):
        if (x+d) in A_set and (x+2*d) in A_set and (x+3*d) in A_set:
            return True
    # x as second: x-d, x, x+d, x+2d
    for d in range(1, x):
        if (x-d) in A_set and (x+d) in A_set and (x+2*d) in A_set:
            return True
    # x as third: x-2d, x-d, x, x+d
    for d in range(1, (x+1)//2):
        if (x-2*d) in A_set and (x-d) in A_set and (x+d) in A_set:
            return True
    # x as fourth: x-3d, x-2d, x-d, x
    for d in range(1, (x+1)//3 + 1):
        if (x-3*d) in A_set and (x-2*d) in A_set and (x-d) in A_set:
            return True
    return False

def _greedy_best_first():
    """Best-first greedy: at each step, add the element with max 3-AP gain that doesn't create a 4-AP."""
    A_set = set()
    candidates = list(range(1, N + 1))
    
    while candidates:
        best_x = None
        best_gain = -1
        for x in candidates:
            if _creates_4ap(x, A_set):
                continue
            gain = _add_3ap_gain(x, A_set)
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x is None:
            break
        A_set.add(best_x)
        candidates.remove(best_x)
    
    return sorted(A_set)

def _greedy_randomized(restart_count=20):
    """Multiple randomized greedy restarts, keep best."""
    best_A = None
    best_count = -1
    rng = random.Random(42)
    
    for r in range(restart_count):
        A_set = set()
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        while candidates:
            # Find all valid candidates and pick best gain
            valid = []
            for x in candidates:
                if _creates_4ap(x, A_set):
                    continue
                gain = _add_3ap_gain(x, A_set)
                valid.append((gain, x))
            if not valid:
                break
            # Pick best gain (with slight randomization for diversity)
            valid.sort(key=lambda t: (-t[0], rng.random()))
            best_gain, best_x = valid[0]
            A_set.add(best_x)
            candidates.remove(best_x)
        
        count = _count_3aps(A_set)
        if count > best_count:
            best_count = count
            best_A = sorted(A_set)
    
    return best_A if best_A else [1]

def _local_search_improve(A, max_iters=500):
    """Try to improve via adding elements or swaps."""
    A_set = set(A)
    current_count = _count_3aps(A_set)
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Try adding any missing element
        for x in range(1, N + 1):
            if x in A_set:
                continue
            if _creates_4ap(x, A_set):
                continue
            new_count = _count_3aps(A_set | {x})
            if new_count > current_count:
                A_set.add(x)
                current_count = new_count
                improved = True
                break
    
    return sorted(A_set)

def construct():
    start_time = time.time()
    deadline = start_time + 60.0 - DEADLINE_MARGIN
    
    # Start with best-first greedy
    best_A = _greedy_best_first()
    best_count = _count_3aps(best_A)
    
    # Try local search improvement
    if time.time() < deadline:
        improved = _local_search_improve(best_A)
        imp_count = _count_3aps(improved)
        if imp_count > best_count:
            best_A = improved
            best_count = imp_count
    
    # Try randomized restarts if time permits
    if time.time() < deadline:
        remaining = deadline - time.time()
        # Estimate time per restart: measure one
        t0 = time.time()
        test_A = _greedy_randomized(restart_count=1)
        t1 = time.time()
        per_restart = t1 - t0
        if per_restart > 0:
            max_restarts = max(1, int(remaining / per_restart) - 1)
            if max_restarts > 0:
                rand_A = _greedy_randomized(restart_count=max_restarts)
                if rand_A:
                    rand_count = _count_3aps(rand_A)
                    if rand_count > best_count:
                        best_A = rand_A
                        best_count = rand_count
    
    # Final local search on best
    if time.time() < deadline:
        improved = _local_search_improve(best_A)
        imp_count = _count_3aps(improved)
        if imp_count > best_count:
            best_A = improved
            best_count = imp_count
    
    # Safety: ensure no 4-AP
    if _has_4ap(best_A):
        # Fallback to simple sequential
        A_set = set()
        for x in range(1, N + 1):
            if not _creates_4ap(x, A_set):
                A_set.add(x)
        best_A = sorted(A_set)
    
    if not best_A:
        best_A = [1]
    
    return best_A
