"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 128}

def _can_add(S_set, S_list, x):
    """Check if adding x to S creates a forbidden triple."""
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            # Triple (a, b, x): need a+b, a+x, b+x all in S∪{x}
            ab = a + b
            ax = a + x
            bx = b + x
            # a+b in S∪{x}: either in S_set or equals x
            if ab != x and ab not in S_set:
                continue
            # a+x in S∪{x}: a+x==x means a==0 (impossible), so must be in S_set
            if ax not in S_set:
                continue
            # b+x in S∪{x}: similarly must be in S_set
            if bx not in S_set:
                continue
            return False
    return True

def _greedy_construct(n, order, deadline):
    """Greedy: add elements in given order if no forbidden triple is created."""
    S_list = []
    S_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(S_set, S_list, x):
            S_list.append(x)
            S_set.add(x)
    return S_list

def _local_improve(S_list, S_set, n, deadline):
    """Try to add each missing element if possible."""
    improved = False
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if _can_add(S_set, S_list, x):
            S_list.append(x)
            S_set.add(x)
            improved = True
    return S_list, S_set, improved

def _try_swap(S_list, S_set, n, deadline):
    """Try removing one element and adding two others to increase size."""
    # This is complex; skip for now and rely on greedy + local improvement
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    n = P['n']
    
    # Seed: all numbers > n/2, giving n/2 elements with no pair sums in range
    best = list(range(n // 2 + 1, n + 1))
    best_set = set(best)
    best_size = len(best)
    
    random.seed(42)
    
    # Strategy 1: Greedy from high to low
    order = list(range(n, 0, -1))
    S = _greedy_construct(n, order, deadline)
    if len(S) > best_size:
        best = S
        best_set = set(S)
        best_size = len(S)
    
    # Strategy 2: Greedy from low to high
    if time.time() < deadline:
        order = list(range(1, n + 1))
        S = _greedy_construct(n, order, deadline)
        if len(S) > best_size:
            best = S
            best_set = set(S)
            best_size = len(S)
    
    # Strategy 3: Multiple random orderings
    for trial in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S = _greedy_construct(n, order, deadline)
        if len(S) > best_size:
            best = S
            best_set = set(S)
            best_size = len(S)
    
    # Strategy 4: Structured - try different threshold ranges
    # Take all numbers from k to n, then try to add smaller ones
    for k in range(1, n + 1, 2):
        if time.time() > deadline:
            break
        S_list = list(range(k, n + 1))
        S_set = set(S_list)
        # Try to add elements below k
        for x in range(k - 1, 0, -1):
            if time.time() > deadline:
                break
            if x not in S_set and _can_add(S_set, S_list, x):
                S_list.append(x)
                S_set.add(x)
        if len(S_list) > best_size:
            best = S_list
            best_set = S_set
            best_size = len(S_list)
    
    # Local improvement on best
    if time.time() < deadline:
        improved_list, improved_set, _ = _local_improve(best, set(best), n, deadline)
        if len(improved_list) > best_size:
            best = improved_list
            best_size = len(best)
    
    # Final verification
    def _is_valid(S):
        S_set = set(S)
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                for k in range(j + 1, len(S)):
                    a, b, c = S[i], S[j], S[k]
                    if (a + b) in S_set and (a + c) in S_set and (b + c) in S_set:
                        return False
        return True
    
    if not _is_valid(best):
        # Fallback to seed
        return list(range(n // 2 + 1, n + 1))
    
    # Ensure types are correct
    result = [int(x) for x in best]
    return result
# EVOLVE-BLOCK-END
