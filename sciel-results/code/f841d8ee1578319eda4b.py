import time
import random

def _is_valid_exact(s):
    """
    Exact check using bitsets with early termination.
    """
    n = len(s)
    if n < 2:
        return True
    
    sums = [0] * (n + 1)
    sums[0] = 1
    
    for x in s:
        for k in range(n, 0, -1):
            sums[k] |= (sums[k-1] << x)
            
    # Check for overlaps between different cardinalities
    for k in range(n + 1):
        for j in range(k + 1, n + 1):
            if sums[k] & sums[j]:
                return False
    return True

def _check_separation(s):
    """
    Check if the set s (sorted) satisfies the strong separation condition:
    Max sum of size k < Min sum of size k+1 for all k.
    This is a sufficient condition for validity.
    """
    n = len(s)
    if n < 2:
        return True
    
    # Prefix sums
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + s[i]
        
    # Min sum of size k is prefix[k]
    # Max sum of size k is prefix[n] - prefix[n-k]
    
    for k in range(1, n):
        max_k = prefix[n] - prefix[n - k]
        min_k1 = prefix[k + 1]
        if max_k >= min_k1:
            return False
    return True

def _greedy_construct_strong():
    """
    Greedy construction using the strong separation condition.
    """
    s = []
    s_set = set()
    
    for _ in range(60): # Max possible size
        added = False
        for x in range(1, 257):
            if x in s_set:
                continue
            new_s = sorted(s + [x])
            if _check_separation(new_s):
                s = new_s
                s_set.add(x)
                added = True
                break
        if not added:
            break
            
    return s

def _greedy_construct_exact(start_set=None):
    """
    Greedy construction using exact validity check.
    """
    if start_set is None:
        s = []
    else:
        s = start_set[:]
    s_set = set(s)
    
    improved = True
    while improved:
        improved = False
        for x in range(1, 257):
            if x in s_set:
                continue
            if _is_valid_exact(s + [x]):
                s.append(x)
                s_set.add(x)
                improved = True
                break
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Strategy 1: Strong Greedy (Fast, sufficient condition)
    strong_set = _greedy_construct_strong()
    
    # Strategy 2: Exact Greedy from scratch
    exact_set_1 = _greedy_construct_exact(None)
    
    # Strategy 3: Exact Greedy from Strong Set
    exact_set_2 = _greedy_construct_exact(strong_set)
    
    # Pick the best
    best_set = max([strong_set, exact_set_1, exact_set_2], key=len)
    
    return list(best_set)
