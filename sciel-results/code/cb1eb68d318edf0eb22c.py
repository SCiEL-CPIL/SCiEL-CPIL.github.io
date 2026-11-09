import time
import random

N = 256

def _validate(S):
    """Return True iff S satisfies: for all a<b<=c in S, a does not divide b+c."""
    L = sorted(S)
    m = len(L)
    for i in range(m):
        a = L[i]
        for j in range(i + 1, m):
            b = L[j]
            for k in range(j, m):
                c = L[k]
                if (b + c) % a == 0:
                    return False
    return True

def _can_add(x, S):
    """
    Check if x can be added to S while maintaining validity.
    S is a set of integers.
    x is an integer not in S.
    """
    # We need to check all triples (a, b, c) in S U {x} that involve x.
    # The condition is: for a < b <= c, a !| (b+c).
    
    # Case 1: x is 'a' (smallest)
    # a=x, b in S, c in S, x < b <= c
    # Check if x | (b+c)
    # Optimization: 
    # If we precomputed nothing, we just iterate.
    # To make it faster, we can iterate b in S where b > x.
    # Then c in S where c >= b.
    # If (b+c) % x == 0, return False.
    
    # Case 2: x is 'b' (middle)
    # a in S, b=x, c in S, a < x <= c
    # Check if a | (x+c)
    
    # Case 3: x is 'c' (largest)
    # a in S, b in S, c=x, a < b <= x
    # Check if a | (b+x)
    
    # Iterate through S to find relevant elements.
    # For efficiency, we can split S into S_less_x, S_eq_x (none, x not in S), S_greater_x.
    # But simple loops are fine for N=256.
    
    S_list = list(S)
    
    # Case 1: x is a
    for b in S_list:
        if b <= x:
            continue
        for c in S_list:
            if c < b:
                continue
            if (b + c) % x == 0:
                return False
                
    # Case 2: x is b
    for a in S_list:
        if a >= x:
            continue
        for c in S_list:
            if c < x:
                continue
            if (x + c) % a == 0:
                return False
                
    # Case 3: x is c
    for b in S_list:
        if b > x:
            continue
        for a in S_list:
            if a >= b:
                continue
            if (b + x) % a == 0:
                return False
                
    return True

def _greedy_descending():
    n = N
    S = set()
    for x in range(n, 0, -1):
        if _can_add(x, S):
            S.add(x)
    return S

def _greedy_ascending():
    n = N
    S = set()
    for x in range(1, n + 1):
        if _can_add(x, S):
            S.add(x)
    return S

def _greedy_random(seed):
    random.seed(seed)
    n = N
    S = set()
    candidates = list(range(1, n + 1))
    random.shuffle(candidates)
    for x in candidates:
        if _can_add(x, S):
            S.add(x)
    return S

def _interval_seed(start_frac):
    n = N
    start = int(n * start_frac) + 1
    if start < 1: start = 1
    if start > n: return set()
    S = set(range(start, n + 1))
    # Filter out invalid ones? No, just return the set. 
    # It might be invalid. We should validate or repair.
    # For initialization, we just want a large set to start from.
    # We will use local search to repair/optimize.
    # But if it's very invalid, local search might get stuck.
    # Better to return a valid subset of the interval.
    valid_S = set()
    for x in range(n, start - 1, -1):
        if _can_add(x, valid_S):
            valid_S.add(x)
    return valid_S

def _local_search_add(S, deadline):
    """Try to add elements to S. Returns (new_S, added_count)."""
    added = 0
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if x in S:
                continue
            if _can_add(x, S):
                S.add(x)
                added += 1
                changed = True
    return S, added

def _local_search_swap(S, deadline):
    """Try to remove one element and add two others. Returns (new_S, improved)."""
    for y in list(S):
        if time.time() > deadline:
            break
        S_temp = S - {y}
        added_count = 0
        for x in range(1, N + 1):
            if x == y or x in S:
                continue
            if _can_add(x, S_temp):
                S_temp.add(x)
                added_count += 1
                if added_count == 2:
                    break
        if added_count == 2:
            return S_temp, True
    return S, False

def _local_search_remove_add(S, deadline):
    """Try to remove one element and add one other, then try to add more."""
    for y in list(S):
        if time.time() > deadline:
            break
        S_temp = S - {y}
        for x in range(1, N + 1):
            if x == y or x in S:
                continue
            if _can_add(x, S_temp):
                S_new = S_temp | {x}
                # Check if we can add more after this swap
                S_new, added = _local_search_add(S_new, deadline)
                if len(S_new) > len(S):
                    return S_new, True
    return S, False

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Generate initial candidates
    candidates = []
    candidates.append(_greedy_descending())
    candidates.append(_greedy_ascending())
    
    # Random restarts
    for i in range(10):
        candidates.append(_greedy_random(i))
        
    # Interval seeds with various fractions
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        candidates.append(_interval_seed(frac))
    
    # Pick the best valid candidate
    best_set = set()
    best_size = 0
    for S in candidates:
        if _validate(S):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
    
    if not best_set:
        best_set = {N}
        best_size = 1
        
    # Local Search: repeatedly try to add elements
    while time.time() < deadline:
        best_set, added = _local_search_add(best_set, deadline)
        if added == 0:
            # Try swaps: remove 1, add 2
            new_set, improved = _local_search_swap(best_set, deadline)
            if improved:
                best_set = new_set
                best_size = len(best_set)
                continue
            # Try remove-add to escape local minima
            new_set, improved = _local_search_remove_add(best_set, deadline)
            if improved:
                best_set = new_set
                best_size = len(best_set)
                continue
            # Stuck, break
            break
            
    return sorted(best_set)

if __name__ == "__main__":
    _s = construct()
    assert _validate(set(_s)), "constructed set is invalid"
    print("size", len(_s), "first", _s[:5], "last", _s[-5:])
