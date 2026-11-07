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

def _greedy_construct(max_size=60):
    """
    Greedy construction: try to add the smallest possible element that maintains validity.
    """
    s = []
    s_set = set()
    
    for x in range(1, 257):
        if len(s) >= max_size:
            break
        test_s = s + [x]
        if _is_valid_exact(test_s):
            s.append(x)
            s_set.add(x)
    
    return s

def _try_add(s, s_set, x):
    """Try to add x to s, return True if successful."""
    if x in s_set:
        return False
    if _is_valid_exact(s + [x]):
        s.append(x)
        s_set.add(x)
        return True
    return False

def _try_swap(s, s_set, i, j):
    """Try to swap s[i] with j, return True if successful."""
    if j in s_set:
        return False
    new_s = s[:]
    new_s[i] = j
    new_s_set = set(new_s)
    if len(new_s_set) != len(new_s):
        return False
    if _is_valid_exact(new_s):
        s[:] = new_s
        s_set.clear()
        s_set.update(new_s)
        return True
    return False

def _try_two_swap(s, s_set, i, j, k, l):
    """Try to swap s[i] with j and s[k] with l, return True if successful."""
    if j in s_set or l in s_set:
        return False
    if j == l:
        return False
    new_s = s[:]
    new_s[i] = j
    new_s[k] = l
    new_s_set = set(new_s)
    if len(new_s_set) != len(new_s):
        return False
    if _is_valid_exact(new_s):
        s[:] = new_s
        s_set.clear()
        s_set.update(new_s)
        return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Phase 1: Greedy construction to get a good starting point
    best_set = _greedy_construct(60)
    
    # Phase 2: Local search to improve
    for restart in range(200):
        if time.time() > deadline - 3.0:
            break
        
        # Start from best or random
        if restart == 0:
            s = best_set[:]
        else:
            # Random restart: try different strategies
            strategy = random.random()
            if strategy < 0.3:
                # Greedy from scratch
                s = _greedy_construct(random.randint(20, 60))
            elif strategy < 0.6:
                # Random subset of best
                if len(best_set) >= 10:
                    n_take = random.randint(10, len(best_set))
                    s = random.sample(best_set, n_take)
                else:
                    s = best_set[:]
            else:
                # Random dense cluster
                n_try = random.randint(15, 45)
                a_try = random.randint(1, max(1, 256 - n_try + 1))
                s = list(range(a_try, a_try + n_try))
                if not _is_valid_exact(s):
                    s = _greedy_construct(20)
        
        s_set = set(s)
        
        # Iterative improvement: try to add elements
        improved = True
        while improved:
            if time.time() > deadline - 3.0:
                break
            improved = False
            # Try adding elements in random order for diversity
            candidates = [x for x in range(1, 257) if x not in s_set]
            random.shuffle(candidates)
            for x in candidates:
                if _try_add(s, s_set, x):
                    improved = True
                    if len(s) > len(best_set):
                        best_set = s[:]
                    break
        
        # Try single swaps
        for _ in range(50):
            if time.time() > deadline - 3.0:
                break
            if len(s) < 2:
                break
            i = random.randint(0, len(s) - 1)
            candidates = [x for x in range(1, 257) if x not in s_set]
            if not candidates:
                break
            j = random.choice(candidates)
            if _try_swap(s, s_set, i, j):
                if len(s) > len(best_set):
                    best_set = s[:]
        
        # Try two-swaps for more complex moves
        for _ in range(30):
            if time.time() > deadline - 3.0:
                break
            if len(s) < 4:
                break
            i, k = random.sample(range(len(s)), 2)
            candidates = [x for x in range(1, 257) if x not in s_set]
            if len(candidates) < 2:
                break
            j, l = random.sample(candidates, 2)
            if _try_two_swap(s, s_set, i, j, k, l):
                if len(s) > len(best_set):
                    best_set = s[:]
    
    return list(best_set)
