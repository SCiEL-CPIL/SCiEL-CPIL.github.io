"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import combinations
from fractions import Fraction

# EVOLVE-BLOCK-START
import time as _time
from fractions import Fraction

def _subset_sum_check(target, available_fracs):
    """Check if target can be expressed as sum of a subset of available_fracs."""
    if target == 0:
        return True
    if not available_fracs:
        return False
    if target < 0:
        return False
    # Use DFS with memoization
    # Sort available to help pruning
    n = len(available_fracs)
    # Quick check: sum of all available
    total = sum(available_fracs)
    if target > total:
        return False
    if target == total:
        return True
    
    # DFS
    avail = sorted(available_fracs, reverse=True)
    
    def dfs(idx, remaining):
        if remaining == 0:
            return True
        if idx >= n:
            return False
        if remaining < 0:
            return False
        # Pruning: if sum of all remaining elements < target, fail
        # (precomputed suffix sums would help but let's keep it simple)
        # Try including
        if dfs(idx + 1, remaining - avail[idx]):
            return True
        # Try excluding
        if dfs(idx + 1, remaining):
            return True
        return False
    
    return dfs(0, target)

def _is_valid_set(S):
    """Check if set S is reciprocal-subsum-free."""
    S_list = list(S)
    n = len(S_list)
    for i, a in enumerate(S_list):
        target = Fraction(1, a)
        others = [Fraction(1, t) for t in S_list if t != a]
        if _subset_sum_check(target, others):
            return False
    return True

def _can_add(a, S):
    """Check if element a can be added to set S while maintaining validity."""
    S_list = list(S)
    # Check 1: For new element a, is 1/a a subset sum of {1/t : t in S}?
    others_for_a = [Fraction(1, t) for t in S_list]
    if _subset_sum_check(Fraction(1, a), others_for_a):
        return False
    
    # Check 2: For each t in S, adding a might create a new subset sum equal to 1/t
    # 1/t = 1/a + subset_sum(S \ {t, a})
    # i.e., 1/t - 1/a = subset_sum of {1/u : u in S \ {t, a}}
    for t in S_list:
        if t == a:
            continue
        diff = Fraction(1, t) - Fraction(1, a)
        if diff < 0:
            continue
        if diff == 0:
            # 1/t = 1/a, but t != a so this can't happen
            continue
        remaining = [Fraction(1, u) for u in S_list if u != t and u != a]
        if _subset_sum_check(diff, remaining):
            return False
    
    return True

def construct():
    start_time = _time.time()
    deadline = 80.0  # safety margin
    
    # Start with the known 12-element set [13, 24]
    best_set = set(range(13, 25))
    
    # Try to add elements from 1 to 12
    candidates = list(range(1, 13))
    
    # Greedy: try adding each candidate
    for a in candidates:
        if _time.time() > deadline:
            break
        if _can_add(a, best_set):
            best_set.add(a)
    
    # Now try to improve: remove some elements and add others
    # Try different orderings
    # Also try: start from scratch with different strategies
    
    # Strategy 2: DFS to find maximum valid set
    # Use the greedy result as a starting point, then try local search
    
    # Let's also try: build from small to large
    best_size = len(best_set)
    best_final = best_set
    
    # Try multiple greedy orderings
    import random
    rng = random.Random(42)
    
    for trial in range(50):
        if _time.time() > deadline:
            break
        
        if trial == 0:
            order = list(range(1, 25))
        elif trial == 1:
            order = list(range(24, 0, -1))
        else:
            order = list(range(1, 25))
            rng.shuffle(order)
        
        current_set = set()
        for a in order:
            if _time.time() > deadline:
                break
            if _can_add(a, current_set):
                current_set.add(a)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_final = current_set
    
    # Local search: try to swap elements to increase size
    for trial in range(100):
        if _time.time() > deadline:
            break
        
        current = set(best_final)
        improved = False
        for remove_elem in list(current):
            if _time.time() > deadline:
                break
            test_set = current - {remove_elem}
            for add_elem in range(1, 25):
                if add_elem in test_set:
                    continue
                if _can_add(add_elem, test_set):
                    new_set = test_set | {add_elem}
                    if len(new_set) > len(current):
                        current = new_set
                        best_final = current
                        best_size = len(current)
                        improved = True
                        break
            if improved:
                break
    
    return sorted(best_final)
# EVOLVE-BLOCK-END
