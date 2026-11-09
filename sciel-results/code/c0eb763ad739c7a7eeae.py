"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _check_valid(s):
    """Check if set s is reciprocal-subsum-free.
    For each x in s, 1/x must NOT equal sum of 1/y for some subset of s\{x}.
    """
    s_list = sorted(s)
    n = len(s_list)
    for i, x in enumerate(s_list):
        others = [y for y in s_list if y != x]
        if not others:
            continue
        # We need to check: does there exist a subset T of others such that sum_{y in T} 1/y = 1/x?
        # Equivalent: sum_{y in T} LCM/others... let's use the LCM trick.
        # Let L = LCM of all elements in others.
        # Then 1/y = L/(y*L) ... no, 1/y * L = L/y.
        # So sum_{y in T} 1/y = 1/x  <=>  sum_{y in T} L/y = L/x.
        # But L might not be divisible by x. Let's use LCM of x and all others.
        # Actually, let L = LCM(x, *others). Then 1/x = L/x and 1/y = L/y.
        # So we need subset sum: find subset T of {L/y for y in others} that sums to L/x.
        
        # Compute LCM of x and all others
        L = x
        for y in others:
            L = L * y // gcd(L, y)
        
        target = L // x
        values = [L // y for y in others]
        
        # Check if any subset of values sums to target
        # Use bitset or DFS. Since values can be large, let's use a set-based approach or DFS with pruning.
        # For efficiency, sort values in descending order (larger first) for better pruning.
        values_sorted = sorted(values, reverse=True)
        m = len(values_sorted)
        
        # Quick check: if sum of all values < target, impossible
        if sum(values_sorted) < target:
            continue
        
        # DFS with memoization or just careful DFS
        # Since m can be up to 23, we need to be careful.
        # Use a set to track achievable sums? That could be large.
        # Let's use DFS with pruning.
        
        found = [False]
        
        def dfs(idx, curr):
            if found[0]:
                return
            if curr == target:
                found[0] = True
                return
            if idx == m:
                return
            if curr > target:
                return
            # Pruning: even if we take all remaining, can we reach target?
            remaining_sum = sum(values_sorted[idx:])
            if curr + remaining_sum < target:
                return
            
            # Include
            dfs(idx + 1, curr + values_sorted[idx])
            if found[0]:
                return
            # Exclude
            dfs(idx + 1, curr)
        
        dfs(0, 0)
        if found[0]:
            return False
    return True

def _is_valid_fast(s, deadline=None):
    """Faster validity check with deadline support."""
    s_list = sorted(s)
    n = len(s_list)
    for i, x in enumerate(s_list):
        if deadline is not None and time.time() > deadline:
            return None
        others = [y for y in s_list if y != x]
        if not others:
            continue
        L = x
        for y in others:
            L = L * y // gcd(L, y)
        target = L // x
        values = [L // y for y in others]
        if sum(values) < target:
            continue
        values_sorted = sorted(values, reverse=True)
        m = len(values_sorted)
        
        # Prefix sums for pruning
        suffix = [0] * (m + 1)
        for i in range(m - 1, -1, -1):
            suffix[i] = suffix[i + 1] + values_sorted[i]
        
        found = [False]
        
        def dfs(idx, curr):
            if found[0] or (deadline is not None and time.time() > deadline):
                return
            if curr == target:
                found[0] = True
                return
            if idx == m:
                return
            if curr > target:
                return
            if curr + suffix[idx] < target:
                return
            dfs(idx + 1, curr + values_sorted[idx])
            if found[0]:
                return
            dfs(idx + 1, curr)
        
        dfs(0, 0)
        if found[0]:
            return False
    return True

def _greedy_build(order, deadline):
    """Build a valid set by greedily adding elements in the given order."""
    s = set()
    for x in order:
        if deadline is not None and time.time() > deadline:
            break
        candidate = s | {x}
        result = _is_valid_fast(candidate, deadline)
        if result is True:
            s = candidate
        elif result is None:
            break
    return s

def _local_search(s, deadline):
    """Improve set s by removing conflicting elements and adding new ones."""
    s = set(s)
    
    # Remove conflicting elements until valid
    max_removals = 20
    for _ in range(max_removals):
        if deadline is not None and time.time() > deadline:
            break
        result = _is_valid_fast(s, deadline)
        if result is True:
            break
        elif result is None:
            break
        else:
            # Find a conflict: for some x, there's a subset of others summing to 1/x
            # Remove x (the target) to resolve
            s_list = sorted(s)
            removed = False
            for x in s_list:
                if deadline is not None and time.time() > deadline:
                    break
                others = [y for y in s_list if y != x]
                if not others:
                    continue
                L = x
                for y in others:
                    L = L * y // gcd(L, y)
                target = L // x
                values = [L // y for y in others]
                if sum(values) < target:
                    continue
                values_sorted = sorted(values, reverse=True)
                m = len(values_sorted)
                suffix = [0] * (m + 1)
                for i in range(m - 1, -1, -1):
                    suffix[i] = suffix[i + 1] + values_sorted[i]
                found = [False]
                def dfs(idx, curr):
                    if found[0] or (deadline is not None and time.time() > deadline):
                        return
                    if curr == target:
                        found[0] = True
                        return
                    if idx == m:
                        return
                    if curr > target:
                        return
                    if curr + suffix[idx] < target:
                        return
                    dfs(idx + 1, curr + values_sorted[idx])
                    if found[0]:
                        return
                    dfs(idx + 1, curr)
                dfs(0, 0)
                if found[0]:
                    s.discard(x)
                    removed = True
                    break
            if not removed:
                break
    
    # Try to add elements
    for x in range(1, 25):
        if deadline is not None and time.time() > deadline:
            break
        if x in s:
            continue
        candidate = s | {x}
        result = _is_valid_fast(candidate, deadline)
        if result is True:
            s = candidate
        elif result is None:
            break
    
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    best_result = set()
    best_size = 0
    
    # Strategy 1: Greedy from high to low
    order1 = list(range(24, 0, -1))
    s1 = _greedy_build(order1, deadline)
    if len(s1) > best_size:
        best_result = s1
        best_size = len(s1)
    
    # Strategy 2: Greedy from low to high
    order2 = list(range(1, 25))
    s2 = _greedy_build(order2, deadline)
    if len(s2) > best_size:
        best_result = s2
        best_size = len(s2)
    
    # Strategy 3: Start with large numbers, local search
    s3 = set(range(13, 25))
    s3_improved = _local_search(s3, deadline)
    if len(s3_improved) > best_size:
        best_result = s3_improved
        best_size = len(s3_improved)
    
    # Strategy 4: Multiple random restarts
    random.seed(42)
    for attempt in range(50):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Random order
        order = list(range(1, 25))
        random.shuffle(order)
        s = _greedy_build(order, deadline)
        if len(s) > best_size:
            best_result = s
            best_size = len(s)
        
        # Local search on the result
        s_improved = _local_search(s, deadline)
        if len(s_improved) > best_size:
            best_result = s_improved
            best_size = len(s_improved)
    
    # Strategy 5: Try specific known-good patterns
    # Sets of large numbers tend to work better since 1/y is small
    for start in range(10, 20):
        if deadline is not None and time.time() > deadline - 5:
            break
        s_test = set(range(start, 25))
        s_improved = _local_search(s_test, deadline)
        if len(s_improved) > best_size:
            best_result = s_improved
            best_size = len(s_improved)
    
    # Final validation
    if deadline is not None and time.time() > deadline - 2:
        # Return best found
        pass
    else:
        final_check = _is_valid_fast(best_result, deadline)
        if final_check is not True:
            # Fallback to a known valid set
            fallback = set(range(13, 25))
            if _is_valid_fast(fallback) is True:
                return list(fallback)
    
    return list(best_result)

# EVOLVE-BLOCK-END
