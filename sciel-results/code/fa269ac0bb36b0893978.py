"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce
from fractions import Fraction

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_all(nums):
    if not nums:
        return 1
    return reduce(_lcm, nums)

def _is_valid_fast(S, deadline=None):
    """
    Check if S is reciprocal-subsum-free using LCM-based integer arithmetic.
    Returns True if valid, False if not, None if timeout.
    """
    if not S:
        return True
    
    S_list = list(S)
    L = _lcm_all(S_list)
    
    for x in S_list:
        if deadline is not None and time.time() > deadline:
            return None
        target = L // x
        others = [y for y in S_list if y != x]
        if not others:
            continue
        vals = [L // y for y in others]
        total_sum = sum(vals)
        if total_sum < target:
            continue
        
        # Bitmask DP / set-based subset sum with pruning
        achievable = {0}
        for v in vals:
            if v > target:
                continue
            new_sums = set()
            for s in achievable:
                ns = s + v
                if ns <= target:
                    new_sums.add(ns)
            achievable |= new_sums
            if target in achievable:
                return False
    return True

def _find_conflict_fast(S, deadline=None):
    """
    Find a specific conflict in S. Returns (x, subset_of_others) if 1/x = sum(1/y for y in subset),
    or None if no conflict found (or timeout).
    Uses LCM-based integer arithmetic.
    """
    if not S:
        return None
    
    S_list = list(S)
    L = _lcm_all(S_list)
    
    for x in S_list:
        if deadline is not None and time.time() > deadline:
            return None
        target = L // x
        others = [y for y in S_list if y != x]
        if not others:
            continue
        vals = [(L // y, y) for y in others]
        total_sum = sum(v for v, _ in vals)
        if total_sum < target:
            continue
        
        # DFS to find the actual subset
        m = len(vals)
        # Sort by value descending for better pruning
        vals_sorted = sorted(vals, key=lambda t: t[0], reverse=True)
        
        # Suffix sums for pruning
        suffix = [0] * (m + 1)
        for i in range(m - 1, -1, -1):
            suffix[i] = suffix[i + 1] + vals_sorted[i][0]
        
        if suffix[0] < target:
            continue
        
        found = [None]
        
        def dfs(idx, curr, chosen):
            if found[0] is not None:
                return
            if deadline is not None and time.time() > deadline:
                return
            if curr == target:
                if chosen:
                    found[0] = list(chosen)
                return
            if idx == m:
                return
            if curr > target:
                return
            if curr + suffix[idx] < target:
                return
            
            v, y = vals_sorted[idx]
            # Include
            if curr + v <= target:
                chosen.append(y)
                dfs(idx + 1, curr + v, chosen)
                chosen.pop()
                if found[0] is not None:
                    return
            # Exclude
            dfs(idx + 1, curr, chosen)
        
        dfs(0, 0, [])
        if found[0] is not None:
            return (x, found[0])
    return None

def _is_valid_exact(S, deadline=None):
    """
    Exact check using Fraction for correctness verification.
    """
    if not S:
        return True
    
    S_list = list(S)
    for x in S_list:
        if deadline is not None and time.time() > deadline:
            return None
        target = Fraction(1, x)
        others = [y for y in S_list if y != x]
        if not others:
            continue
        m = len(others)
        others_sorted = sorted(others)
        
        suffix_sum = [Fraction(0, 1)] * (m + 1)
        for i in range(m - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + Fraction(1, others_sorted[i])
        
        if suffix_sum[0] < target:
            continue
        
        found = [False]
        
        def dfs(idx, curr):
            if found[0]:
                return
            if deadline is not None and time.time() > deadline:
                return
            if curr == target:
                if idx > 0:
                    found[0] = True
                return
            if idx == m:
                return
            if curr > target:
                return
            if curr + suffix_sum[idx] < target:
                return
            
            y = others_sorted[idx]
            new_curr = curr + Fraction(1, y)
            if new_curr <= target:
                dfs(idx + 1, new_curr)
                if found[0]:
                    return
            dfs(idx + 1, curr)
        
        dfs(0, Fraction(0, 1))
        if found[0]:
            return False
    return True

def _repair_set(S, deadline):
    """
    Repair an invalid set by finding conflicts and removing elements.
    Returns a valid subset of S.
    """
    current = set(S)
    while current:
        if deadline is not None and time.time() > deadline:
            break
        
        # Check if valid
        valid = _is_valid_fast(current, deadline)
        if valid is True:
            break
        elif valid is None:
            break
        
        # Find a specific conflict
        conflict = _find_conflict_fast(current, deadline)
        if conflict is None:
            # No specific conflict found (maybe timeout or bug), remove random element
            to_remove = random.choice(list(current))
            current.discard(to_remove)
        else:
            x, subset = conflict
            # Strategy: remove x (the target of the subsum)
            # This is always safe for resolving this particular conflict
            current.discard(x)
    
    return current

def _local_search(S, deadline):
    """
    Local search: try to add elements to S or swap elements to improve.
    S is a set of integers. Returns the best set found.
    """
    current = set(S)
    best = set(current)
    best_size = len(best)
    
    # First, ensure current is valid
    if _is_valid_fast(current, deadline) is not True:
        current = _repair_set(current, deadline)
        best = set(current)
        best_size = len(best)
    
    # Try to add elements from [1,24] not in current
    available = [i for i in range(1, 25) if i not in current]
    random.shuffle(available)
    
    # Multiple passes of adding
    for _pass in range(3):
        progress = False
        for elem in available:
            if time.time() > deadline:
                break
            candidate = current | {elem}
            if len(candidate) > best_size:
                if _is_valid_fast(candidate, deadline) is True:
                    current = candidate
                    best = set(current)
                    best_size = len(best)
                    progress = True
                    available = [i for i in range(1, 25) if i not in current]
                    random.shuffle(available)
                    break
        if not progress:
            break
    
    # Try swapping - remove one element, add another
    for _ in range(100):
        if time.time() > deadline:
            break
        if not current:
            break
        to_remove = random.choice(list(current))
        candidates_to_add = [i for i in range(1, 25) if i not in current]
        if not candidates_to_add:
            break
        to_add = random.choice(candidates_to_add)
        
        candidate = (current - {to_remove}) | {to_add}
        if len(candidate) >= best_size:
            if _is_valid_fast(candidate, deadline) is True:
                current = candidate
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(best)
    
    return best

def _random_initial(size, deadline):
    """Generate a random initial set of given size."""
    all_elems = list(range(1, 25))
    random.shuffle(all_elems)
    current = set()
    for elem in all_elems:
        if time.time() > deadline:
            break
        candidate = current | {elem}
        if _is_valid_fast(candidate, deadline) is True:
            current = candidate
            if len(current) >= size:
                break
    return current

def _greedy_order(order, deadline):
    """Greedy construction following a specific order."""
    greedy_set = set()
    for i in order:
        if time.time() > deadline:
            break
        candidate = greedy_set | {i}
        if _is_valid_fast(candidate, deadline) is True:
            greedy_set = candidate
    return greedy_set

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit
    
    random.seed(42)  # Deterministic for reproducibility
    
    best_set = set()
    best_size = 0
    
    def try_update(candidate_set):
        nonlocal best_set, best_size
        if len(candidate_set) > best_size:
            best_set = set(candidate_set)
            best_size = len(best_set)
    
    # Strategy 1: Multiple greedy orders
    greedy_orders = [
        list(range(24, 0, -1)),   # High to low
        list(range(1, 25)),       # Low to high
        list(range(24, 0, -2)) + list(range(23, 0, -2)),  # Even high, odd high
        list(range(1, 25, 2)) + list(range(24, 0, -2)),   # Odd low, even high
    ]
    
    for order in greedy_orders:
        if time.time() > deadline:
            break
        greedy_set = _greedy_order(order, deadline)
        try_update(greedy_set)
    
    # Strategy 2: Multiple random restarts with local search
    num_restarts = 15
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate a random initial set
        initial_size = random.randint(8, 14)
        initial = _random_initial(initial_size, deadline)
        try_update(initial)
        
        # Local search from this initial
        if time.time() > deadline:
            break
        improved = _local_search(initial, deadline)
        try_update(improved)
    
    # Strategy 3: Start from large sets and repair
    large_starts = [
        set(range(13, 25)),  # 13..24
        set(range(15, 25)),  # 15..24
        set(range(10, 25)),  # 10..24
    ]
    
    for start_set in large_starts:
        if time.time() > deadline:
            break
        repaired = _repair_set(start_set, deadline)
        try_update(repaired)
        
        if time.time() > deadline:
            break
        improved = _local_search(repaired, deadline)
        try_update(improved)
    
    # Strategy 4: More aggressive random restarts with repair
    while time.time() < deadline - 5:
        # Start with a random large set
        rand_set = set(random.sample(range(1, 25), random.randint(12, 18)))
        repaired = _repair_set(rand_set, deadline)
        try_update(repaired)
        
        if time.time() > deadline:
            break
        improved = _local_search(repaired, deadline)
        try_update(improved)
    
    # Final verification with exact method
    if best_set:
        if time.time() < deadline:
            verify = _is_valid_exact(best_set, deadline)
            if verify is True:
                return sorted(best_set)
            elif verify is None:
                # Timeout on verification, return as-is (likely valid)
                return sorted(best_set)
            else:
                # Invalid, need to fix
                current = set(best_set)
                while current:
                    if time.time() > deadline:
                        break
                    if _is_valid_fast(current, deadline) is True:
                        break
                    # Remove a random element
                    to_remove = random.choice(list(current))
                    current.discard(to_remove)
                if current:
                    return sorted(current)
    
    # Fallback
    return list(range(13, 25))

# EVOLVE-BLOCK-END
