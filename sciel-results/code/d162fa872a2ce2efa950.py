import time
import random

# EVOLVE-BLOCK-START
import sys

MAX_N = 512
MAX_SIZE = 48
MAX_SUM = MAX_SIZE * MAX_N  # 24576
MASK = (1 << (MAX_SUM + 1)) - 1

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_set = []
    best_len = 0
    
    # Strategies for initial 3-AP free set generation
    # 1. Ascending
    # 2. Descending
    # 3. Random shuffles
    # 4. Modular orders (e.g., sort by x % 3, x % 4, etc.)
    
    orders = []
    orders.append(list(range(1, MAX_N + 1)))
    orders.append(list(range(MAX_N, 0, -1)))
    
    # Add random shuffles
    for _ in range(50):
        s = list(range(1, MAX_N + 1))
        random.shuffle(s)
        orders.append(s)
        
    # Add modular orders
    for m in range(2, 10):
        s = list(range(1, MAX_N + 1))
        s.sort(key=lambda x: (x % m, x))
        orders.append(s)
        
    for order in orders:
        if time.time() > deadline:
            break
            
        # Step 1: Build a 3-AP free set greedily
        current_set = []
        current_set_set = set()
        
        for x in order:
            if len(current_set) >= MAX_SIZE:
                break
            if x in current_set_set:
                continue
                
            # Check if adding x creates a 3-AP
            valid = True
            # Case 1: x is the mean of two elements in current_set
            # 2x = u + w  => w = 2x - u
            for u in current_set:
                w = 2 * x - u
                if w in current_set_set and w != u:
                    valid = False
                    break
            
            # Case 2: x is part of a 3-AP where some u in current_set is the mean
            # 2u = x + w  => w = 2u - x
            if valid:
                for u in current_set:
                    w = 2 * u - x
                    if w in current_set_set and w != u and w != x:
                        valid = False
                        break
            
            # Case 3: x is part of a 3-AP where some w in current_set is the mean
            # 2w = x + u  => u = 2w - x
            if valid:
                for w in current_set:
                    u = 2 * w - x
                    if u in current_set_set and u != w and u != x:
                        valid = False
                        break
            
            if valid:
                current_set.append(x)
                current_set_set.add(x)
                
        # Step 2: Verify full condition and repair
        current_set = list(set(current_set)) # Ensure distinct, though greedy ensures it
        current_set.sort()
        
        # Repair loop
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Check validity
            is_valid, violator, subset = check_full_condition(current_set)
            if is_valid:
                if len(current_set) > best_len:
                    best_len = len(current_set)
                    best_set = current_set
                break
            else:
                # Remove one element from the violating subset or the mean
                # Heuristic: Remove the mean? Or remove a random element from subset?
                # Removing the mean breaks the specific conflict but might allow others.
                # Removing an element from the subset breaks this conflict.
                # Let's try removing the mean first, as it's the "center".
                # Or try removing elements from the subset to see if we can keep the mean.
                
                # Simple repair: Remove violator
                current_set.remove(violator)
                current_set_set.discard(violator)
                changed = True
                
                # Alternatively, try removing elements from subset
                # If we remove violator, we might drop size too fast.
                # Let's try a few random removals from subset in a small loop?
                # For now, just remove violator and restart the check.
                
        # If we broke out of repair because of timeout, we might have a valid set in best_set
        # But we should check if current_set is valid at the end of the while loop if it wasn't broken by timeout
        if time.time() < deadline:
            is_valid, _, _ = check_full_condition(current_set)
            if is_valid and len(current_set) > best_len:
                best_len = len(current_set)
                best_set = current_set

    return best_set

def check_full_condition(S):
    """
    Returns (is_valid, violator, subset).
    If valid, violator is None.
    """
    n = len(S)
    if n < 3:
        return True, None, None
        
    # Precompute DP for S
    # dp[c] is bitset of sums reachable by c elements
    dp = [0] * (n + 1)
    dp[0] = 1
    
    for u in S:
        for c in range(n, 0, -1):
            dp[c] |= (dp[c-1] << u) & MASK
            
    for v in S:
        U_size = n - 1
        for k in range(2, U_size + 1):
            target = k * v
            if target > MAX_SUM:
                continue
            if not (dp[k] >> target) & 1:
                continue
            
            # Found a sum. Now check if it's reachable from S\{v}
            # We need to find the subset.
            # Re-run DP for S\{v} to check reachability
            S_ex = [x for x in S if x != v]
            if is_sum_reachable_excluding_fast(S_ex, k, target):
                # Find the subset for reporting (optional, just need to know it's invalid)
                # For repair, we just need to know it's invalid.
                return False, v, None
                
    return True, None, None

def is_sum_reachable_excluding_fast(S, k, target):
    """
    Check if target is reachable by k elements from S.
    S is already S\{v}.
    """
    n = len(S)
    if k > n:
        return False
    if k < 2:
        return False
        
    dp = [0] * (n + 1)
    dp[0] = 1
    
    for u in S:
        for c in range(n, 0, -1):
            dp[c] |= (dp[c-1] << u) & MASK
            
    return bool((dp[k] >> target) & 1)

# EVOLVE-BLOCK-END
