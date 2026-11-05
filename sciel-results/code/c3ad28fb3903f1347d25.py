"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 85.0  # seconds, safety margin before 90s limit

def _check_x_mean_of_subset(selected, x):
    """
    Check if x is the mean of any subset of at least 2 elements from selected.
    Uses bitset DP.
    """
    n = len(selected)
    if n < 2:
        return False
    
    zs = [y - x for y in selected]
    max_sum = sum(abs(z) for z in zs)
    offset = max_sum
    mask = (1 << (2 * max_sum + 1)) - 1
    
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    for z in zs:
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if z >= 0:
                shifted = dp[k-1] << z
            else:
                shifted = dp[k-1] >> (-z)
            dp[k] |= (shifted & mask)
            if k >= 2 and (dp[k] & (1 << offset)):
                return True
    for k in range(2, n + 1):
        if dp[k] & (1 << offset):
            return True
    return False

def _check_existing_violation(selected, x):
    """
    For each a in selected, check if a is now the mean of a subset of (selected \ {a}) ∪ {x}
    that includes x.
    """
    for a_idx, a in enumerate(selected):
        others = selected[:a_idx] + selected[a_idx+1:]
        m = len(others)
        if m < 1:
            continue
        # Need: exists S ⊆ others, |S| >= 1, sum(S) = (|S|+1)*a - x
        # DP on others
        max_sum = sum(abs(y) for y in others)
        offset = max_sum
        mask = (1 << (2 * max_sum + 1)) - 1
        dp = [0] * (m + 1)
        dp[0] = 1 << offset
        for y in others:
            for k in range(m, 0, -1):
                if dp[k-1] == 0:
                    continue
                if y >= 0:
                    shifted = dp[k-1] << y
                else:
                    shifted = dp[k-1] >> (-y)
                dp[k] |= (shifted & mask)
        for k in range(1, m + 1):
            target = (k + 1) * a - x
            if -max_sum <= target <= max_sum:
                if dp[k] & (1 << (offset + target)):
                    return True
    return False

def _quick_pair_check(selected, x):
    """
    Quick check for 2-element subsets.
    """
    # 1. Is x mean of a pair in selected?
    for i in range(len(selected)):
        for j in range(i+1, len(selected)):
            if selected[i] + selected[j] == 2 * x:
                return True
    # 2. Is any a in selected mean of pair (x, y)?
    sel_set = set(selected)
    for a in selected:
        y = 2 * a - x
        if y in sel_set and y != a:
            return True
    return False

def _full_check(C):
    """
    Full verification.
    """
    for i in range(len(C)):
        a = C[i]
        others = C[:i] + C[i+1:]
        n = len(others)
        if n < 2:
            continue
        zs = [y - a for y in others]
        max_sum = sum(abs(z) for z in zs)
        offset = max_sum
        mask = (1 << (2 * max_sum + 1)) - 1
        dp = [0] * (n + 1)
        dp[0] = 1 << offset
        for z in zs:
            for k in range(n, 0, -1):
                if dp[k-1] == 0:
                    continue
                if z >= 0:
                    shifted = dp[k-1] << z
                else:
                    shifted = dp[k-1] >> (-z)
                dp[k] |= (shifted & mask)
                if k >= 2 and (dp[k] & (1 << offset)):
                    return True
        for k in range(2, n + 1):
            if dp[k] & (1 << offset):
                return True
    return False

def _greedy_build(order, start_time, deadline):
    """
    Greedily build a set from the given order.
    """
    selected = []
    for x in order:
        if time.time() - start_time > deadline:
            break
        if len(selected) >= 48:
            break
        
        # Quick pair check first (fast)
        if _quick_pair_check(selected, x):
            continue
        
        # Check if x is mean of subset of selected (size >= 2)
        if len(selected) >= 2:
            if _check_x_mean_of_subset(selected, x):
                continue
        
        # Check if any existing element is now violated by adding x
        if _check_existing_violation(selected, x):
            continue
        
        selected.append(x)
    
    return selected

def construct():
    start_time = time.time()
    deadline = _DEADLINE - 3
    best_set = []
    
    candidates = list(range(1, 513))
    
    # Generate diverse orders
    orders = []
    
    # Structured orders
    orders.append(candidates[:])       # Increasing
    orders.append(candidates[::-1])    # Decreasing
    
    # Spread out orders
    for lo_start in range(0, 20, 5):
        spread_order = []
        lo, hi = 1 + lo_start, 512 - lo_start
        while lo <= hi:
            spread_order.append(hi)
            if lo != hi:
                spread_order.append(lo)
            lo += 1
            hi -= 1
        orders.append(spread_order)
    
    # Random orders with different seeds
    rng = random.Random(42)
    for seed in range(200):
        if time.time() - start_time > deadline - 2:
            break
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        orders.append(rand_order)
    
    # Try each order
    for order in orders:
        if time.time() - start_time > deadline:
            break
        result = _greedy_build(order, start_time, deadline)
        if len(result) > len(best_set):
            best_set = result[:]
    
    # Final verification
    if best_set:
        if _full_check(best_set):
            return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        else:
            return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
