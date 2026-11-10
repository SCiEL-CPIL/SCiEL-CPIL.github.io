import time
import random
import sys

def _check_addition_fast(current_set, x):
    """
    Check if adding x to current_set creates any violation.
    Returns True if valid, False otherwise.
    """
    n = len(current_set)
    if n < 2:
        return True
    
    # 1. Check if x is the mean of any subset of current_set
    # sum(S) = k * x for S subset of current_set, |S| = k >= 2
    m = n
    max_target = m * x
    
    # DP for current_set
    # dp[k] is bitset of sums of k elements
    dp = [0] * (m + 1)
    dp[0] = 1
    for elem in current_set:
        for k in range(m, 0, -1):
            dp[k] |= (dp[k-1] << elem)
            # Mask to keep integers small
            if max_target < 50000:
                dp[k] &= (1 << (max_target + 1)) - 1
    
    for k in range(2, m + 1):
        target = k * x
        if (dp[k] >> target) & 1:
            return False

    # 2. Check if any a in current_set becomes the mean of a subset including x
    # Subset T from current_set \ {a} such that x + sum(T) = a * (1 + |T|)
    # sum(T) = a * (1 + |T|) - x
    # Let |T| = t. t >= 1.
    # T is subset of current_set \ {a} of size t.
    
    for a in current_set:
        # Build DP for current_set \ {a}
        others = [y for y in current_set if y != a]
        m_o = len(others)
        if m_o < 1:
            continue
        
        dp_o = [0] * (m_o + 1)
        dp_o[0] = 1
        max_t_target = a * (m_o + 1) - x
        if max_t_target < 0:
            continue
        
        mask = (1 << (max_t_target + 1)) - 1
        
        for elem in others:
            for k in range(m_o, 0, -1):
                dp_o[k] |= (dp_o[k-1] << elem)
                dp_o[k] &= mask
        
        for t in range(1, m_o + 1):
            target = a * (t + 1) - x
            if 0 <= target <= max_t_target:
                if (dp_o[t] >> target) & 1:
                    return False

    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    best_size = 0
    
    # Strategy: Multiple random restarts with greedy filling
    # We will try to be more aggressive with the search.
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Random starting set
        # Pick a random subset of [1, 512] of size 5-15
        start_size = random.randint(5, 15)
        initial = random.sample(range(1, 513), start_size)
        
        selected = []
        remaining = list(range(1, 513))
        random.shuffle(remaining)
        
        # First, try to add from initial? No, just use remaining.
        # We can prioritize numbers that are "good" (e.g. powers of 2, primes)
        # But random is fine.
        
        for x in remaining:
            if time.time() > deadline:
                break
            if len(selected) >= 48:
                break
            if _check_addition_fast(selected, x):
                selected.append(x)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
            if best_size == 48:
                break
        
    # If we didn't reach 48, try to improve the best set found
    if best_size < 48 and time.time() < deadline - 10:
        current = best_set[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add any remaining element
            for x in range(1, 513):
                if x in current:
                    continue
                if len(current) >= 48:
                    break
                if _check_addition_fast(current, x):
                    current.append(x)
                    improved = True
                    break
            
            if not improved:
                # Try swapping: remove one, add two?
                # Or just try to find a better configuration?
                # For now, just stop.
                pass

    return best_set[:48]
