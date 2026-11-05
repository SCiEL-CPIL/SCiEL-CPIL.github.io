"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _subset_sum_bitset_dp(elements):
    """
    Compute dp[k] as a bitset where bit s is set if sum s is achievable
    with exactly k elements from 'elements'.
    Returns list of integers (bitsets), indexed by k.
    """
    n = len(elements)
    if n == 0:
        return [1]  # dp[0] = 1 (sum 0 with 0 elements)
    
    # Max possible sum
    max_sum = sum(elements)
    
    # dp[k] is an integer bitset
    dp = [0] * (n + 1)
    dp[0] = 1  # sum 0 with 0 elements
    
    for elem in elements:
        # Iterate backwards to avoid using element multiple times
        for k in range(n, 0, -1):
            if dp[k-1] != 0:
                dp[k] |= (dp[k-1] << elem)
    
    return dp

def _is_valid_addition(selected, candidate):
    """
    Check if adding 'candidate' to 'selected' keeps the set valid.
    Uses bitset DP for efficient subset sum checks.
    """
    n = len(selected)
    if n == 0:
        return True
    
    # Check 1: Is candidate the mean of any subset of >=2 from selected?
    # Need: sum(S) = k * candidate for some S subset of selected, |S| >= 2
    if n >= 2:
        max_sum = sum(selected)
        dp = _subset_sum_bitset_dp(selected)
        for k in range(2, n + 1):
            target = k * candidate
            if target > max_sum:
                break
            if (dp[k] >> target) & 1:
                return False
    
    # Check 2: For each a in selected, is a the mean of a subset including candidate?
    # For a in selected: others = selected \ {a}
    # Need: sum(S) + candidate = (|S|+1) * a for some S subset of others, |S| >= 1
    # => sum(S) = (|S|+1) * a - candidate
    # Let j = |S|, j >= 1. Target = (j+1)*a - candidate
    for a in selected:
        others = [x for x in selected if x != a]
        m = len(others)
        if m == 0:
            continue
        
        max_others = sum(others)
        dp2 = _subset_sum_bitset_dp(others)
        
        for j in range(1, m + 1):
            target = (j + 1) * a - candidate
            if target < 0:
                continue
            if target > max_others:
                # Since (j+1)*a increases with j and candidate is fixed,
                # larger j gives larger target, so we can break
                break
            if (dp2[j] >> target) & 1:
                return False
    
    return True

def _greedy_build(order, deadline, current_best_size):
    """
    Greedily build a set from the given order, stopping at deadline.
    Returns the best set found.
    """
    selected = []
    selected_set = set()
    
    for x in order:
        if time.time() > deadline:
            break
        if x in selected_set:
            continue
        if len(selected) >= 48:
            break
        
        if _is_valid_addition(selected, x):
            selected.append(x)
            selected_set.add(x)
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Generate smooth numbers (2^i * 3^j) as they have fewer additive relations
    smooth = set()
    for i in range(10):  # 2^0 to 2^9
        for j in range(10):  # 3^0 to 3^9
            val = (1 << i) * (3 ** j)
            if 1 <= val <= 512:
                smooth.add(val)
    
    smooth_list = sorted(smooth)
    
    # Also consider all numbers 1 to 512
    all_numbers = list(range(1, 513))
    
    # Strategy 1: Greedy with smooth numbers first, then others
    # Sort by a heuristic: prefer numbers with more factors of 2 and 3
    def smoothness_score(x):
        score = 0
        temp = x
        while temp % 2 == 0:
            temp //= 2
            score += 1
        while temp % 3 == 0:
            temp //= 3
            score += 1
        # Penalize if remaining part is large
        if temp > 1:
            score -= temp // 10
        return score
    
    sorted_by_smoothness = sorted(all_numbers, key=lambda x: -smoothness_score(x))
    
    best_set = []
    best_size = 0
    
    # Try multiple strategies
    strategies = [
        ("smooth_first", smooth_list + [x for x in all_numbers if x not in smooth]),
        ("smoothness_sorted", sorted_by_smoothness),
        ("ascending", all_numbers),
        ("descending", sorted(all_numbers, reverse=True)),
    ]
    
    # Add a few random permutations
    random.seed(42)
    for trial in range(5):
        shuffled = all_numbers[:]
        random.shuffle(shuffled)
        strategies.append((f"random_{trial}", shuffled))
    
    for name, order in strategies:
        if time.time() > deadline - 5:
            break
        
        current = _greedy_build(order, deadline - 2, best_size)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            
            if best_size >= 48:
                break
    
    # If we haven't reached 48, try to add more numbers
    if best_size < 48 and time.time() < deadline - 5:
        remaining = [x for x in all_numbers if x not in best_set]
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() > deadline - 2:
                break
            if len(best_set) >= 48:
                break
            if _is_valid_addition(best_set, x):
                best_set.append(x)
    
    return best_set[:48]
# EVOLVE-BLOCK-END
