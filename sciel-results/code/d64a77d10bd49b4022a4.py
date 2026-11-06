"""Construct a large set of distinct positive integers <=512 with no element equal to the mean of any subset of >=2 other elements."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

MAX_N = 512
MAX_K = 48
DEADLINE_S = 85  # safety margin before 90s kill

def _subset_sums_dp(elements):
    """
    Return a list of sets: sums[k] = set of all subset sums of size k.
    elements: list of positive ints.
    """
    n = len(elements)
    sums = [set() for _ in range(n + 1)]
    sums[0].add(0)
    for x in elements:
        # iterate backwards to avoid using x multiple times
        for k in range(n, 0, -1):
            for s in list(sums[k - 1]):
                sums[k].add(s + x)
    return sums

def _is_valid_set(selected):
    """
    Check if no element in selected is the mean of any subset of >=2 other elements.
    Uses DP subset sums for efficiency.
    """
    if len(selected) < 3:
        return True
    n = len(selected)
    sel = list(selected)
    # For each element a, check if a is the mean of some subset of others of size >=2
    # a = sum(subset)/len(subset) => sum(subset) = a * len(subset)
    # We need subset of OTHER elements (not including a) of size k >= 2 with sum = a*k
    
    # Precompute subset sums for the full set once, but we need to exclude 'a'
    # Better: for each a, build subset sums of others
    # But that's O(n^2 * 2^n) worst case... let's be smarter.
    
    # Alternative: compute all subset sums of the full set, then for each a,
    # check if there exists k>=2 such that a*k is in sums[k] and the subset doesn't include a.
    # This is tricky. Let's just do per-element check with DP on others.
    
    # For n up to 48, doing DP 48 times with 47 elements each is feasible if we optimize.
    # Actually, let's precompute subset sums for the full set, and also track which elements are used.
    # That's hard. Let's do a different approach:
    
    # For each pair (a, k) where k >= 2, check if there's a subset of size k from others with sum = a*k.
    # We can use a bitset-based DP for each 'others' set.
    
    # Given time constraints, let's use a simpler check for small sets and DP for larger.
    # Actually, the original approach with combinations is too slow for n>15 or so.
    
    # Let's use numpy bitset approach if available, else pure Python DP.
    try:
        import numpy as np
        use_numpy = True
    except ImportError:
        use_numpy = False
    
    for i, a in enumerate(sel):
        others = [x for j, x in enumerate(sel) if j != i]
        m = len(others)
        if m < 2:
            continue
        
        if use_numpy:
            # Bitset DP: dp[k] is a bitset where bit s is set if sum s is achievable with k elements
            # Max possible sum: sum(others) <= 512*48 = 24576
            max_sum = sum(others)
            # dp[k] will be a Python int acting as a bitset
            dp = [0] * (m + 1)
            dp[0] = 1  # bit 0 set: sum 0 with 0 elements
            
            for x in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            
            # Check for each k >= 2: is bit (a*k) set in dp[k]?
            for k in range(2, m + 1):
                target = a * k
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return False
        else:
            # Pure Python DP with sets
            sums = [set() for _ in range(m + 1)]
            sums[0].add(0)
            for x in others:
                for k in range(m, 0, -1):
                    for s in list(sums[k-1]):
                        sums[k].add(s + x)
            for k in range(2, m + 1):
                if a * k in sums[k]:
                    return False
    
    return True

def _greedy_with_ordering(order, deadline):
    """
    Greedily add elements in the given order, checking validity each time.
    Returns the selected set.
    """
    selected = []
    start_time = time.time()
    
    for x in order:
        if time.time() - start_time > deadline:
            break
        if len(selected) >= MAX_K:
            break
        if x in selected:
            continue
        candidate = selected + [x]
        if _is_valid_set(candidate):
            selected = candidate
    
    return selected

def construct():
    start_time = time.time()
    deadline = DEADLINE_S - 5  # 5s safety margin
    
    best = []
    
    # Strategy 1: Powers of 2 seed + modular ordering
    # Start with powers of 2, which are known to have good subset-sum properties
    powers_of_2 = [2**i for i in range(1, 9) if 2**i <= MAX_N]  # 2,4,8,...,256
    powers_of_2 = [x for x in powers_of_2 if x <= MAX_N]
    
    # Try different orderings of remaining numbers
    remaining = [x for x in range(1, MAX_N + 1) if x not in powers_of_2]
    
    # Ordering 1: by value modulo some prime
    for mod in [7, 11, 13, 17, 23]:
        if time.time() - start_time > deadline:
            break
        order = sorted(remaining, key=lambda x: (x % mod, x))
        order = powers_of_2 + order
        result = _greedy_with_ordering(order, deadline - (time.time() - start_time))
        if len(result) > len(best):
            best = result
    
    # Ordering 2: descending (large numbers first)
    if time.time() - start_time < deadline:
        order = sorted(remaining, reverse=True)
        order = powers_of_2 + order
        result = _greedy_with_ordering(order, deadline - (time.time() - start_time))
        if len(result) > len(best):
            best = result
    
    # Ordering 3: ascending
    if time.time() - start_time < deadline:
        order = sorted(remaining)
        order = powers_of_2 + order
        result = _greedy_with_ordering(order, deadline - (time.time() - start_time))
        if len(result) > len(best):
            best = result
    
    # Ordering 4: random shuffles
    if time.time() - start_time < deadline:
        for trial in range(10):
            if time.time() - start_time > deadline:
                break
            random.shuffle(remaining)
            order = powers_of_2 + remaining
            result = _greedy_with_ordering(order, deadline - (time.time() - start_time))
            if len(result) > len(best):
                best = result
    
    # If we haven't reached MAX_K, try a more targeted approach
    if len(best) < MAX_K and time.time() - start_time < deadline:
        # Try adding numbers that are "far" from existing elements
        selected = list(best)
        for x in range(1, MAX_N + 1):
            if time.time() - start_time > deadline:
                break
            if len(selected) >= MAX_K:
                break
            if x in selected:
                continue
            candidate = selected + [x]
            if _is_valid_set(candidate):
                selected = candidate
        if len(selected) > len(best):
            best = selected
    
    return best
# EVOLVE-BLOCK-END
