import time
import random

def construct():
    """
    Construct a set of up to 48 distinct positive integers <= 512 such that no element
    is the exact arithmetic mean of any subset of at least two OTHER elements.
    Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    best_set = []
    
    # Strategy 1: Greedy Descending (Large numbers are safer)
    if time.time() < deadline:
        s = _greedy_construct(range(512, 0, -1), deadline)
        if len(s) > len(best_set):
            best_set = s
            if len(best_set) == 48:
                return best_set

    # Strategy 2: Greedy Ascending
    if time.time() < deadline:
        s = _greedy_construct(range(1, 513), deadline)
        if len(s) > len(best_set):
            best_set = s
            if len(best_set) == 48:
                return best_set

    # Strategy 3: Random Shuffles
    if len(best_set) < 48 and time.time() < deadline:
        # Use a fixed seed for reproducibility in case of random failures, 
        # or just let it be random. Random is better for exploration.
        for _ in range(20): # Try up to 20 random permutations
            if time.time() > deadline:
                break
            candidates = list(range(1, 513))
            random.shuffle(candidates)
            s = _greedy_construct(candidates, deadline)
            if len(s) > len(best_set):
                best_set = s
                if len(best_set) == 48:
                    break
                    
    return best_set

def _greedy_construct(candidates, deadline):
    """
    Greedily add elements from candidates if valid.
    """
    selected = []
    selected_set = set()
    
    for x in candidates:
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if x in selected_set:
            continue
            
        if _is_valid_addition(selected, x):
            selected.append(x)
            selected_set.add(x)
            
    return selected

def _is_valid_addition(selected, x):
    """
    Check if adding x to selected maintains the no-mean property.
    """
    n = len(selected)
    
    # Check 1: Is x the mean of a subset of selected?
    # Sum of k elements = k * x, k >= 2.
    if n >= 2:
        if _subset_mean_exists(selected, x):
            return False

    # Check 2: For each a in selected, is a the mean of a subset including x?
    # Subset T from selected \ {a}, size k >= 1.
    # Sum(T) + x = (k+1) * a
    # Sum(T) = (k+1) * a - x
    # We need to check if Sum(T) is achievable by k elements from selected \ {a}.

    for a in selected:
        others = [y for y in selected if y != a]
        if not others:
            continue

        if _subset_sum_with_target_exists(others, a, x):
            return False

    return True

def _subset_mean_exists(selected, x):
    """
    Check if there is a subset of 'selected' of size k >= 2
    such that sum(subset) == k * x.
    """
    n = len(selected)
    if n < 2:
        return False

    max_sum = sum(selected)
    
    # dp[k] is a bitset of achievable sums with k elements
    dp = [0] * (n + 1)
    dp[0] = 1  # sum 0 with 0 elements

    for elem in selected:
        for k in range(n - 1, -1, -1):
            if dp[k]:
                dp[k + 1] |= (dp[k] << elem)

    # Check if for any k >= 2, bit (k*x) is set in dp[k]
    for k in range(2, n + 1):
        target = k * x
        if target > max_sum:
            break
        if (dp[k] >> target) & 1:
            return True

    return False

def _subset_sum_with_target_exists(others, a, x):
    """
    Check if there exists k >= 1 such that sum of k elements from 'others'
    equals (k+1)*a - x.
    """
    n = len(others)
    if n == 0:
        return False

    max_sum = sum(others)

    # dp[k] is a bitset of achievable sums with k elements
    dp = [0] * (n + 1)
    dp[0] = 1

    for elem in others:
        for k in range(n - 1, -1, -1):
            if dp[k]:
                dp[k + 1] |= (dp[k] << elem)

    for k in range(1, n + 1):
        target = (k + 1) * a - x
        if target < 0:
            continue
        if target > max_sum:
            continue
        if (dp[k] >> target) & 1:
            return True

    return False
