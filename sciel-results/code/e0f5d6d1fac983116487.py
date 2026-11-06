import time
import random

def _build_dp(elements):
    """Build DP bitsets for subset sums by cardinality."""
    n = len(elements)
    if n == 0:
        return [0] * 1
    dp = [0] * (n + 1)
    dp[0] = 1
    for e in elements:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << e)
    return dp

def _check_valid_incremental(A, x):
    """
    Check if adding x to set A keeps the set mean-free.
    A: list of current elements (all distinct positive integers)
    x: candidate element to add
    Returns True if A ∪ {x} is valid.
    """
    n = len(A)
    if n == 0:
        return True
    
    # Check 1: Is x the mean of any subset of A of size k >= 2?
    # Since all elements in A are distinct positive integers, and x could be smaller or larger.
    # We need: sum of k elements from A == k * x for some k >= 2.
    dp_A = _build_dp(A)
    for k in range(2, n + 1):
        target = k * x
        if target <= 0:
            continue
        # Max possible sum of k elements from A
        # We can just check the bit
        if (dp_A[k] >> target) & 1:
            return False
    
    # Check 2: Is any a in A the mean of a subset of (A ∪ {x}) \ {a} of size k >= 2?
    # The subset must include x (otherwise it was already checked when A was built).
    # Subset S ⊆ (A ∪ {x}) \ {a}, |S| = k >= 2, x ∈ S, sum(S) = k * a
    # Let T = S \ {x}, so T ⊆ A \ {a}, |T| = k - 1 = j >= 1
    # sum(T) + x = k * a => sum(T) = k * a - x = (j+1) * a - x = j*a + a - x
    # So we need: exists j >= 1, T ⊆ A\{a}, |T| = j, sum(T) = j*a + a - x
    
    A_sorted = sorted(A)
    
    for a in A:
        others = [v for v in A if v != a]
        if len(others) < 1:
            continue
        
        others_sorted = sorted(others)
        m = len(others)
        
        # Check if any j in 1..m has target = j*a + a - x within [min_sum_j, max_sum_j]
        needs_dp = False
        for j in range(1, m + 1):
            target = j * a + a - x
            if target <= 0:
                continue
            min_sum = sum(others_sorted[:j])
            max_sum = sum(others_sorted[-j:])
            if min_sum <= target <= max_sum:
                needs_dp = True
                break
        
        if not needs_dp:
            continue
        
        dp_others = _build_dp(others)
        
        for j in range(1, m + 1):
            target = j * a + a - x
            if target <= 0:
                continue
            if (dp_others[j] >> target) & 1:
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Strategy: Descending greedy - start with largest numbers
    # This is the most effective ordering for this problem
    current_set = []
    for x in range(512, 0, -1):
        if time.time() > deadline:
            break
        if _check_valid_incremental(current_set, x):
            current_set.append(x)
    
    best_set = current_set[:]
    best_len = len(current_set)
    
    if best_len >= 48:
        return best_set
    
    # If descending greedy didn't reach 48, try some random restarts
    # with time-bounded execution
    attempts = 0
    while time.time() < deadline - 5.0 and best_len < 48:
        attempts += 1
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        current_set = []
        for x in candidates:
            if time.time() > deadline - 2.0:
                break
            if _check_valid_incremental(current_set, x):
                current_set.append(x)
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
        
        if best_len >= 48:
            break
    
    return best_set
