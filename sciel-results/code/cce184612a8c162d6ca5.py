"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 50  # 50s safety margin before 60s limit

    # The minimum number of terms is 8.
    # Try k=8 first using meet-in-the-middle.
    result = _mitm_search(8, start, deadline)
    if result is not None:
        return result

    # Fallback if k=8 search fails
    result = _mitm_search(9, start, deadline)
    if result is not None:
        return result

    # Fallback: DP over divisors of 2520 (minimizing count)
    return _dp_fallback()


def _mitm_search(k, start, deadline):
    """
    Bidirectional meet-in-the-middle search for exactly k distinct denominators >= 5
    whose reciprocals sum to 1.
    
    Strategy:
    - Split the k terms into two groups: first m terms and last (k-m) terms.
    - Precompute all possible subsets of size m from a bounded range of denominators.
    - Precompute all possible subsets of size (k-m) from a complementary range.
    - Combine to find pairs that sum to 1.
    """
    from fractions import Fraction
    import sys
    
    sys.setrecursionlimit(10000)
    
    # We need k terms. Let's split into m and k-m.
    # For k=8, m=4 is a good split.
    m = k // 2
    n = k - m  # k-m
    
    # The denominators must be distinct and >= 5.
    # We'll search in a bounded range. For k=8, denominators are typically small.
    # Let's try denominators in [5, D] for some reasonable D.
    
    # First, let's do a DFS to find a solution with pruning, but using exact fractions.
    # This is more reliable than float-based pruning.
    
    def dfs_exact(target, min_den, remaining, current):
        """DFS using exact Fraction arithmetic."""
        if time.time() > deadline:
            return None
        if remaining == 0:
            return current if target == 0 else None
        
        if target <= 0:
            return None
        
        # Max possible sum with 'remaining' terms starting from min_den
        # 1/min_den + 1/(min_den+1) + ... + 1/(min_den+remaining-1)
        max_sum = sum(Fraction(1, min_den + i) for i in range(remaining))
        if target > max_sum:
            return None
        
        # Min possible sum: we need target, so 1/d >= target - max_sum_of_rest
        # Actually, d must satisfy: 1/d <= target (so d >= 1/target)
        # And: 1/d + 1/(d+1) + ... + 1/(d+remaining-1) >= target
        # Simple lower bound: d >= ceil(1/target) if target < 1
        # Better: d must be such that remaining terms can sum to target - 1/d
        
        # Upper bound on d: 1/d <= target => d >= 1/target
        # Also need: 1/d + (remaining-1) terms from d+1 >= target
        # The smallest d gives the largest sum, so we need to find d where sum is still >= target
        
        # Let's bound d: d >= ceil(1/target) but also d < remaining / target (rough)
        # More carefully: 1/d + 1/(d+1) + ... + 1/(d+remaining-1) >= target
        # For large d, this sum ~ remaining/d, so d <= remaining/target roughly
        
        # Lower bound for d
        d_min = max(min_den, 1)
        # Upper bound: d <= remaining / target approximately
        # target = num/den, so remaining/target = remaining * den / num
        num, den = target.numerator, target.denominator
        d_max = min(10000000, (remaining * den) // num + 5)
        
        if d_max < d_min:
            return None
        
        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return None
            
            new_target = target - Fraction(1, d)
            if new_target < 0:
                break
            if new_target == 0:
                if remaining == 1:
                    return current + [d]
                else:
                    continue
            
            result = dfs_exact(new_target, d + 1, remaining - 1, current + [d])
            if result is not None:
                return result
        return None
    
    # Try exact DFS first with better pruning
    result = dfs_exact(Fraction(1, 1), 5, k, [])
    if result is not None:
        return result
    
    # If exact DFS didn't find it in time, try a different approach:
    # Meet-in-the-middle with precomputation
    
    # Split k terms: first m terms have denominators d1 < d2 < ... < dm
    # Last n terms have denominators d_{m+1} < ... < d_k
    # All denominators >= 5 and distinct.
    
    # Precompute all subsets of size m from [5, D1]
    # Precompute all subsets of size n from [D2, D2+D3] where D2 > D1
    
    # This is complex. Let's try a smarter DFS with memoization.
    
    # Alternative: use the known solution structure.
    # For k=8, one known solution is:
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/21 + 1/30 + 1/42
    # Let's verify:
    test = [5, 6, 7, 10, 14, 21, 30, 42]
    if len(test) == k and all(5 <= d <= 10000000 for d in test):
        s = sum(Fraction(1, d) for d in test)
        if s == 1:
            return test
    
    # Another known solution for k=8:
    test2 = [5, 6, 7, 10, 14, 21, 30, 42]
    # Try other combinations
    test3 = [5, 6, 7, 10, 14, 21, 30, 42]
    
    # Let's do a more targeted search
    # We know denominators are likely small for k=8.
    # Search denominators in [5, 100]
    
    def dfs_bounded(target, min_den, max_den, remaining, current):
        if time.time() > deadline:
            return None
        if remaining == 0:
            return current if target == 0 else None
        if target <= 0:
            return None
        
        max_sum = sum(Fraction(1, min_den + i) for i in range(remaining))
        if target > max_sum:
            return None
        
        d_min = max(min_den, 1)
        d_max = min(max_den, (remaining * target.denominator) // target.numerator + 5)
        
        if d_max < d_min:
            return None
        
        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return None
            
            new_target = target - Fraction(1, d)
            if new_target < 0:
                break
            if new_target == 0:
                if remaining == 1:
                    return current + [d]
                else:
                    continue
            
            result = dfs_bounded(new_target, d + 1, max_den, remaining - 1, current + [d])
            if result is not None:
                return result
        return None
    
    # Try bounded search
    for max_d in [50, 100, 200, 500, 1000, 5000, 10000]:
        result = dfs_bounded(Fraction(1, 1), 5, max_d, k, [])
        if result is not None:
            return result
    
    return None


def _dp_fallback():
    """Subset DP over divisors of 2520, minimizing count."""
    target = 2520
    # dp[sum] = (count, list)
    dp = {0: (0, [])}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        # Iterate over a copy of keys to avoid updating current iteration's items
        items = list(dp.items())
        for total, (count, values) in items:
            new_total = total + w
            if new_total <= target:
                new_count = count + 1
                if new_total not in dp or new_count < dp[new_total][0]:
                    dp[new_total] = (new_count, values + [d])
        if target in dp:
            return dp[target][1]
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
