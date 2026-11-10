"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """
    Returns 1..256 distinct integer denominators in [5, 10000000]
    whose reciprocal sum is exactly 1; minimizes their count.
    """
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    # The minimum number of distinct denominators >= 5 summing to 1 is 8.
    # (Max sum with 7 terms: 1/5+1/6+1/7+1/8+1/9+1/10+1/11 < 1)
    # We use DP on a small LCM to find this quickly and reliably.
    
    # Try LCM 2520 (LCM of 1..10). This is sufficient for 8-term solutions.
    result = _solve_dp(2520)
    if result is not None:
        return result
    
    if time.time() > deadline:
        raise RuntimeError("No solution found before deadline")
        
    # Fallback to larger LCM if needed (unlikely)
    result = _solve_dp(27720)
    if result is not None:
        return result
        
    if time.time() > deadline:
        raise RuntimeError("No solution found before deadline")
    
    # Final fallback: DFS for k=8
    result = _search_k_terms(8)
    if result is not None:
        return result

    raise RuntimeError("No solution found")

def _solve_dp(target):
    """
    Finds the minimum number of distinct divisors of 'target' (>= 5)
    that sum to 'target'.
    """
    INF = float('inf')
    dp = [INF] * (target + 1)
    parent = [-1] * (target + 1)
    dp[0] = 0
    
    # Generate divisors of target >= 5
    divisors = []
    for d in range(5, target + 1):
        if target % d == 0:
            divisors.append(d)
    
    # 0/1 Knapsack to minimize count
    for d in divisors:
        w = target // d
        # Iterate backwards to ensure each d is used at most once
        for s in range(target, w - 1, -1):
            if dp[s - w] + 1 < dp[s]:
                dp[s] = dp[s - w] + 1
                parent[s] = d
                
    if dp[target] == INF:
        return None
        
    # Reconstruct the solution
    result = []
    s = target
    while s > 0:
        d = parent[s]
        if d == -1:
            return None
        result.append(d)
        s -= target // d
        
    # Verify the solution
    if _verify(result):
        return result
        
    return None

def _verify(denoms):
    """Verifies if the list of denominators is valid."""
    if not denoms:
        return False
    if len(denoms) > 256:
        return False
    if len(set(denoms)) != len(denoms):
        return False
    for d in denoms:
        if not (5 <= d <= 10000000):
            return False
    total = sum(Fraction(1, d) for d in denoms)
    return total == 1

def _search_k_terms(k):
    """
    Backtracking search for k denominators.
    Used as a fallback if DP fails.
    """
    def backtrack(remaining_terms, min_denom, remaining_sum, current):
        if remaining_terms == 0:
            return list(current) if remaining_sum == 0 else None
        
        if remaining_terms == 1:
            if remaining_sum > 0:
                d = 1 / remaining_sum
                if d.denominator == 1 and d.numerator >= min_denom and d.numerator <= 10000000:
                    current.append(d.numerator)
                    return list(current)
            return None
        
        # Pruning: max possible sum
        max_possible = Fraction(0, 1)
        for i in range(remaining_terms):
            max_possible += Fraction(1, min_denom + i)
        if max_possible < remaining_sum:
            return None
            
        # Lower bound for next denominator
        if remaining_sum > 0:
            inv = Fraction(1, 1) / remaining_sum
            if inv.denominator == 1:
                low_d = max(min_denom, inv.numerator)
            else:
                low_d = max(min_denom, inv.numerator // inv.denominator + 1)
        else:
            low_d = min_denom
            
        # Upper bound (heuristic)
        upper_d = min(10000000, low_d + 10000)
        
        for d in range(low_d, upper_d + 1):
            frac_d = Fraction(1, d)
            if frac_d >= remaining_sum:
                continue
            new_remaining = remaining_sum - frac_d
            if new_remaining < 0:
                continue
            
            # Check if new_remaining is achievable
            max_rem = Fraction(0, 1)
            for i in range(1, remaining_terms):
                max_rem += Fraction(1, d + i)
            if new_remaining > max_rem:
                continue
                
            current.append(d)
            result = backtrack(remaining_terms - 1, d + 1, new_remaining, current)
            if result is not None:
                return result
            current.pop()
            
        return None
        
    return backtrack(k, 5, Fraction(1, 1), [])

# EVOLVE-BLOCK-END
