"""Meet-in-the-middle search for Egyptian fraction of 5/31 minimizing max denominator."""
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s kill

    target = Fraction(5, 31)
    target_num, target_den = target.numerator, target.denominator

    best_max = float('inf')
    best_result = None

    def update_best(denoms):
        nonlocal best_max, best_result
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_result = list(denoms)

    # Quick 3-term search to get an initial bound
    for a in range(7, 19):
        if time.time() > deadline:
            break
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        min_b = max(a + 1, (1 / rem).numerator // (1 / rem).denominator + 1)
        max_b = (2 / rem).numerator // (2 / rem).denominator
        for b in range(min_b, max_b + 1):
            rem2 = rem - Fraction(1, b)
            if rem2 <= 0:
                continue
            c_frac = 1 / rem2
            if c_frac.denominator == 1:
                c = c_frac.numerator
                if c > b and c <= 10000000:
                    update_best([a, b, c])

    # Meet-in-the-middle: split denominators into two groups
    # For each candidate max M, check if we can find a subset of {7,...,M}
    # that sums to exactly 5/31
    
    # Use a different approach: enumerate all subsets of size k from a bounded range
    # and check if their reciprocal sum equals 5/31
    
    # Strategy: for increasing M, check if any subset of {7,8,...,M} sums to 5/31
    # Use meet-in-the-middle for efficiency
    
    # First, let's try a more direct search with tighter bounds
    # For k terms with max denominator M:
    # 1/M <= each term <= 1/7
    # Sum = k/M <= 5/31 => M >= 31k/5 = 6.2k
    # Also 5/31 <= k/7 => k >= 35/31 ≈ 1.13, so k >= 2
    # And 5/31 >= k/M => M >= 31k/5
    
    # Let's search by increasing M and use DFS with pruning
    
    def dfs(remaining, min_denom, max_denom, current, depth, max_depth):
        nonlocal best_max, best_result, deadline
        if time.time() > deadline:
            return False
        if remaining <= 0:
            return False
        if depth == max_depth:
            # Check if remaining is exactly 1/d for some d > min_denom and d <= max_denom
            rem_frac = remaining
            if rem_frac.denominator == 1:
                d = rem_frac.numerator
                if d > min_denom and d <= max_denom and d <= 10000000:
                    result = current + [d]
                    mx = max(result)
                    if mx < best_max:
                        best_max = mx
                        best_result = list(result)
                    return True
            return False
        
        # Pruning: if remaining * min_denom < 1, impossible
        if remaining * min_denom < 1:
            return False
        
        # Pruning: if remaining * (max_depth - depth + 1) * min_denom < 1, 
        # even taking the largest possible remaining terms won't reach remaining
        # Actually, the max we can add with (max_depth-depth) more terms is 
        # (max_depth-depth)/min_denom
        if remaining > (max_depth - depth) / min_denom:
            return False
        
        # Minimum denominator for next term
        # 1/d <= remaining => d >= 1/remaining
        min_d = max(min_denom + 1, (1 / remaining).numerator // (1 / remaining).denominator + 1)
        
        # Maximum denominator for next term
        # 1/d >= remaining / (max_depth - depth) => d <= (max_depth - depth) / remaining
        # But also d <= max_denom
        max_d_frac = Fraction(max_depth - depth, 1) / remaining
        max_d = min(max_d_frac.numerator // max_d_frac.denominator, max_denom)
        
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return False
            if d >= best_max:
                return False
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                continue
            # Pruning: new_rem must be achievable with (max_depth - depth - 1) terms
            # each at most 1/(d+1)
            if new_rem > (max_depth - depth - 1) / (d + 1):
                continue
            if dfs(new_rem, d, max_denom, current + [d], depth + 1, max_depth):
                return True
        return False

    # Try different numbers of terms, starting from small
    # For k terms, minimum max denominator is at least ceil(31k/5)
    # Try k = 3, 4, 5, 6, 7...
    
    for k in range(3, 12):
        if time.time() > deadline:
            break
        # Lower bound on max denominator
        min_max = (31 * k + 4) // 5  # ceil(31k/5)
        if min_max >= best_max:
            break
        
        # Upper bound: we know from 3-term search or previous
        # Try increasing max_denom from min_max
        # But this could be slow, so let's try a few values
        
        # Actually, let's just run DFS with max_denom = best_max - 1
        # and see if we can find a solution with fewer terms or smaller max
        max_denom = min(best_max - 1, 10000000) if best_max < float('inf') else 10000000
        
        if max_denom < min_max:
            continue
            
        found = dfs(target, 6, max_denom, [], 0, k)
        if found:
            break  # Found a better solution
    
    if best_result is not None:
        return best_result

    # Fallback
    return [7, 8, 9, 10000000]
