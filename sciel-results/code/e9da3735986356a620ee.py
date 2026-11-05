"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    # We want the minimum number k of distinct denominators d_i in [5, 10^7]
    # such that sum(1/d_i) = 1.
    #
    # Strategy: For small k (5, 6, 7, ...), do a bounded DFS with strong pruning.
    # Use integer arithmetic via common denominators to avoid Fraction overhead
    # where possible, but Fraction is fine for the small depths we need.
    #
    # Key insight: for k terms summing to 1 with denominators >= 5,
    # the first denominator d1 must satisfy 1/d1 <= 1, i.e. d1 >= 1,
    # and we need (k-1) more terms summing to 1 - 1/d1 = (d1-1)/d1.
    #
    # We'll search k from 5 upward, trying to find ANY valid solution.
    # The minimum known for denominators >= 5 is around 8-10 terms.
    
    # First, try to find a solution with k terms for k = 5, 6, 7, 8, ...
    # Use DFS with pruning based on remaining sum and number of terms left.
    
    def search_k_terms(k, max_den=10**7):
        """Find k distinct denominators in [5, max_den] summing to 1."""
        if time.time() > deadline:
            return None
        
        def dfs(remaining, min_den, current, depth):
            if time.time() > deadline:
                return None
            if depth == k:
                if remaining == 0:
                    return list(current)
                return None
            if remaining <= 0:
                return None
            
            terms_left = k - depth
            # Lower bound: each remaining term is at most 1/min_den
            # So max possible sum = terms_left / min_den
            # If remaining > terms_left / min_den, impossible
            if remaining > terms_left / min_den:
                return None
            
            # Upper bound on next denominator:
            # 1/d <= remaining => d >= 1/remaining
            # Also d >= min_den
            # And we need enough room: the smallest sum with terms_left terms
            # starting from d is bounded below by sum of 1/(d+i) for i=0..terms_left-1
            # But that's complex. Simpler: d can't be too large.
            # If 1/d < remaining / terms_left, then even if all remaining terms
            # are 1/d (which they can't be since distinct), we can't reach remaining.
            # Actually: remaining = 1/d + sum(remaining terms)
            # sum(remaining terms) <= (terms_left - 1) / (d+1) [since distinct and increasing]
            # So remaining <= 1/d + (terms_left - 1)/(d+1)
            # This gives an upper bound on d.
            
            num, den = remaining.numerator, remaining.denominator
            # d >= ceil(den/num)
            d_low = max(min_den, (den + num - 1) // num)
            
            # Upper bound: from remaining <= 1/d + (terms_left-1)/(d+1)
            # remaining - (terms_left-1)/(d+1) <= 1/d
            # This is a quadratic in d. Let's just cap d reasonably.
            # A simple cap: d <= den * terms_left + 100 (heuristic)
            d_high = min(max_den, den * terms_left + 1000)
            
            # Also, if d is too large, 1/d is too small
            # We need remaining - 1/d >= 0, so d >= 1/remaining = den/num
            # Already handled by d_low.
            
            for d in range(d_low, d_high + 1):
                if time.time() > deadline:
                    return None
                new_remaining = remaining - Fraction(1, d)
                if new_remaining < 0:
                    break
                if new_remaining == 0:
                    if depth + 1 == k:
                        current.append(d)
                        result = list(current)
                        current.pop()
                        return result
                    # If we found a complete sum but haven't used all k terms,
                    # we can't add more terms (they'd make sum > 1)
                    # Actually, we need exactly k terms, so this is only valid
                    # if depth + 1 == k
                    continue
                current.append(d)
                result = dfs(new_remaining, d + 1, current, depth + 1)
                current.pop()
                if result is not None:
                    return result
            return None
        
        return dfs(Fraction(1, 1), 5, [], 0)
    
    # Try k from 5 upward
    for k in range(5, 20):
        if time.time() > deadline:
            break
        result = search_k_terms(k)
        if result is not None:
            return result
    
    # Fallback: use the original DP approach
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
