"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction
    from heapq import heappush, heappop

    deadline = time.time() + 80.0  # safety margin before 90s kill
    target = Fraction(5, 31)

    # We want to minimize the largest denominator.
    # Strategy: Binary search on the maximum allowed denominator K.
    # For a given K, check if there exists a set of distinct denominators
    # in [1, K] summing to 1/d = target.
    #
    # The check is done via a DFS that tries to complete the fraction using
    # denominators <= K. To make it efficient, we use a greedy lower bound
    # on the next denominator: d >= ceil(1/r) where r is the remaining fraction.
    # We also prune if the sum of the reciprocals of the largest available
    # denominators (greedily) cannot reach the target.
    #
    # Actually, a simpler and more robust approach for small K:
    # Use a DFS that picks denominators in increasing order (or any order)
    # but bounded by K. Since K is small in the optimal case, the search space
    # is manageable.
    #
    # Let's find a good upper bound first using the greedy algorithm.
    r = target
    greedy_denoms = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        greedy_denoms.append(d)
        r -= Fraction(1, d)
    upper_bound = max(greedy_denoms)

    # Lower bound: at least 8 (since 5/31 > 1/8, and we need at least 1 term,
    # but actually the max denom must be at least ceil(31/5) = 7? No, 1/7 < 5/31?
    # 31/5 = 6.2. So 1/7 < 5/31. But we need sum of reciprocals.
    # If we use 1 term, d=31/5 not integer.
    # If 2 terms: 1/a + 1/b = 5/31. (5a-31)(5b-31) = 31^2 = 961.
    # Divisors of 961: 1, 31, 961.
    # 5a-31=1 => a=32/5 no.
    # 5a-31=31 => a=62/5 no.
    # 5a-31=961 => a=992/5 no.
    # So no 2-term solution.
    # 3 terms?
    # Let's just binary search from a reasonable lower bound.
    # Lower bound: max(denom) must be at least ceil(1/target) = 7?
    # Actually, if max denom is K, the largest possible sum with distinct denoms <= K is
    # 1/1 + 1/2 + ... but we are constrained by distinctness and the specific target.
    # A safe lower bound is 7. Let's start from 7.
    lower_bound = 7

    def can_solve(K, target_frac):
        """
        Check if target_frac can be represented as sum of distinct reciprocals
        with all denominators <= K.
        Returns the list of denominators if found, else None.
        """
        # We use a DFS.
        # State: current remainder, min allowed denominator (to ensure distinctness and sorted order),
        # and the list of chosen denominators.
        # To prune:
        # 1. If remainder == 0, success.
        # 2. If remainder < 0, fail.
        # 3. Lower bound for next denominator d: d >= ceil(1/remainder).
        # 4. Upper bound for next denominator d: d <= K.
        # 5. Pruning by maximum possible sum:
        #    The maximum sum we can achieve with denominators in [d, K] is
        #    sum_{i=d}^{K} 1/i. If this is < remainder, then fail.
        #    (This is a loose bound but helps).
        
        # Precompute prefix sums of reciprocals for fast max-sum check?
        # Or just compute on the fly if K is small.
        # K is likely < 100, so on-the-fly is fine.

        def max_sum_from(d, K):
            s = 0
            for i in range(d, K + 1):
                s += Fraction(1, i)
            return s

        def dfs(rem, min_d, path):
            if time.time() > deadline:
                return None
            if rem == 0:
                return path[:]
            if rem < 0:
                return None
            
            # Lower bound for d
            d_low = (rem.denominator + rem.numerator - 1) // rem.numerator
            if d_low < min_d:
                d_low = min_d
            
            # Upper bound for d
            d_high = K
            
            # If d_low > d_high, no solution
            if d_low > d_high:
                return None
            
            # Pruning: check if max possible sum from d_low to K is enough
            # Actually, we need to be careful. The max sum using ANY distinct denoms from [d_low, K]
            # is sum(1/i for i in range(d_low, K+1)).
            # If this sum < rem, we cannot reach rem.
            if max_sum_from(d_low, K) < rem:
                return None

            for d in range(d_low, d_high + 1):
                new_rem = rem - Fraction(1, d)
                if new_rem < 0:
                    break # Since d is increasing, 1/d is decreasing, so new_rem will only get smaller?
                          # Wait. rem - 1/d. As d increases, 1/d decreases, so rem - 1/d increases.
                          # So if new_rem < 0, it means 1/d > rem. For larger d, 1/d is smaller, so new_rem might become positive.
                          # So we should NOT break. We should continue.
                          # However, if d is very small, 1/d is large.
                          # Let's re-evaluate.
                          # If rem = 0.1, d=10 => new_rem=0. d=9 => new_rem=0.1-0.111 < 0.
                          # So if new_rem < 0, we can skip this d, but larger d might work.
                          # So no break. Just continue.
                if new_rem == 0:
                    return path + [d]
                
                # Recurse
                result = dfs(new_rem, d + 1, path + [d])
                if result is not None:
                    return result
            return None

        return dfs(target_frac, 1, [])

    # Binary search for the smallest K
    lo = lower_bound
    hi = upper_bound
    
    # First, ensure hi is a valid solution.
    # The greedy solution uses denominators up to upper_bound, so hi is valid.
    
    best_solution = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        # Check if mid is a valid max denominator
        sol = can_solve(mid, target)
        if sol is not None:
            best_solution = sol
            hi = mid - 1
        else:
            lo = mid + 1

    if best_solution is not None:
        return best_solution
    else:
        # Fallback to greedy if search fails or times out
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values

# EVOLVE-BLOCK-END
