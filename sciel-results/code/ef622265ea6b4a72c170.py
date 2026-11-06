def construct():
    import time
    from fractions import Fraction

    target = Fraction(5, 31)
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    # We want to find distinct denominators d_1 < d_2 < ... < d_k
    # such that sum(1/d_i) = 5/31, minimizing d_k.

    # Strategy:
    # 1. Enumerate the first denominator d1. Since 1/d1 < 5/31, d1 > 31/5 = 6.2, so d1 >= 7.
    #    Also 1/d1 > 5/31 - (sum of remaining terms). Since we want to minimize max,
    #    we should try smaller d1 first? Actually, for minimizing the max denominator,
    #    we want the expansion to be "tight".
    #    
    #    Let's think about lower bounds.
    #    If we have k terms, the max denominator D must satisfy:
    #    5/31 = sum 1/d_i <= k/D  => D <= 31k/5.
    #    Also, the smallest denominator d1 >= ceil(31/5) = 7.
    #    
    #    Let's try to find a solution with a small maximum denominator.
    #    We can iterate over the maximum denominator K and check if a solution exists.
    #    Or, we can do a depth-first search with pruning based on the remaining sum
    #    and the maximum allowed denominator.

    # Let's use a DFS approach.
    # State: (remaining_sum, min_denom, max_allowed)
    # We want to find a set of distinct denominators >= min_denom and <= max_allowed
    # that sum to remaining_sum.

    # To make this efficient, we can use the property that if we have remaining_sum R
    # and need to pick denominators from [d_min, K], the next denominator d must satisfy:
    # 1/d <= R  => d >= 1/R
    # Also, if we pick d, the new remaining sum is R - 1/d.
    # We need R - 1/d > 0.
    # Also, the remaining sum R - 1/d must be representable by denominators in [d+1, K].
    # A necessary condition is that R - 1/d <= (number of remaining terms) / (d+1).
    # But we don't know the number of remaining terms.
    # However, we know that the maximum possible sum of reciprocals from [d+1, K] is
    # sum_{i=d+1}^{K} 1/i. If R - 1/d > sum_{i=d+1}^{K} 1/i, then it's impossible.
    # This is a strong pruning condition.

    # Let's precompute prefix sums of reciprocals for efficiency.
    # But K is unknown. We can estimate an upper bound for K.
    # Greedy method gives an upper bound.
    def greedy_expansion(target):
        denoms = []
        r = target
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            denoms.append(d)
            r -= Fraction(1, d)
        return denoms

    # Get a baseline solution to set an initial upper bound for K.
    baseline = greedy_expansion(target)
    initial_max = max(baseline)
    
    # We will search for K from a lower bound up to initial_max.
    # Lower bound for K:
    # Since 5/31 is the sum, and all denominators are <= K,
    # and we need at least 1 term, K >= 7.
    # Actually, if we use just one term, 1/d = 5/31 => d = 31/5 = 6.2, not integer.
    # If we use two terms, 1/d1 + 1/d2 = 5/31.
    # d1 >= 7.
    # 1/d2 = 5/31 - 1/d1 = (5d1 - 31)/(31d1).
    # d2 = 31d1 / (5d1 - 31).
    # For d2 to be integer, 5d1 - 31 must divide 31d1.
    # Let 5d1 - 31 = x. Then d1 = (x+31)/5.
    # d2 = 31(x+31)/5x = 31/5 + 31^2/5x.
    # This is getting complicated. Let's just iterate K.

    # To speed up the search, we can precompute the harmonic sums H(n) = sum_{i=1}^n 1/i
    # up to some limit, say initial_max.
    # Then sum_{i=d+1}^{K} 1/i = H(K) - H(d).

    max_K = initial_max
    # Precompute harmonic numbers up to max_K
    H = [0.0] * (max_K + 1)
    for i in range(1, max_K + 1):
        H[i] = H[i-1] + 1.0/i

    # We will try K from 7 to max_K.
    # For each K, we check if there is a valid expansion.
    # To avoid redundant work, we can start from the smallest K and stop when we find a solution.
    # But is it guaranteed that if a solution exists for K, it exists for K+1?
    # Not necessarily, because the denominators must be distinct and <= K.
    # However, if we have a solution with max denominator K, we can always add 1/(K+1) and adjust?
    # No, that changes the sum.
    # But we can always take the solution for K and it is also a valid solution for K+1 (since all denoms <= K < K+1).
    # So the set of valid solutions is monotonic with respect to K.
    # Therefore, we can binary search for the minimum K!

    # Wait, is it true that if a solution exists for K, it exists for K+1?
    # Yes, because the constraints are: denominators are distinct, in [1, K], and sum to 5/31.
    # If we relax K to K+1, the set of allowed denominators expands, so any solution for K is also a solution for K+1.
    # So the predicate "exists a solution with max denominator <= K" is monotonic.
    # We can binary search for the minimum K.

    # Let's define the predicate:
    def exists_solution(K):
        """Check if there is an Egyptian fraction expansion of 5/31 with all denominators <= K."""
        # DFS: pick denominators d1 < d2 < ... < dk, all <= K.
        # d1 must be >= 7 (since 1/6 > 5/31? No, 1/6 = 0.166, 5/31 = 0.161. So 1/6 > 5/31.
        # Actually, 1/6 = 0.1666, 5/31 = 0.16129. So 1/6 > 5/31.
        # So d1 must be >= 7.
        
        # We'll use a recursive DFS with pruning.
        # State: (remaining_sum, next_min_denom)
        # We want to find a set of denominators from [next_min_denom, K] that sum to remaining_sum.
        
        # Pruning conditions:
        # 1. If remaining_sum == 0: success (if we've picked at least one denominator? No, empty set sums to 0).
        #    But we need sum to be 5/31, so we start with remaining_sum = 5/31.
        #    If remaining_sum becomes 0, we have a solution.
        # 2. If remaining_sum < 0: fail.
        # 3. If remaining_sum > 0 and next_min_denom > K: fail.
        # 4. If remaining_sum > sum_{i=next_min_denom}^{K} 1/i: fail.
        #    This is the key pruning condition.
        # 5. Also, the next denominator d must satisfy 1/d <= remaining_sum, i.e., d >= 1/remaining_sum.
        #    So d must be in [max(next_min_denom, ceil(1/remaining_sum)), K].
        
        # Let's implement this.
        # To avoid deep recursion and overhead, we can use an iterative approach or optimize the recursion.
        # Given the small size (K <= initial_max, which is likely small), recursion should be fine.
        
        # We'll use a list to store the current path.
        path = []
        
        def dfs(rem, min_d):
            if time.time() > deadline:
                return False
            if rem == 0:
                return True
            if rem < 0 or min_d > K:
                return False
            
            # Pruning: max possible sum from [min_d, K]
            # If rem > H[K] - H[min_d-1], it's impossible.
            if rem > H[K] - H[min_d-1]:
                return False
            
            # The next denominator d must be >= ceil(1/rem)
            # Since rem is a Fraction, 1/rem = rem.denominator / rem.numerator
            d_min = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
            
            # Also, d must be <= K
            if d_min > K:
                return False
            
            for d in range(d_min, K + 1):
                new_rem = rem - Fraction(1, d)
                if new_rem < 0:
                    break  # Since d is increasing, 1/d is decreasing, so new_rem will only get more negative
                if dfs(new_rem, d + 1):
                    return True
            return False
        
        return dfs(target, 7)

    # Binary search for the minimum K
    lo = 7
    hi = initial_max
    
    # First, check if hi is valid (it should be, from the greedy method)
    if not exists_solution(hi):
        # This should not happen, but just in case
        return baseline

    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        if exists_solution(mid):
            hi = mid
        else:
            lo = mid + 1

    min_K = lo

    # Now, we need to actually find the solution for min_K.
    # We can modify the DFS to record the path.
    def find_solution(K):
        path = []
        
        def dfs(rem, min_d):
            if time.time() > deadline:
                return False
            if rem == 0:
                return True
            if rem < 0 or min_d > K:
                return False
            
            if rem > H[K] - H[min_d-1]:
                return False
            
            d_min = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
            if d_min > K:
                return False
            
            for d in range(d_min, K + 1):
                new_rem = rem - Fraction(1, d)
                if new_rem < 0:
                    break
                path.append(d)
                if dfs(new_rem, d + 1):
                    return True
                path.pop()
            return False
        
        if dfs(target, 7):
            return path[:]
        return None

    solution = find_solution(min_K)
    
    if solution is None:
        # Fallback to baseline if something went wrong
        return baseline
    
    return solution
