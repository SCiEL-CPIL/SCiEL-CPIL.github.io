def construct():
    import time
    from fractions import Fraction

    target = Fraction(5, 31)
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute: for a given max_d and count k, what's the max possible sum
    # of k distinct reciprocals all >= min_d and <= max_d?
    # Max sum: 1/min_d + 1/(min_d+1) + ... + 1/(min_d+k-1)
    # Min sum: 1/max_d + 1/(max_d-1) + ... + 1/(max_d-k+1)
    
    # Use float for fast bound checks
    def max_sum_float(min_d, k):
        s = 0.0
        for i in range(k):
            s += 1.0 / (min_d + i)
        return s

    def min_sum_float(max_d, k):
        s = 0.0
        for i in range(k):
            s += 1.0 / (max_d - i)
        return s

    target_float = 5.0 / 31.0

    def dfs(rem, min_d, max_d, depth, path):
        if time.time() > deadline:
            return False
        if depth == 0:
            return rem == 0
        if rem <= 0:
            return False

        # Float-based pruning (fast)
        if max_sum_float(min_d, depth) < target_float - sum(1.0/d for d in path):
            return False
        if min_sum_float(max_d, depth) > target_float - sum(1.0/d for d in path):
            return False

        # Exact check with Fraction
        rem_exact = rem
        if rem_exact <= 0:
            return False

        # Bounds on d
        # We need 1/d <= rem, so d >= 1/rem
        # Also d >= min_d
        d_min = max(min_d, (rem_exact.denominator + rem_exact.numerator - 1) // rem_exact.numerator)
        d_max = max_d

        # Additional pruning: even taking the best (largest) remaining reciprocals
        # from d_min..d_min+depth-1 must sum to at least rem
        # And the smallest from max_d-depth+1..max_d must sum to at most rem
        if depth >= 1:
            # Quick float check
            if max_sum_float(d_min, depth) < float(rem_exact):
                return False
            if min_sum_float(d_max, depth) > float(rem_exact):
                return False

        for d in range(d_min, d_max + 1):
            new_rem = rem_exact - Fraction(1, d)
            if new_rem < 0:
                break

            if depth - 1 == 0:
                if new_rem == 0:
                    path.append(d)
                    return True
            else:
                # Float pruning for next level
                if max_sum_float(d + 1, depth - 1) < float(new_rem):
                    break
                if min_sum_float(max_d, depth - 1) > float(new_rem):
                    break

            path.append(d)
            if dfs(new_rem, d + 1, max_d, depth - 1, path):
                return True
            path.pop()

        return False

    def find_solution_for_K(K, max_terms=8):
        """Try to find a solution with all denominators <= K."""
        # Determine minimum number of terms needed
        # Max sum with k terms all <= K: 1/1 + 1/2 + ... but they must be distinct and sum to 5/31
        # Actually min denominator is at least 1, but we need sum = 5/31
        # With k terms, max possible sum is H_k (harmonic), min is k/K (if all were K, but distinct)
        # Better: max sum with k distinct denoms <= K is sum of largest k reciprocals: 1/1+1/2+...+1/k if k<=K
        # But we need sum = 5/31, so we need enough capacity
        
        for depth in range(2, max_terms + 1):
            # Can we fit 'depth' distinct denominators in [1, K] that sum to 5/31?
            # Max sum: 1/1 + 1/2 + ... + 1/depth (if depth <= K)
            # Min sum: 1/K + 1/(K-1) + ... + 1/(K-depth+1)
            if depth <= K:
                max_s = sum(1.0/i for i in range(1, depth+1))
            else:
                max_s = sum(1.0/i for i in range(1, K+1))  # all of them
            
            if depth <= K:
                min_s = sum(1.0/(K-i) for i in range(depth))
            else:
                min_s = 0.0
            
            if max_s < target_float or min_s > target_float:
                continue
            
            path = []
            if dfs(target, 1, K, depth, path):
                return path
        return None

    # Get a baseline solution using greedy
    def greedy_solution(K):
        r = target
        used = set()
        result = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d > K:
                return None
            # Find smallest available d >= d_min
            dd = d
            while dd in used:
                dd += 1
            if dd > K:
                return None
            new_r = r - Fraction(1, dd)
            if new_r < 0:
                return None
            result.append(dd)
            used.add(dd)
            r = new_r
        return result

    # Try small K values sequentially
    best_max = float('inf')
    best_denoms = []

    # First get a baseline
    baseline = greedy_solution(1000)
    if baseline:
        best_max = max(baseline)
        best_denoms = baseline[:]
    else:
        # Fallback: simple greedy without K limit
        r = target
        used = set()
        best_denoms = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            while d in used:
                d += 1
            best_denoms.append(d)
            used.add(d)
            r -= Fraction(1, d)
        best_max = max(best_denoms)

    # Now try to improve by searching K from small values up
    # The theoretical minimum K: we need sum of reciprocals = 5/31
    # With 2 terms: 1/a + 1/b = 5/31 => 31(a+b) = 5ab => 5ab - 31a - 31b = 0
    # (5a-31)(5b-31) = 31^2 = 961
    # 961 = 1*961 = 31*31
    # 5a-31=1, 5b-31=961 => a=32/5 not integer
    # 5a-31=31, 5b-31=31 => a=62/5 not integer
    # So no 2-term solution with integer denominators.
    
    # Try K from 7 upward
    for K in range(7, min(best_max, 100) + 1):
        if time.time() > deadline:
            break
        sol = find_solution_for_K(K)
        if sol is not None:
            m = max(sol)
            if m < best_max:
                best_max = m
                best_denoms = sol[:]
            # If we found a solution with max == K, we can stop (K is the bound)
            # But the actual max might be less than K, so continue to check smaller... 
            # Actually K is the upper bound, so if max(sol) <= K, we've found one.
            # But we're iterating K upward, so the first K that works gives us the answer.
            break

    # Verify
    if best_denoms:
        s = sum(Fraction(1, d) for d in best_denoms)
        if s == target:
            return best_denoms

    return best_denoms
