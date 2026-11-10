# EVOLVE-BLOCK-START
def construct():
    from math import factorial
    from itertools import product
    from functools import lru_cache
    import sys
    import bisect
    sys.setrecursionlimit(10000)

    target = factorial(12)  # 479001600

    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    primes = [2, 3, 5, 7, 11]
    target_exp = [10, 5, 2, 1, 1]
    target_state = tuple(target_exp)

    # Generate all divisors of 12!
    def gen_divisors(exp_bounds, idx=0, current=1):
        if idx == len(exp_bounds):
            return [current]
        result = []
        for e in range(exp_bounds[idx] + 1):
            result.extend(gen_divisors(exp_bounds, idx + 1, current * primes[idx] ** e))
        return result

    all_divisors = gen_divisors(target_exp)
    # Filter: > 12
    candidates = [d for d in all_divisors if d > 12]
    candidates.sort()

    # Compute exponent vectors
    def exp_vector(n):
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)

    cand_vecs = [exp_vector(c) for c in candidates]

    # DP to find the minimum max factor
    # dp[state] = (max_factor_used, list_of_indices) is too heavy for memory if we store lists.
    # Instead, we do a binary search on the max factor value M.
    # For a given M, we check if there is a subset of candidates <= M that sums to target_state.
    
    # To speed up, we can precompute for each candidate its index.
    
    def can_achieve(max_val):
        idx = bisect.bisect_right(candidates, max_val)
        if idx == 0:
            return False, None

        # reachable: state -> (prev_state, cand_idx)
        # We only need to know if it's reachable for the check.
        # But for reconstruction, we need the path.
        # Let's just check reachability first.
        reachable = {target_state: None} # Wait, we start from 0 and go to target?
        # No, standard knapsack: start from 0, add items.
        # reachable[0] = None
        # For each item, new_reachable = reachable | {state + item}
        
        reachable = {tuple([0] * 5)}
        
        for i in range(idx):
            cv = cand_vecs[i]
            new_reachable = set()
            for state in reachable:
                new_state = tuple(state[j] + cv[j] for j in range(5))
                if all(new_state[j] <= target_exp[j] for j in range(5)):
                    new_reachable.add(new_state)
            reachable = reachable | new_reachable
            if target_state in reachable:
                return True, None # Found, but no path yet
        
        return False, None

    # Binary search for the minimum max_val
    unique_vals = sorted(set(candidates))
    lo = 0
    hi = len(unique_vals) - 1
    best_val = None
    
    while lo <= hi:
        mid = (lo + hi) // 2
        val = unique_vals[mid]
        ok, _ = can_achieve(val)
        if ok:
            best_val = val
            hi = mid - 1
        else:
            lo = mid + 1

    if best_val is None:
        return [target]

    # Now reconstruct the solution for best_val
    idx_limit = bisect.bisect_right(candidates, best_val)
    
    # dp[i][state] = (prev_state, cand_idx)
    # We use a list of dicts
    dp = [dict() for _ in range(idx_limit + 1)]
    zero_state = tuple([0] * 5)
    dp[0][zero_state] = None

    for i in range(idx_limit):
        cv = cand_vecs[i]
        # Copy previous states
        for state, parent in dp[i].items():
            if state not in dp[i+1]:
                dp[i+1][state] = parent
        
        # Add new states using candidate i
        for state in list(dp[i].keys()):
            new_state = tuple(state[j] + cv[j] for j in range(5))
            if all(new_state[j] <= target_exp[j] for j in range(5)):
                if new_state not in dp[i+1]:
                    dp[i+1][new_state] = (state, i)

    if target_state not in dp[idx_limit]:
        # Fallback
        return [target]

    # Reconstruct
    result_indices = []
    state = target_state
    i = idx_limit
    while i > 0:
        parent = dp[i].get(state)
        if parent is None:
            i -= 1
            continue
        prev_state, cand_idx = parent
        result_indices.append(cand_idx)
        state = prev_state
        i -= 1

    result = [candidates[i] for i in result_indices]
    
    # Verify
    prod = 1
    for r in result:
        prod *= r
    if prod != target:
        return [target]
    if len(result) != len(set(result)):
        return [target]
    if any(r <= 12 for r in result):
        return [target]
        
    return result

# EVOLVE-BLOCK-END
