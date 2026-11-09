import time
from itertools import combinations

def _is_smooth(n, primes):
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1

def _generate_smooth_numbers(limit, primes):
    smooths = set()
    def dfs(idx, current):
        if current > limit:
            return
        if idx == len(primes):
            smooths.add(current)
            return
        p = primes[idx]
        val = current
        while val <= limit:
            dfs(idx + 1, val)
            if val > limit // p:
                break
            val *= p
    dfs(0, 1)
    return sorted(list(smooths))

def _count_distinct_primes(nums, primes_set):
    # This is just for checking, but we know the primes if we construct with them
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # Try with primes {2, 3, 5}
    primes_3 = [2, 3, 5]
    smooth_3 = _generate_smooth_numbers(20000, primes_3)
    smooth_3_set = set(smooth_3)
    
    # Precompute valid partners for each number up to 10000
    # valid_partners[x] = set of y such that x+y is smooth
    # But we only need y > x.
    
    # Let's do backtracking
    # We want to find 12 numbers.
    
    # Optimization: 
    # For a candidate x_k, it must be such that x_i + x_k is smooth for all i < k.
    
    # Let's store smooth numbers in a set for O(1) lookup.
    
    def backtrack(k, current_list, max_val=10000):
        if time.time() > deadline:
            return None
            
        if k == 12:
            return current_list[:]
            
        # Determine candidates for x_k
        # x_k must be > current_list[-1] if k > 1 else >= 1
        min_val = current_list[-1] + 1 if k > 1 else 1
        
        # Candidates must satisfy:
        # 1. x_k <= max_val
        # 2. x_i + x_k is smooth for all i in 0..k-1
        
        # To speed up, we can iterate through possible sums?
        # Or iterate through possible x_k values?
        # Range of x_k is [min_val, max_val].
        # But max_val is 10000. 10000 iterations per step is too slow if we do it naively for 12 steps.
        
        # Better: Iterate through smooth numbers S.
        # x_k = S - x_i.
        # For a fixed i, x_k must be S - x_i for some S in smooth_3.
        # So x_k must be in the set:
        # U = union over i in 0..k-1 of { S - x_i | S in smooth_3, S - x_i > current_list[-1], S - x_i <= max_val }
        
        # Let's compute this union.
        candidates = set()
        for i in range(k):
            xi = current_list[i]
            # S must be > xi + current_list[-1] (if k>1) or > xi + 1
            # Actually S = xi + x_k. Since x_k > current_list[-1], S > xi + current_list[-1].
            # And x_k <= max_val => S <= xi + max_val.
            # Also x_k >= 1 => S >= xi + 1.
            
            # We can iterate through smooth_3 and filter.
            # smooth_3 is sorted.
            # Find start index in smooth_3.
            # S_min = xi + (current_list[-1] + 1) if k > 1 else xi + 1
            # S_max = xi + max_val
            
            # Since smooth_3 is sorted, we can use binary search or just loop if small.
            # len(smooth_3) is ~200. Looping is fine.
            for S in smooth_3:
                if S < xi + 1:
                    continue
                if S > xi + max_val:
                    break
                xk_candidate = S - xi
                if xk_candidate > current_list[-1] if k > 1 else True:
                    candidates.add(xk_candidate)
        
        # Now filter candidates:
        # For each candidate xk, check if x_i + xk is smooth for ALL i < k.
        valid_candidates = []
        for xk in candidates:
            if xk > max_val:
                continue
            valid = True
            for i in range(k):
                if (current_list[i] + xk) not in smooth_3_set:
                    valid = False
                    break
            if valid:
                valid_candidates.append(xk)
        
        valid_candidates.sort()
        
        for xk in valid_candidates:
            current_list.append(xk)
            res = backtrack(k + 1, current_list, max_val)
            if res:
                return res
            current_list.pop()
            
        return None

    # Try 3 primes
    res = backtrack(0, [])
    if res:
        return res
        
    # If not found, try 4 primes {2, 3, 5, 7}
    primes_4 = [2, 3, 5, 7]
    smooth_4 = _generate_smooth_numbers(20000, primes_4)
    smooth_4_set = set(smooth_4)
    
    # Re-run backtrack with smooth_4
    # We need to pass the smooth set to backtrack
    # Let's refactor backtrack to take smooth_set and smooth_list
    
    def backtrack_with_smooth(k, current_list, smooth_list, smooth_set, max_val=10000):
        if time.time() > deadline:
            return None
            
        if k == 12:
            return current_list[:]
            
        min_val = current_list[-1] + 1 if k > 1 else 1
        
        candidates = set()
        for i in range(k):
            xi = current_list[i]
            for S in smooth_list:
                if S < xi + 1:
                    continue
                if S > xi + max_val:
                    break
                xk_candidate = S - xi
                if (k == 1) or (xk_candidate > current_list[-1]):
                    candidates.add(xk_candidate)
        
        valid_candidates = []
        for xk in candidates:
            if xk > max_val:
                continue
            valid = True
            for i in range(k):
                if (current_list[i] + xk) not in smooth_set:
                    valid = False
                    break
            if valid:
                valid_candidates.append(xk)
        
        valid_candidates.sort()
        
        for xk in valid_candidates:
            current_list.append(xk)
            res = backtrack_with_smooth(k + 1, current_list, smooth_list, smooth_set, max_val)
            if res:
                return res
            current_list.pop()
            
        return None

    res = backtrack_with_smooth(0, [], smooth_4, smooth_4_set)
    if res:
        return res
        
    # Fallback: return the seed program's result
    return list(range(1, 13))
