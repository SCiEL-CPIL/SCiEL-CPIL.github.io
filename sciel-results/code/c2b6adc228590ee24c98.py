def construct():
    from math import factorial
    import time
    
    target = factorial(12)
    primes = [2, 3, 5, 7, 11]
    
    # Generate all divisors of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    divisors = [1]
    for p, exp in zip(primes, [10, 5, 2, 1, 1]):
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(exp + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
        
    # Sort divisors
    divisors.sort()
    
    # Map divisor to index for quick lookup? Not strictly needed if we use list.
    
    def find_solution(M):
        # Returns list of factors if feasible, else None
        valid_divs = [d for d in divisors if 12 < d <= M]
        if not valid_divs:
            return None
            
        # DFS
        # State: remaining_target, start_index
        # We use a recursive function that returns the list of factors from this point onwards
        # To avoid deep recursion issues (though depth is small), we can use standard recursion.
        
        # Optimization: Memoization?
        # State is (remaining_target, start_index).
        # remaining_target is a divisor of 12!.
        # start_index is index in valid_divs.
        # This could be large. But valid_divs is small (< 792).
        # Number of states <= 792 * 792 ~ 600k. Acceptable.
        
        memo = {}
        
        def dfs(rem_target, start_idx):
            if rem_target == 1:
                return []
                
            key = (rem_target, start_idx)
            if key in memo:
                return memo[key]
                
            # Pruning: if rem_target < valid_divs[start_idx] (if start_idx < len), no divisor can divide it 
            # because all subsequent divisors are larger.
            # Wait, valid_divs is sorted.
            if start_idx >= len(valid_divs):
                memo[key] = None
                return None
                
            if rem_target < valid_divs[start_idx]:
                memo[key] = None
                return None
                
            for i in range(start_idx, len(valid_divs)):
                d = valid_divs[i]
                if d > rem_target:
                    break
                if rem_target % d == 0:
                    res = dfs(rem_target // d, i + 1)
                    if res is not None:
                        memo[key] = [d] + res
                        return [d] + res
                        # Note: we want to return the list.
                        
            memo[key] = None
            return None
            
        return dfs(target, 0)

    # Binary search for M
    lo = 13
    hi = target
    
    # We know a solution exists for M=target (the single factor 12! is not >12? 
    # Wait, 12! > 12. So {12!} is a valid solution.
    # But we want to minimize max.
    
    while lo < hi:
        mid = (lo + hi) // 2
        if find_solution(mid) is not None:
            hi = mid
        else:
            lo = mid + 1
            
    # Find the actual solution for lo
    sol = find_solution(lo)
    
    # sol should be a list of factors
    # Verify distinctness and product
    if sol:
        # Check distinct
        if len(set(sol)) != len(sol):
            # Should not happen due to index increment
            pass
        # Check product
        prod = 1
        for f in sol:
            prod *= f
        if prod != target:
            # Fallback
            return [target]
            
        return sol
    else:
        return [target]
