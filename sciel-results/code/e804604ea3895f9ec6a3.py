import time
from math import factorial
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    target = factorial(12)
    
    # Prime factorization of 12!
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    target_exps = [10, 5, 2, 1, 1]
    
    # Generate divisors > 12
    divisors = []
    # We need to generate divisors efficiently.
    # Since 12! is not huge, we can just iterate 1..sqrt(12!)
    # Or use the prime factorization to generate them.
    # Iterating is simpler and 21886 iterations is nothing.
    for i in range(1, int(target**0.5) + 1):
        if target % i == 0:
            if i > 12:
                divisors.append(i)
            d2 = target // i
            if d2 != i and d2 > 12:
                divisors.append(d2)
                
    # Calculate exponent vector for each divisor
    def get_exps(n):
        exps = [0] * 5
        for j, p in enumerate(primes):
            while n % p == 0:
                exps[j] += 1
                n //= p
        return tuple(exps)
        
    div_with_exps = []
    for d in divisors:
        exps = get_exps(d)
        div_with_exps.append((d, exps))
        
    # Sort by value
    div_with_exps.sort(key=lambda x: x[0])
    
    # Precompute suffix sums of exponents for pruning
    # We will do this inside can_solve because the list of available divisors changes with M
    # But we can precompute for the full list and then adjust?
    # No, better to compute for the filtered list.
    
    def can_solve(M, need_path=False):
        # Filter divisors <= M
        # Binary search to find the end index
        lo, hi = 0, len(div_with_exps)
        while lo < hi:
            mid = (lo + hi) // 2
            if div_with_exps[mid][0] <= M:
                lo = mid + 1
            else:
                hi = mid
        end_idx = lo
        
        if end_idx == 0:
            return (None, []) if need_path else False
            
        items = div_with_exps[:end_idx]
        n_items = len(items)
        
        # Precompute suffix sums of exponents for pruning
        # suffix_sums[i][j] = sum of exps[j] for items[i:]
        suffix_sums = [[0]*5 for _ in range(n_items + 1)]
        for i in range(n_items - 1, -1, -1):
            for j in range(5):
                suffix_sums[i][j] = suffix_sums[i+1][j] + items[i][1][j]
                
        rem = list(target_exps)
        path = []
        
        def bt(idx):
            # Check if rem is all zeros
            if all(r == 0 for r in rem):
                return True
            if idx >= n_items:
                return False
            
            # Pruning: if any remaining exp is greater than suffix sum, fail
            for j in range(5):
                if rem[j] > suffix_sums[idx][j]:
                    return False
            
            # Option 1: Take item idx
            # Check if we can subtract
            can_take = True
            for j in range(5):
                if rem[j] < items[idx][1][j]:
                    can_take = False
                    break
            
            if can_take:
                # Subtract
                for j in range(5):
                    rem[j] -= items[idx][1][j]
                path.append(items[idx][0])
                
                if bt(idx + 1):
                    return True
                
                path.pop()
                # Add back
                for j in range(5):
                    rem[j] += items[idx][1][j]
            
            # Option 2: Skip item idx
            if bt(idx + 1):
                return True
                
            return False
            
        if bt(0):
            if need_path:
                return (True, path)
            return True
        return (False, []) if need_path else False

    low = 13
    high = target
    best_sol = None
    
    while low <= high:
        if time.time() > deadline - 5.0:
            break
        mid = (low + high) // 2
        sol_exists, path = can_solve(mid, need_path=True)
        if sol_exists:
            best_sol = path
            high = mid - 1
        else:
            low = mid + 1
            
    if best_sol is not None:
        return best_sol
    
    # Fallback
    return [target]
