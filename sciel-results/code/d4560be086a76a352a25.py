import time
from math import factorial
from functools import lru_cache
from itertools import combinations

TARGET = factorial(12)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # Prime factorization of 12!
    # 12! = 479001600
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    
    # Generate all divisors of TARGET
    divisors = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    divisors.sort()
    
    # Filter divisors > 12
    valid_divisors = [d for d in divisors if d > 12]
    
    # For each valid divisor, compute its prime exponent vector
    def prime_exp_vec(n):
        vec = [0] * 5
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)
    
    # Map divisor to its prime exponent vector
    divisor_to_vec = {}
    for d in valid_divisors:
        divisor_to_vec[d] = prime_exp_vec(d)
    
    # Target exponent vector
    target_vec = tuple(exps)
    
    # Bitmask-based subset-sum search over prime exponent vectors
    # State: (e2, e3, e5, e7, e11) -> (max_factor_used, frozenset_of_divisors)
    # But this is too large. Instead, use DP over divisors with state being the exponent vector.
    
    # Better approach: enumerate all subsets of divisors (up to size 12) and check if their
    # exponent vectors sum to target. But 2^|valid_divisors| is too large.
    
    # Alternative: use meet-in-the-middle or DFS with strong pruning.
    # Let's use DFS with memoization on (remaining_vec, last_divisor_index, count_used)
    
    # Actually, let's try a different approach: iterate over possible maximum values M
    # and check if a valid factorization exists with all factors <= M.
    # Binary search on M.
    
    def can_factor_with_max(M, time_limit):
        """Check if TARGET can be factored into distinct integers > 12, all <= M, with product = TARGET."""
        # Filter valid divisors that are <= M
        candidates = [d for d in valid_divisors if d <= M]
        if not candidates:
            return None
        
        # Use DFS with pruning
        best_solution = None
        
        def dfs(remaining_vec, last_idx, current_factors, depth):
            nonlocal best_solution
            
            if time.time() > time_limit:
                return False
            
            # Check if remaining_vec is all zeros
            if all(v == 0 for v in remaining_vec):
                if len(current_factors) >= 1:
                    best_solution = list(current_factors)
                return True
            
            # If remaining is a single prime power that's a valid divisor
            # Check if remaining corresponds to a valid divisor > last
            rem_val = 1
            for i, p in enumerate(primes):
                rem_val *= (p ** remaining_vec[i])
            
            # If rem_val is a valid divisor > last_factor and <= M
            last_factor = current_factors[-1] if current_factors else 12
            if rem_val > last_factor and rem_val <= M and rem_val > 12 and rem_val in divisor_to_vec:
                current_factors.append(rem_val)
                best_solution = list(current_factors)
                current_factors.pop()
                return True
            
            # Try each candidate divisor
            for i in range(last_idx + 1, len(candidates)):
                d = candidates[i]
                vec = divisor_to_vec[d]
                
                # Check if d's vec can be subtracted from remaining_vec
                can_subtract = True
                for j in range(5):
                    if vec[j] > remaining_vec[j]:
                        can_subtract = False
                        break
                if not can_subtract:
                    continue
                
                # Pruning: if d >= best_max so far, skip
                if best_solution is not None and d >= max(best_solution):
                    break
                
                # New remaining vector
                new_rem = tuple(remaining_vec[j] - vec[j] for j in range(5))
                
                current_factors.append(d)
                found = dfs(new_rem, i, current_factors, depth + 1)
                current_factors.pop()
                
                if found:
                    return True
                
                # If we've tried enough at this depth, break
                if depth > 11:
                    break
            
            return False
        
        dfs(target_vec, -1, [], 0)
        return best_solution
    
    # Binary search on the maximum factor
    # Lower bound: at least 13 (smallest factor > 12)
    # Upper bound: TARGET (single factor)
    
    lo = 13
    hi = TARGET
    best_solution = [TARGET]
    
    # First, try to find any solution to establish a good upper bound
    # Use a quick DFS to find a solution
    def quick_find_solution():
        nonlocal deadline
        best = [TARGET]
        
        def dfs(remaining_vec, last_idx, current_factors, depth):
            nonlocal best
            if time.time() > deadline:
                return
            
            if all(v == 0 for v in remaining_vec):
                if len(current_factors) >= 1:
                    if max(current_factors) < max(best):
                        best = list(current_factors)
                return
            
            rem_val = 1
            for i, p in enumerate(primes):
                rem_val *= (p ** remaining_vec[i])
            
            last_factor = current_factors[-1] if current_factors else 12
            if rem_val > last_factor and rem_val > 12 and rem_val in divisor_to_vec:
                current_factors.append(rem_val)
                if max(current_factors) < max(best):
                    best = list(current_factors)
                current_factors.pop()
                return
            
            for i in range(last_idx + 1, len(valid_divisors)):
                d = valid_divisors[i]
                vec = divisor_to_vec[d]
                
                can_subtract = True
                for j in range(5):
                    if vec[j] > remaining_vec[j]:
                        can_subtract = False
                        break
                if not can_subtract:
                    continue
                
                if d >= max(best):
                    break
                
                new_rem = tuple(remaining_vec[j] - vec[j] for j in range(5))
                
                current_factors.append(d)
                dfs(new_rem, i, current_factors, depth + 1)
                current_factors.pop()
                
                if depth > 11:
                    break
        
        dfs(target_vec, -1, [], 0)
        return best
    
    # Find an initial solution
    initial_solution = quick_find_solution()
    if initial_solution and len(initial_solution) > 1:
        hi = max(initial_solution)
        best_solution = initial_solution
    else:
        hi = TARGET
    
    # Binary search for the minimum possible max factor
    while lo < hi:
        mid = (lo + hi) // 2
        sol = can_factor_with_max(mid, deadline)
        if sol is not None:
            hi = mid
            best_solution = sol
        else:
            lo = mid + 1
    
    # Verify the solution
    if best_solution and len(best_solution) >= 1 and len(best_solution) <= 12:
        prod = 1
        for f in best_solution:
            prod *= f
        if prod == TARGET and all(f > 12 for f in best_solution) and len(set(best_solution)) == len(best_solution):
            return best_solution
    
    return [TARGET]
