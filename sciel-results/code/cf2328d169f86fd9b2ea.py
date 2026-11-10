import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    N = 479001600  # 12!
    
    # Prime factorization of 12!
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    target_exp = [10, 5, 2, 1, 1]
    target_tuple = tuple(target_exp)
    
    # Generate all divisors of N
    divisors = []
    
    def gen_divisors(idx, current_exp, current_val):
        if idx == len(primes):
            if current_val > 12:
                divisors.append((current_val, tuple(current_exp)))
            return
        for e in range(target_exp[idx] + 1):
            current_exp.append(e)
            gen_divisors(idx + 1, current_exp, current_val * (primes[idx] ** e))
            current_exp.pop()
            
    gen_divisors(0, [], 1)
    
    # Sort divisors by value
    divisors.sort(key=lambda x: x[0])
    
    unique_vals = sorted(set(d[0] for d in divisors))
    
    # We iterate M in increasing order
    # For each M, we check if a valid subset exists
    # We stop at the first valid M
    
    # To speed up, we can pre-filter divisors > 12
    
    for M in unique_vals:
        if time.time() > deadline:
            break
            
        # Filter divisors <= M
        valid_divs = [d for d in divisors if d[0] <= M]
        if not valid_divs:
            continue
            
        n_v = len(valid_divs)
        
        # Precompute suffix sums of exponents
        suf_exp = [None] * (n_v + 1)
        suf_exp[n_v] = [0] * 5
        for i in range(n_v - 1, -1, -1):
            prev = suf_exp[i+1]
            curr_exp = valid_divs[i][1]
            suf_exp[i] = [prev[j] + curr_exp[j] for j in range(5)]
            
        # Check if total exponents can cover target
        can_cover = True
        for j in range(5):
            if suf_exp[0][j] < target_exp[j]:
                can_cover = False
                break
        if not can_cover:
            continue
            
        # Precompute suffix products
        # If product of remaining factors < remaining target, prune
        suf_prod = [1] * (n_v + 1)
        for i in range(n_v - 1, -1, -1):
            suf_prod[i] = suf_prod[i+1] * valid_divs[i][0]
            
        memo = {}
        
        def dfs(rem_exp_tuple, start_idx, current_prod):
            key = (rem_exp_tuple, start_idx)
            if key in memo:
                return memo[key]
                
            rem_exp = list(rem_exp_tuple)
            if all(e == 0 for e in rem_exp):
                memo[key] = []
                return []
                
            if start_idx >= n_v:
                memo[key] = None
                return None
                
            # Prune 1: Exponent sum
            for j in range(5):
                if suf_exp[start_idx][j] < rem_exp[j]:
                    memo[key] = None
                    return None
                    
            # Prune 2: Product
            # The maximum product we can form from start_idx onwards is suf_prod[start_idx]
            # The remaining target is N // current_prod
            rem_target = N // current_prod
            if suf_prod[start_idx] < rem_target:
                memo[key] = None
                return None
                
            # Try picking a divisor
            for i in range(start_idx, n_v):
                val, exp = valid_divs[i]
                fits = True
                for j in range(5):
                    if exp[j] > rem_exp[j]:
                        fits = False
                        break
                if fits:
                    new_rem = tuple(rem_exp[j] - exp[j] for j in range(5))
                    res = dfs(new_rem, i + 1, current_prod * val)
                    if res is not None:
                        result = [val] + res
                        memo[key] = result
                        return result
                        
            memo[key] = None
            return None
            
        res = dfs(target_tuple, 0, 1)
        if res is not None:
            return res
            
    # Fallback
    return [N]
