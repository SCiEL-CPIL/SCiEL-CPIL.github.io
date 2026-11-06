import time
import math

def construct():
    start_time = time.time()
    timeout = 55.0
    
    target = math.factorial(12)
    
    # Prime factorization of 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    target_exp = [10, 5, 2, 1, 1]
    
    # Generate all divisors of target
    divisors = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
    
    # Filter divisors > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    
    # Precompute exponent vectors for each valid divisor
    def get_exp_vec(n):
        vec = [0] * 5
        for j, p in enumerate(primes):
            while n % p == 0:
                vec[j] += 1
                n //= p
        return tuple(vec)
    
    valid_divs_vec = [get_exp_vec(d) for d in valid_divs]
    
    n_valid = len(valid_divs)
    
    target_vec = tuple(target_exp)
    
    # Precompute: for each index i, the product of valid_divs[i:] 
    # This helps with pruning
    
    for M_idx in range(n_valid):
        M = valid_divs[M_idx]
        
        # Candidates are valid_divs[0 ... M_idx]
        candidates = valid_divs[:M_idx+1]
        candidates_vec = valid_divs_vec[:M_idx+1]
        
        n_cand = len(candidates)
        
        # Memoization
        memo = {}
        
        def dfs(start_idx, remaining_vec, count, current_factors):
            if time.time() - start_time > timeout:
                return None
            
            if all(v == 0 for v in remaining_vec):
                if count > 0:
                    return list(current_factors)
                return None
            
            if count >= 12:
                return None
            
            # Compute remaining product
            remaining_product = 1
            for j, p in enumerate(primes):
                remaining_product *= p ** remaining_vec[j]
            
            if remaining_product <= 12 and remaining_product > 1:
                return None
            
            # Memoization key
            key = (start_idx, remaining_vec)
            if key in memo:
                return memo[key]
            
            if start_idx >= n_cand:
                memo[key] = None
                return None
            
            # Pruning: if the product of all remaining candidates from start_idx 
            # is less than remaining_product, we can't reach it
            # But we can only use distinct ones, and at most (12-count) of them
            # Simple check: product of largest (12-count) candidates from start_idx
            # must be >= remaining_product
            slots_left = 12 - count
            if slots_left <= 0:
                memo[key] = None
                return None
            
            # Get the largest slots_left candidates from start_idx to end
            available = list(range(start_idx, n_cand))
            if len(available) < slots_left:
                # Not enough candidates
                # Check if product of all available >= remaining_product
                prod_all = 1
                for idx in available:
                    prod_all *= candidates[idx]
                if prod_all < remaining_product:
                    memo[key] = None
                    return None
            else:
                # Take the largest slots_left
                largest_indices = available[-slots_left:]
                prod_largest = 1
                for idx in largest_indices:
                    prod_largest *= candidates[idx]
                if prod_largest < remaining_product:
                    memo[key] = None
                    return None
            
            # Try each candidate from start_idx to end
            for i in range(start_idx, n_cand):
                c = candidates[i]
                cv = candidates_vec[i]
                
                if c > remaining_product:
                    break
                
                can_divide = True
                for j in range(5):
                    if remaining_vec[j] < cv[j]:
                        can_divide = False
                        break
                
                if can_divide:
                    new_remaining = tuple(remaining_vec[j] - cv[j] for j in range(5))
                    current_factors.append(c)
                    res = dfs(i + 1, new_remaining, count + 1, current_factors)
                    current_factors.pop()
                    if res:
                        memo[key] = res
                        return res
            
            memo[key] = None
            return None
        
        sol = dfs(0, target_vec, 0, [])
        
        if sol:
            return sol
            
        if time.time() - start_time > timeout:
            break

    return [target]
