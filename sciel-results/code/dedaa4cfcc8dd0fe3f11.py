import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill

    N = 479001600  # 12!
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    target_exp = [10, 5, 2, 1, 1]
    target_tuple = tuple(target_exp)
    
    # Generate all divisors of N with exponent vectors
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
    
    # Sort divisors by value ascending
    divisors.sort(key=lambda x: x[0])
    
    n_divs = len(divisors)
    
    # Get unique divisor values for binary search
    unique_vals = sorted(set(d[0] for d in divisors))
    
    def check_validity(max_val):
        """
        Check if a valid set of distinct divisors <= max_val exists
        whose product is N and count <= 12.
        Returns the list of factors if found, else None.
        """
        # Filter divisors to those <= max_val
        valid_divs = [d for d in divisors if d[0] <= max_val]
        if not valid_divs:
            return None
            
        n_v = len(valid_divs)
        
        # Precompute suffix sums of exponents for pruning
        suf = [None] * (n_v + 1)
        suf[n_v] = [0] * 5
        for i in range(n_v - 1, -1, -1):
            prev = suf[i+1]
            curr_exp = valid_divs[i][1]
            suf[i] = [prev[j] + curr_exp[j] for j in range(5)]
        
        # Check if total available exponents can cover target
        for j in range(5):
            if suf[0][j] < target_exp[j]:
                return None

        # Memoized DFS
        # State: (rem_exp_tuple, start_idx, count_used)
        # But count_used makes state space larger; instead we track count in path
        # Actually, let's use (rem_exp_tuple, start_idx) and track count separately
        # Since we need count <= 12, and each step adds 1 factor, we can prune if count > 12
        
        memo = {}
        
        def dfs(rem_exp_tuple, start_idx, count):
            key = (rem_exp_tuple, start_idx)
            if key in memo:
                result = memo[key]
                if result is None:
                    return None
                # Check if the stored solution has count + len(result) <= 12
                # Actually, memo stores the full remaining solution, so we need to be careful
                # Let's not memoize with count in key, but check count constraint in the result
                pass
            
            # Better: memoize without count, but verify count at the end
            # Actually, the count constraint is global, so let's include it differently
            # For now, let's just do DFS and check count at the leaf
            
            rem_exp = list(rem_exp_tuple)
            
            if all(e == 0 for e in rem_exp):
                # Found a complete factorization
                return []  # No more factors needed
            
            if start_idx >= n_v:
                return None
                
            # Pruning: if remaining divisors can't provide enough exponents
            suffix = suf[start_idx]
            for j in range(5):
                if suffix[j] < rem_exp[j]:
                    return None
                    
            # Try each candidate divisor from start_idx onwards
            # Sort candidates by value descending to find solutions faster?
            # Actually, ascending is fine for correctness; we just need to find ANY valid set
            
            for i in range(start_idx, n_v):
                val, exp = valid_divs[i]
                # Check if this divisor fits in remaining exponents
                fits = True
                for j in range(5):
                    if exp[j] > rem_exp[j]:
                        fits = False
                        break
                if not fits:
                    continue
                    
                new_rem = tuple(rem_exp[j] - exp[j] for j in range(5))
                res = dfs(new_rem, i + 1, count + 1)
                if res is not None:
                    # Check count constraint
                    total_count = count + 1 + len(res)
                    if total_count <= 12:
                        return [val] + res
                    # If count exceeds 12, this branch is invalid, continue searching
            
            return None
        
        result = dfs(target_tuple, 0, 0)
        return result

    # Binary search on M (the maximum allowed factor value)
    best_result = None
    low = 0
    high = len(unique_vals) - 1
    
    while low <= high:
        if time.time() > deadline:
            break
            
        mid = (low + high) // 2
        M = unique_vals[mid]
        
        res = check_validity(M)
        
        if res is not None:
            best_result = res
            high = mid - 1
        else:
            low = mid + 1
            
    if best_result is None:
        # Fallback: try a few more values or return trivial
        # The trivial solution is [N] but N > 12, so it's valid but not optimal
        # Let's try to find any valid solution by scanning up
        for M in unique_vals:
            if time.time() > deadline:
                break
            res = check_validity(M)
            if res is not None:
                best_result = res
                break
    
    if best_result is None:
        best_result = [N]
        
    return best_result
