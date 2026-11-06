import math
import time

def construct():
    T12 = math.factorial(12)
    start_time = time.time()
    deadline = start_time + 80

    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors of 12! that are > 12
    all_divisors = []
    
    def generate_divisors(idx, current_val):
        if idx == len(primes):
            if current_val > 12:
                all_divisors.append(current_val)
            return
        p = primes[idx]
        e = exponents[idx]
        val = 1
        for i in range(e + 1):
            generate_divisors(idx + 1, current_val * val)
            val *= p

    generate_divisors(0, 1)
    all_divisors.sort()
    n_div = len(all_divisors)
    
    # Map each divisor to its prime exponent vector
    divisor_to_exp = {}
    for d in all_divisors:
        temp = d
        exp_vec = []
        for p in primes:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            exp_vec.append(count)
        divisor_to_exp[d] = tuple(exp_vec)
    
    div_exp_list = [divisor_to_exp[d] for d in all_divisors]
    
    # Precompute for quick check if divisor fits remaining exponents
    # We will use DFS with memoization
    
    # Binary search on the maximum factor L
    low = all_divisors[0]
    high = all_divisors[-1]
    
    def can_solve(max_val):
        memo = {}
        
        def dfs(rem_exp, min_idx):
            if rem_exp[0] == 0 and rem_exp[1] == 0 and rem_exp[2] == 0 and rem_exp[3] == 0 and rem_exp[4] == 0:
                return True
            
            if min_idx >= n_div:
                return False
            
            # If the smallest available divisor is greater than max_val, we can't use any more
            if all_divisors[min_idx] > max_val:
                return False
                
            key = (rem_exp, min_idx)
            if key in memo:
                return memo[key]
            
            for i in range(min_idx, n_div):
                d = all_divisors[i]
                if d > max_val:
                    break # Since sorted, no more valid divisors
                
                exp_vec = div_exp_list[i]
                
                can_apply = True
                new_rem_exp = [0, 0, 0, 0, 0]
                for j in range(5):
                    if rem_exp[j] < exp_vec[j]:
                        can_apply = False
                        break
                    new_rem_exp[j] = rem_exp[j] - exp_vec[j]
                
                if not can_apply:
                    continue
                
                if dfs(tuple(new_rem_exp), i + 1):
                    memo[key] = True
                    return True
            
            memo[key] = False
            return False
        
        return dfs(tuple(exponents), 0)

    best_L = -1
    for _ in range(20): # Enough for binary search
        if low > high:
            break
        mid = (low + high) // 2
        if can_solve(mid):
            best_L = mid
            high = mid - 1
        else:
            low = mid + 1
            
    if best_L == -1:
        # Fallback to single factor
        return [T12]
        
    # Reconstruct solution
    memo = {}
    choice = {}
    
    def dfs_reconstruct(rem_exp, min_idx, max_val):
        if rem_exp[0] == 0 and rem_exp[1] == 0 and rem_exp[2] == 0 and rem_exp[3] == 0 and rem_exp[4] == 0:
            return []
        
        if min_idx >= n_div:
            return None
            
        if all_divisors[min_idx] > max_val:
            return None
            
        key = (rem_exp, min_idx)
        if key in memo:
            if memo[key]:
                # We stored the choice, need to retrieve path
                # Actually, storing the whole path in memo is expensive.
                # Better to store the chosen index.
                pass
        
        for i in range(min_idx, n_div):
            d = all_divisors[i]
            if d > max_val:
                break
            
            exp_vec = div_exp_list[i]
            
            can_apply = True
            new_rem_exp = [0, 0, 0, 0, 0]
            for j in range(5):
                if rem_exp[j] < exp_vec[j]:
                    can_apply = False
                    break
                new_rem_exp[j] = rem_exp[j] - exp_vec[j]
            
            if not can_apply:
                continue
            
            sub_path = dfs_reconstruct(tuple(new_rem_exp), i + 1, max_val)
            if sub_path is not None:
                return [d] + sub_path
        
        return None

    solution = dfs_reconstruct(tuple(exponents), 0, best_L)
    
    if solution is None:
        return [T12]
        
    return solution
