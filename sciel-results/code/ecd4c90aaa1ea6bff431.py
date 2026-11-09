import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 45  # Safety margin for 60s limit
    
    target = math.factorial(12)
    
    # Get all divisors of target > 12
    divisors = set()
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            if i > 12:
                divisors.add(i)
            d2 = target // i
            if d2 > 12:
                divisors.add(d2)
    
    sorted_divs = sorted(divisors)
    
    def min_prod(x, m):
        p = 1
        for i in range(1, m+1):
            p *= (x + i)
            if p > target:
                return p
        return p

    best_global_max = float('inf')
    best_solution = None
    
    for k in range(1, 13):
        if time.time() > deadline:
            break
            
        def dfs(rem, k_left, last_val, current_max, path):
            nonlocal best_global_max, best_solution
            if time.time() > deadline:
                return False
            
            if k_left == 1:
                if rem > last_val and rem > 12:
                    new_max = max(current_max, rem)
                    if new_max < best_global_max:
                        best_global_max = new_max
                        best_solution = path + [rem]
                    return True
                return False
            
            # Pruning: rem must be product of k_left distinct integers > last_val
            if rem < min_prod(last_val, k_left):
                return False
            
            found_any = False
            for d in sorted_divs:
                if d <= last_val:
                    continue
                if d > rem:
                    break
                if rem % d != 0:
                    continue
                
                next_rem = rem // d
                k_next = k_left - 1
                
                # Pruning: next_rem must be >= min_prod(d, k_next)
                if next_rem < min_prod(d, k_next):
                    continue
                
                new_max = max(current_max, d)
                if new_max >= best_global_max:
                    continue
                
                if dfs(next_rem, k_next, d, new_max, path + [d]):
                    found_any = True
            
            return found_any

        dfs(target, k, 12, 0, [])
        
    if best_solution is None:
        return [target]
        
    return best_solution
