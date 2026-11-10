import math
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    # 1. Prime Factorization of N
    # 12! = 2^10 * 3^5 * 5^2 * 7^2
    # 12! = 1024 * 243 * 25 * 49 = 479001600
    primes = [2, 3, 5, 7]
    exponents = [10, 5, 2, 2]
    
    # 2. Generate all divisors of N
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    
    # Sort divisors
    divisors.sort()
    
    # Map value to index for fast lookup (optional, but good for hashing)
    # We will use the sorted list and binary search or sets.
    # For distinctness, we need to ensure we don't pick the same number twice.
    
    best_max_val = N
    best_factors = [N]
    
    # Deadline: 80 seconds to be safe (limit is 90s)
    DEADLINE = start_time + 80
    
    # We want to minimize the maximum factor.
    # Let M be the maximum allowed factor.
    # We can binary search M? Or just iterate possible M values?
    # The factors must be > 12.
    # If we fix the set of allowed factors to be divisors of N in (12, M],
    # we need to find a subset of DISTINCT factors from this set that multiply to N.
    # This is a subset product problem with distinct elements.
    
    # Since N is fixed, the number of factors is small (max 12, but practically < 8).
    # Let's use a DFS that picks factors in increasing order.
    # To minimize the MAX factor, we should try to keep factors small.
    # However, we also need the product to reach N.
    
    # Strategy:
    # Iterate M from 13 upwards.
    # For each M, check if there exists a subset of distinct divisors of N, all in (12, M], that product to N.
    # The first M for which this is true gives the optimal solution (minimized max factor).
    # Wait, is it possible that a larger M allows a solution with a smaller max factor?
    # No, the max factor is by definition the largest element in the set.
    # So if a solution exists with max factor M, it also exists with max factor M' >= M.
    # We want the minimum M.
    
    # So, we can iterate M from 13 to N.
    # For each M, we have a candidate set S_M = {d in divisors | 12 < d <= M}.
    # We need to find a subset of S_M with product N.
    
    # Optimization:
    # Instead of checking every M, we can just DFS over the divisors.
    # But the "Minimize Max" objective suggests we should search for solutions with small max factors first.
    
    # Alternative DFS:
    # Sort divisors > 12.
    # DFS picks a factor.
    # We want to prune branches where the current max factor is already >= best_max_val.
    # This is standard branch and bound.
    
    # Let's stick to the DFS approach but optimize it heavily.
    # State: (current_product, last_index_in_sorted_divisors, depth)
    # We iterate through divisors in increasing order.
    # This naturally tries to use small factors first.
    # If we find a solution, its max factor is the last one picked (since we pick in increasing order? No, we can pick any subset).
    # Actually, if we iterate candidates in increasing order, and pick one, the next one must be larger.
    # This ensures distinctness and sorted order.
    # The max factor of the solution will be the last one picked.
    
    # If we find a solution with max factor M, we can prune any branch where the next candidate is >= M (if we haven't finished) or if we are considering a new solution, we only accept it if its max factor is < current best.
    
    # Let's refine the DFS:
    # candidates = sorted divisors > 12
    # dfs(target_rem, start_idx, current_max, path)
    #   if target_rem == 1:
    #       if current_max < best_max_val:
    #           update best
    #       return
    #   
    #   for i from start_idx to len(candidates):
    #       c = candidates[i]
    #       if c > best_max_val: break  # Pruning: if candidate is already >= best, no need to continue as we are picking in increasing order, so all subsequent are larger.
    #       if target_rem % c != 0: continue
    #       
    #       # Pruning: Can we reach target_rem / c with remaining factors?
    #       # Min product of remaining factors (distinct, > c):
    #       # This is tricky because we don't know how many factors are left.
    #       # But we know the next factor must be > c.
    #       # Let next_min = c + 1 (but must be a divisor).
    #       # Actually, just checking if target_rem / c can be formed by factors > c is the recursive step.
    #       
    #       new_target = target_rem // c
    #       # Additional pruning:
    #       # The smallest possible factor we can pick next is the smallest divisor of new_target that is > c.
    #       # If no such divisor exists and new_target != 1, fail.
    #       # Also, if new_target is not 1, it must have a factor > c.
    #       # In fact, new_target itself must be > c if we pick only one more factor? No.
    #       # If we pick one more factor, it must be new_target. So new_target > c.
    #       # If we pick more, the product of them is new_target.
    #       
    #       # Simple prune: if new_target != 1 and new_target < c: return (from loop) or break?
    #       # If new_target < c, then any factor > c will make product > new_target?
    #       # No, we need product = new_target. If all factors are > c, and c >= 13, then factors >= 14.
    #       # If new_target < 14, it's impossible to form it with factors > c (since c >= 13).
    #       # More generally, if new_target < c+1 (approx), it's hard.
    #       # Actually, if new_target < c, it's impossible because the next factor must be > c.
    #       # Wait, if we pick factor c, remaining is new_target.
    #       # Next factors must be > c.
    #       # So the product of remaining factors must be >= (c+1) * (c+2) ... ? No, just > c is not enough for product.
    #       # The smallest factor is > c. So the product of at least one factor is > c.
    #       # So if new_target <= c, it is impossible (unless new_target == 1, which is handled).
    #       # So: if new_target != 1 and new_target <= c: continue (or break? No, larger c makes it worse? No, c is fixed in this iteration).
    #       # If new_target <= c, we can't form it with factors > c. So skip this c.
    #       
    #       if new_target != 1 and new_target <= c:
    #           continue
    #           
    #       path.append(c)
    #       dfs(new_target, i + 1, max(current_max, c), path)
    #       path.pop()
    
    # This DFS explores solutions in order of increasing smallest factor?
    # Not exactly, but it tends to find solutions with smaller factors first.
    # The `if c > best_max_val: break` is the key pruning for minimization.
    # Since we iterate candidates in increasing order, once we hit a candidate >= best_max_val, all subsequent candidates are also >= best_max_val.
    # Since the factors must be distinct and we pick in increasing order, the last factor picked will be the max.
    # If we are at a node where we are picking a factor c, and c >= best_max_val, any solution extending this path will have max factor >= c >= best_max_val.
    # So we can prune.
    
    # This should be very efficient if we find a good solution early.
    # To find a good solution early, we should start with a decent upper bound.
    # [N] is the trivial solution. Max = N.
    # We can quickly find a 2-factor solution?
    # e.g., 13 * (N/13) if N%13==0.
    # N = 12! is not divisible by 13.
    # Divisors of N are composed of 2,3,5,7.
    # Smallest divisor > 12 is 14 (2*7).
    # Next is 15 (3*5).
    # 14 * 15 = 210.
    # N / 210 = 2281000?
    # Let's just let the DFS find it.
    
    # To speed up, we can pre-compute a good initial solution.
    # Greedy: pick smallest divisor > 12, divide, repeat.
    # This might fail if the remaining number has no divisors > last picked.
    # But it gives a candidate.
    
    def get_initial_solution():
        rem = N
        factors = []
        last = 12
        while rem > 1:
            found = False
            for d in candidates:
                if d <= last:
                    continue
                if rem % d == 0:
                    factors.append(d)
                    rem //= d
                    last = d
                    found = True
                    break
            if not found:
                # If no divisor > last, we must use rem itself?
                # But rem must be > last.
                if rem > last:
                    factors.append(rem)
                    rem = 1
                else:
                    return None # Failed
        return factors

    # Prepare candidates
    candidates = [d for d in divisors if d > 12]
    
    # Get initial solution to set best_max_val
    init_sol = get_initial_solution()
    if init_sol:
        best_max_val = max(init_sol)
        best_factors = init_sol
    else:
        best_max_val = N
        best_factors = [N]
        
    # DFS
    def dfs(target_rem, start_idx, current_max, path):
        nonlocal best_max_val, best_factors
        
        if time.time() > DEADLINE:
            return
            
        if target_rem == 1:
            if current_max < best_max_val:
                best_max_val = current_max
                best_factors = list(path)
            return
        
        # Heuristic: If target_rem is prime? No, we can check if it has any divisors in candidates[start_idx:].
        # If not, return.
        
        for i in range(start_idx, len(candidates)):
            c = candidates[i]
            
            # Pruning: if c >= best_max_val, break
            # Because we are picking in increasing order, the max factor of any solution from here will be at least c.
            # If c >= best_max_val, we cannot improve.
            if c >= best_max_val:
                break
                
            if target_rem % c != 0:
                continue
                
            new_target = target_rem // c
            
            # Pruning: if new_target != 1 and new_target <= c, we can't form new_target with factors > c.
            # The next factor must be > c.
            # So the product of remaining factors must be > c.
            # Thus new_target must be > c.
            if new_target != 1 and new_target <= c:
                continue
                
            # Additional Pruning:
            # The smallest possible factor we can pick next is candidates[i+1] (if exists).
            # If new_target < candidates[i+1], it's impossible?
            # Not necessarily, because we could pick multiple factors.
            # But the product of any set of factors > c is at least (c+1) * ...
            # Actually, if new_target is not 1, it must be divisible by some d in candidates[i+1:].
            # We can check this?
            # It might be expensive to check for each c.
            # Let's rely on the recursion to fail fast.
            
            path.append(c)
            dfs(new_target, i + 1, max(current_max, c), path)
            path.pop()
            
            if time.time() > DEADLINE:
                return

    dfs(N, 0, 0, [])
    
    return best_factors

# EVOLVE-BLOCK-START
# Structural Change: 
# 1. Reformulated as searching for a subset of divisors.
# 2. Used sorted candidates and "break" pruning based on best_max_val.
#    This exploits the objective (minimize max) directly in the search loop.
#    Previous approaches searched for fixed k or didn't prune as aggressively based on the current best max.
# 3. Initial solution via greedy to set a tight upper bound early.
# EVOLVE-BLOCK-END
