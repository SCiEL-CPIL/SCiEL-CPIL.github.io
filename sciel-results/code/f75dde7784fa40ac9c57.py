import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    N = factorial(12)
    
    # Generate all divisors of N
    divisors = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
        i += 1
    divisors.sort()
    
    # We want to minimize the largest factor M.
    # Iterate M from 13 upwards.
    # The factors must be > 12.
    
    # Optimization: Pre-compute divisors > 12
    valid_divisors = [d for d in divisors if d > 12]
    
    # We can also bound M.
    # The minimum possible max factor is at least (N)^(1/k) for some k.
    # But simply iterating M from 13 is fine.
    
    best_solution = None
    
    # To speed up, we can start M from a slightly higher value if we have a good guess,
    # but 13 is small.
    
    for M in range(13, N + 1):
        if time.time() > deadline:
            break
            
        # Candidates are divisors d such that 12 < d <= M
        candidates = [d for d in valid_divisors if d <= M]
        
        if not candidates:
            continue
            
        # DFS to find if we can form N using distinct candidates
        # We sort candidates in descending order to prune faster?
        # If we pick large numbers first, we reduce target faster.
        candidates.sort(reverse=True)
        
        def dfs(target, candidates_list, current_set):
            if target == 1:
                return list(current_set)
            
            # If target is prime and not in candidates, fail?
            # Or if target < min(candidates_list), fail?
            if not candidates_list:
                return None
            if target < candidates_list[-1]: # candidates are desc, so last is smallest
                return None
                
            for i, d in enumerate(candidates_list):
                if d > target:
                    continue
                if target % d != 0:
                    continue
                
                # Try d
                current_set.append(d)
                # New candidates are those after i (to ensure distinctness and order)
                # Wait, if we pick d, we can't pick it again.
                # Since we are iterating through a sorted list, and we remove d,
                # we should pass candidates_list[i+1:] to ensure we don't pick d again
                # and we maintain the "distinct" property.
                # Also, picking in descending order ensures we don't miss combinations?
                # Yes, this is a standard subset sum/product DFS.
                
                res = dfs(target // d, candidates_list[i+1:], current_set)
                if res:
                    return res
                current_set.pop()
            return None
        
        solution = dfs(N, candidates, [])
        if solution:
            # Verify solution
            prod = 1
            for x in solution:
                prod *= x
            if prod == N and len(set(solution)) == len(solution) and all(x > 12 for x in solution) and len(solution) <= 12:
                return solution
            else:
                # Should not happen, but break if logic error
                break
                
    # If no solution found within time or range (should not happen for 12!),
    # return the trivial solution
    return [N]
