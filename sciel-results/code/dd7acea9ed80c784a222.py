import math
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin before 60s limit

    target = math.factorial(12)  # 479001600
    
    # Known solution with max factor 33
    known_sol = [24, 25, 27, 28, 32, 33]
    best_sol = known_sol
    best_max = max(known_sol)
    
    # Verify known solution
    prod = 1
    for x in best_sol:
        prod *= x
    if prod != target:
        return [target]
    
    # Calculate all divisors of target that are > 12
    divs = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            if i > 12:
                divs.append(i)
            other = target // i
            if other != i and other > 12:
                divs.append(other)
    divs.sort()
    
    # We want to find if there's a solution with max < best_max
    # Try D from 13 up to best_max - 1
    for D in range(13, best_max):
        if time.time() > deadline:
            break
        
        # Candidates are divisors d such that 13 <= d <= D
        cands = [d for d in divs if 13 <= d <= D]
        if not cands:
            continue
        
        solution = None
        used = set()
        path = []
        
        def dfs(rem, min_idx, depth):
            nonlocal solution
            if solution is not None:
                return
            if time.time() > deadline:
                return
            if rem == 1:
                if len(path) >= 1:
                    solution = path[:]
                return
            
            # Max 12 factors
            if depth >= 12:
                return
            
            # Iterate candidates from min_idx onwards
            for i in range(min_idx, len(cands)):
                d = cands[i]
                if d > rem:
                    break
                
                if rem % d == 0:
                    # Check that d is not already in path (distinctness)
                    if d not in used:
                        used.add(d)
                        path.append(d)
                        dfs(rem // d, i + 1, depth + 1)
                        path.pop()
                        used.remove(d)
                        
                        if solution is not None:
                            return
        
        dfs(target, 0, 0)
        
        if solution is not None:
            best_sol = solution
            best_max = D
            break  # Found the smallest D, so we're done
    
    # Return the best solution found
    return best_sol
