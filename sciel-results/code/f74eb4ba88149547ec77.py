def construct():
    import time
    from fractions import Fraction
    import heapq
    
    target = Fraction(5, 31)
    deadline = time.time() + 85
    
    best_solution = [8, 31, 248]
    best_max = 248
    
    # A* search
    # State: (max_denom, remaining, min_denom, path)
    heap = [(0, target, 1, [])]
    visited = {(target, 1): 0}
    
    found = False
    found_solution = None
    
    while heap:
        if time.time() > deadline:
            break
            
        md, rem, min_d, path = heapq.heappop(heap)
        
        # If the current max denominator is already >= best_max, we can stop
        # because we are searching for < best_max
        if md >= best_max:
            break
            
        # If we've seen this state (rem, min_d) with a smaller or equal md, skip
        if visited.get((rem, min_d), float('inf')) < md:
            continue
            
        if rem == 0:
            found = True
            found_solution = path
            break
            
        # Limit depth to avoid infinite loops or very long paths
        if len(path) > 15:
            continue
            
        # Next denominator k must satisfy:
        # 1. k > min_d
        # 2. 1/k < rem  => k > 1/rem
        # 3. k < best_max (to keep max_denom < best_max)
        
        k_min = max(min_d, int(1 / rem) + 1)
        k_max = best_max - 1
        
        if k_min > k_max:
            continue
            
        for k in range(k_min, k_max + 1):
            new_rem = rem - Fraction(1, k)
            if new_rem < 0:
                continue
                
            new_md = max(md, k)
            new_min_d = k + 1
            new_path = path + [k]
            
            # Prune if we've seen a better max_denom for this state
            if new_md >= visited.get((new_rem, new_min_d), float('inf')):
                continue
                
            visited[(new_rem, new_min_d)] = new_md
            heapq.heappush(heap, (new_md, new_rem, new_min_d, new_path))
            
    if found and found_solution:
        # Verify the solution
        s = sum(Fraction(1, d) for d in found_solution)
        if s == target and len(set(found_solution)) == len(found_solution) and all(1 <= d <= 10000000 for d in found_solution):
            return found_solution
            
    return best_solution
