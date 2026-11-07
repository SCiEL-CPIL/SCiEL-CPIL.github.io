import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    best_width = 100  # Initial guess, will be refined
    best_sol = None
    
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    scan_limit = 200000
    
    for L in range(1, scan_limit + 1):
        if time.time() > deadline:
            break
            
        if best_width == n - 1:
            break
            
        upper_bound = L + best_width - 1
        
        candidates = []
        valid = True
        for d in divisors:
            # Smallest multiple of d >= L
            k = (L + d - 1) // d
            if k < 1:
                k = 1
            m = k * d
            
            if m > upper_bound:
                valid = False
                break
                
            mults = []
            while m <= upper_bound:
                mults.append(m)
                m += d
            candidates.append(mults)
            
        if not valid:
            continue
            
        # Backtracking search for a distinct assignment
        # We want to find ANY assignment, as any assignment in this window
        # has span <= best_width - 1, which is an improvement.
        
        def find_solution(idx, used, path):
            if idx == n:
                return path[:]
            
            # Heuristic: try values in order
            for val in candidates[idx]:
                if val in used:
                    continue
                used.add(val)
                path.append(val)
                res = find_solution(idx + 1, used, path)
                if res:
                    return res
                path.pop()
                used.remove(val)
            return None
            
        sol = find_solution(0, set(), [])
        if sol:
            w = max(sol) - min(sol)
            if w < best_width:
                best_width = w
                best_sol = sol
            
    if best_sol is None:
        # Fallback: just return the divisors (valid, but wide span)
        # Or find the first valid greedy solution
        for L in range(1, 10000):
            x = [((L + d - 1) // d) * d for d in divisors]
            if len(set(x)) == n:
                best_sol = x
                break
        if best_sol is None:
            best_sol = divisors[:]
            
    return [int(x) for x in best_sol]
