import random
import time

def solve_turnpike(diffs):
    if not diffs:
        return [0]
    
    D = set(diffs)
    L = max(D)
    D.remove(L)
    
    P = [0, L]
    
    def backtrack(P, D):
        if not D:
            return P[:]
        
        d_max = max(D)
        
        candidates = set()
        for p in P:
            x1 = p + d_max
            x2 = p - d_max
            if 0 <= x1 <= L and x1 not in P:
                candidates.add(x1)
            if 0 <= x2 <= L and x2 not in P:
                candidates.add(x2)
                
        # Optimization: Try candidates that are "likely"
        # Just iterate
        for x in candidates:
            valid = True
            new_diffs = []
            for p in P:
                d = abs(x - p)
                if d not in D:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                new_D = set(D)
                for d in new_diffs:
                    new_D.remove(d)
                
                res = backtrack(P + [x], new_D)
                if res is not None:
                    return res
        
        return None

    return backtrack(P, D)

def construct(n_rows, k):
    row_len = k + 1
    K = k * (k + 1) // 2
    total_diffs = n_rows * K
    
    M_min = total_diffs
    
    best_rows = None
    best_M = float('inf')
    
    start_time = time.time()
    max_time = 110
    
    # Start from a high M and go down
    # 112 is the counting bound.
    # We can start at 112.
    for M in range(112, M_min - 1, -1):
        if time.time() - start_time > max_time:
            break
            
        # If we already have a solution with M, we don't need to find another one for this M
        # But we do need to find one to prove M is feasible.
        # If M < best_M, we are trying to improve.
        
        # If M >= best_M, we don't care if we find one, as we already have a better one.
        # Wait, if we start at 112 and find a solution, best_M becomes 112.
        # Then we try 111. If we find one, best_M becomes 111.
        # So we only need to find a solution if M < best_M?
        # No, we need to find a solution to update best_M.
        # If M >= best_M, we can skip?
        # Yes, if we have a solution with max entry 100, we don't need to find a solution with max entry 105.
        # So we can skip M if M >= best_M.
        
        if M >= best_M:
            continue
            
        # Try to find a solution with max difference <= M
        found = False
        attempts = 500
        
        for _ in range(attempts):
            if time.time() - start_time > max_time:
                break
                
            diffs_pool = random.sample(range(1, M + 1), total_diffs)
            random.shuffle(diffs_pool)
            groups = []
            for i in range(n_rows):
                groups.append(diffs_pool[i*K : (i+1)*K])
            
            rows = []
            valid = True
            for g in groups:
                pts = solve_turnpike(g)
                if pts is None:
                    valid = False
                    break
                pts.sort()
                if pts[0] != 0:
                    valid = False
                    break
                rows.append(pts)
            
            if valid:
                max_entry = max(max(r) for r in rows)
                if max_entry <= M:
                    best_rows = [r[:] for r in rows]
                    best_M = max_entry
                    found = True
                    break
        
        # If we didn't find a solution, we continue to next M (lower)
        # This is correct because if M is infeasible, M-1 is also infeasible.
        # But we might have just been unlucky.
        # However, with 500 attempts, it's unlikely to be unlucky if a solution exists.
        
    return best_rows
