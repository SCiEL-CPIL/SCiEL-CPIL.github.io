import math
import time

MAX_MOD = 128
DEADLINE_MARGIN = 5.0
CSP_TIME_LIMIT = 2.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _solve_csp(moduli, time_limit):
    if not moduli:
        return []
    
    n = len(moduli)
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                gcds[i][j] = math.gcd(moduli[i], moduli[j])
    
    domains = [list(range(m)) for m in moduli]
    assignment = [None] * n
    start_time = time.time()
    
    def is_valid(i, r):
        for j in range(n):
            if assignment[j] is not None and i != j:
                if not _disjoint(r, moduli[i], assignment[j], moduli[j]):
                    return False
        return True

    def backtrack():
        if time.time() - start_time > time_limit:
            return False, "timeout"
            
        # MRV
        min_size = float('inf')
        min_idx = -1
        for k in range(n):
            if assignment[k] is None:
                if len(domains[k]) < min_size:
                    min_size = len(domains[k])
                    min_idx = k
        
        if min_idx == -1:
            return True, "success"
            
        for r in domains[min_idx]:
            if is_valid(min_idx, r):
                assignment[min_idx] = r
                
                saved_domains = []
                consistent = True
                for k in range(n):
                    if assignment[k] is None and k != min_idx:
                        old_domain = domains[k][:]
                        new_domain = [rk for rk in old_domain if _disjoint(r, moduli[min_idx], rk, moduli[k])]
                        domains[k] = new_domain
                        saved_domains.append((k, old_domain))
                        if not new_domain:
                            consistent = False
                            break
                        
                if consistent:
                    res, status = backtrack()
                    if res:
                        return True, status
                
                for k, old_domain in saved_domains:
                    domains[k] = old_domain
                
                assignment[min_idx] = None
        
        return False, "failed"

    res, status = backtrack()
    if res:
        return [(assignment[i], moduli[i]) for i in range(n)]
    else:
        return None

def construct():
    t0 = time.time()
    deadline = t0 + 60.0 - DEADLINE_MARGIN
    
    all_mods = list(range(2, MAX_MOD + 1))
    
    best_result = None
    best_len = 0
    
    # Strategy: Remove smallest moduli first
    # This is likely to find a solution quickly.
    
    current_set = all_mods[:]
    
    while current_set:
        if time.time() > deadline:
            break
            
        # Solve CSP
        result = _solve_csp(current_set, CSP_TIME_LIMIT)
        
        if result:
            if len(result) > best_len:
                best_len = len(result)
                best_result = result
            break
        
        # Remove the smallest modulus
        m_to_remove = min(current_set)
        current_set.remove(m_to_remove)
        
    if best_result is None:
        return [[0, 2]]
        
    return [[r, m] for r, m in best_result]
