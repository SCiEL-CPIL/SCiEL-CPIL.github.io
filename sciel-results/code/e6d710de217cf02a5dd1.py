import math
import time

def _has_perfect_matching(m, L, n=12):
    adj = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, L + 1):
            if (m + j) % i == 0:
                mask |= (1 << (j - 1))
        adj[i] = mask
        
    for i in range(1, n + 1):
        if adj[i] == 0:
            return False
            
    match_r = [0] * L
    
    def dfs(i, visited_mask):
        candidates = adj[i] & ~visited_mask
        while candidates:
            low_bit = candidates & -candidates
            j_idx = low_bit.bit_length() - 1
            candidates ^= low_bit
            
            if match_r[j_idx] == 0 or dfs(match_r[j_idx], visited_mask | low_bit):
                match_r[j_idx] = i
                return True
        return False
        
    count = 0
    for i in range(1, n + 1):
        if dfs(i, 0):
            count += 1
    return count == n

def _compute_f(m, n=12, max_L=144):
    for L in range(n, max_L + 1):
        if _has_perfect_matching(m, L, n):
            return L
    return max_L

def construct():
    lcm_val = 1
    for i in range(1, 13):
        lcm_val = lcm_val * i // math.gcd(lcm_val, i)
        
    best_m = 0
    best_L = 0
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    for m in range(lcm_val):
        if time.time() > deadline:
            break
            
        L = _compute_f(m)
        if L > best_L:
            best_L = L
            best_m = m
            if best_L == 144:
                break
                
    return int(best_m)

# EVOLVE-BLOCK-START
import math
import time

def _has_perfect_matching(adj, L, n=12):
    for i in range(1, n + 1):
        if adj[i] == 0:
            return False
    
    match_r = [0] * L
    
    def dfs(i, visited_mask):
        candidates = adj[i] & ~visited_mask
        while candidates:
            low_bit = candidates & -candidates
            j_idx = low_bit.bit_length() - 1
            candidates ^= low_bit
            
            if match_r[j_idx] == 0 or dfs(match_r[j_idx], visited_mask | low_bit):
                match_r[j_idx] = i
                return True
        return False
        
    count = 0
    for i in range(1, n + 1):
        if dfs(i, 0):
            count += 1
    return count == n

def _build_adj(m, L, n=12):
    adj = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, L + 1):
            if (m + j) % i == 0:
                mask |= (1 << (j - 1))
        adj[i] = mask
    return adj

def _compute_f(m, n=12, max_L=144):
    lo, hi = n, max_L
    result = max_L
    while lo <= hi:
        mid = (lo + hi) // 2
        adj = _build_adj(m, mid, n)
        if _has_perfect_matching(adj, mid, n):
            result = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return result

def construct():
    lcm_val = 1
    for i in range(1, 13):
        lcm_val = lcm_val * i // math.gcd(lcm_val, i)
        
    best_m = 0
    best_L = 0
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    for m in range(lcm_val):
        if time.time() > deadline:
            break
        L = _compute_f(m)
        if L > best_L:
            best_L = L
            best_m = m
            if best_L == 144:
                break
                
    return int(best_m)
# EVOLVE-BLOCK-END
