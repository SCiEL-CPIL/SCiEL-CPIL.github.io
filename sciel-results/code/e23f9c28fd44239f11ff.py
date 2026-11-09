import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _compute_lcm_range(n):
    result = 1
    for i in range(1, n + 1):
        result = _lcm(result, i)
    return result

def _has_perfect_matching(m, L, n=12):
    """Check if there's a perfect matching in interval (m, m+L] for indices 1..n."""
    adj = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        # (m+j) % i == 0 => j ≡ (-m) mod i
        r = (-m) % i
        if r == 0:
            r = i
        for j in range(r, L + 1, i):
            mask |= (1 << (j - 1))
        adj[i] = mask
        if mask == 0:
            return False
    
    # Sort indices by fewest options first for better DFS performance
    order = sorted(range(1, n + 1), key=lambda i: bin(adj[i]).count('1'))
    
    match_pos = [-1] * L
    
    def dfs(i, visited):
        candidates = adj[i]
        while candidates:
            low_bit = candidates & (-candidates)
            j_idx = low_bit.bit_length() - 1
            candidates ^= low_bit
            if visited & low_bit:
                continue
            visited |= low_bit
            if match_pos[j_idx] == -1 or dfs(match_pos[j_idx], visited):
                match_pos[j_idx] = i
                return True
        return False
    
    for i in order:
        if not dfs(i, 0):
            return False
    return True

def _compute_f(m, n=12, max_L=144):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_1..b_n with i|b_i."""
    # Binary search since property is monotone in L
    lo, hi = 1, max_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 seconds, leaving 5s safety margin
    
    n = 12
    period = _compute_lcm_range(n)  # 27720
    
    best_m = 12
    best_f = _compute_f(best_m, n)
    
    m = 0
    while m < period:
        if time.time() > deadline:
            break
        
        f_val = _compute_f(m, n)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            if best_f >= 144:
                break
        
        m += 1
    
    return int(best_m)
