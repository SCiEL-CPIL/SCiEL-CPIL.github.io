"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

N = 12
MAX_M = 1000000
L_MAX = 144

def has_matching(m, L):
    """Check if there exist distinct b_1,...,b_12 in (m, m+L] with i|b_i."""
    candidates = []
    for i in range(1, N+1):
        start = (m // i + 1) * i
        mults = []
        x = start
        end = m + L
        while x <= end:
            mults.append(x)
            x += i
        candidates.append(mults)
    
    match_r = {}
    def dfs(u, visited):
        for v in candidates[u]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_r or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    for u in range(1, N+1):
        visited = set()
        if not dfs(u, visited):
            return False
    return True

def compute_f(m):
    for L in range(1, L_MAX+1):
        if has_matching(m, L):
            return L
    return L_MAX

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_m = 12
    best_f = compute_f(12)
    
    # Focused candidate set
    candidates = set()
    
    # Small m
    for m in range(0, 300):
        candidates.add(m)
    
    # Near multiples of 27720
    for k in range(0, MAX_M // 27720 + 1):
        base = k * 27720
        for offset in range(-30, 31):
            m = base + offset
            if 0 <= m <= MAX_M:
                candidates.add(m)
    
    candidates = sorted(candidates)
    
    for m in candidates:
        if time.time() > deadline:
            break
        f_val = compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            if best_f == L_MAX:
                break
    
    return int(best_m)

# EVOLVE-BLOCK-END
