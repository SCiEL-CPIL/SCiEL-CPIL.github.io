import time
import math

PERIOD = 27720
N = 12
MAX_M = 1000000
MAX_L = 144

def has_valid_set(m_mod, L):
    # adj[i] is a list of valid right-side nodes (offsets 1..L) for left node i
    # i ranges 1..12
    adj = [[] for _ in range(N)]
    for i in range(1, N + 1):
        # We need (m_mod + j) % i == 0
        # j % i == (-m_mod) % i
        rem = (-m_mod) % i
        # j can be rem, rem+i, rem+2i...
        # But j must be in 1..L
        # Note: if rem is 0, j can be i, 2i... (since j>=1)
        # If rem > 0, j can be rem, rem+i...
        
        # Actually, simpler:
        # start = rem
        # if start == 0: start = i  # because j >= 1
        # for j in range(start, L+1, i):
        #    adj[i-1].append(j)
        
        # Let's just loop j from 1 to L? No, that's slow if done naively for each i.
        # But L is small (<=144).
        # Let's precompute valid offsets for each i and m_mod?
        # m_mod changes in outer loop.
        
        # Efficient way:
        if rem == 0:
            first = i
        else:
            first = rem
            
        if first <= L:
            adj[i-1] = list(range(first, L + 1, i))
            
    # Bipartite Matching (Hopcroft-Karp or DFS)
    # Since N=12 is small, DFS with backtracking (Kuhn's algorithm) is very fast.
    match_r = [-1] * (L + 1) # match_r[j] = i matched to j, or -1
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v] - 1, visited):
                match_r[v] = u + 1
                return True
        return False

    count = 0
    for u in range(N):
        visited = [False] * (L + 1)
        if dfs(u, visited):
            count += 1
            
    return count == N

def f_12(m_mod):
    lo, hi = 12, MAX_L
    while lo < hi:
        mid = (lo + hi) // 2
        if has_valid_set(m_mod, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    best_r = 0
    best_val = 0
    
    # Iterate through all residues
    for r in range(PERIOD):
        if time.time() - start_time > 85:
            break
        
        val = f_12(r)
        if val > best_val:
            best_val = val
            best_r = r
            
    # Map best_r to largest m
    if best_r > MAX_M:
        m = best_r
    else:
        k_max = (MAX_M - best_r) // PERIOD
        m = k_max * PERIOD + best_r
        
    if m < 0: m = 0
    if m > MAX_M: m = MAX_M
        
    return int(m)
