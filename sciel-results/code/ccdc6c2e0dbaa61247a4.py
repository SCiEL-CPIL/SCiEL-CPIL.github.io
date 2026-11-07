import time
import math
from functools import reduce
from collections import deque

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _hopcroft_karp(adj, n_left, n_right):
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left
    
    def bfs():
        q = deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    return matching

def _can_match(m, L, n=12):
    if L < n:
        return False
        
    # Map k (1..L) to index in right side
    # Right side nodes are k in 1..L
    # adj[i] will contain k's such that i+1 divides m+k
    
    adj = [[] for _ in range(n)]
    
    for i in range(n):
        idx = i + 1
        # k must satisfy m + k is divisible by idx
        # k = m + k - m
        # m + k ≡ 0 (mod idx) => k ≡ -m (mod idx)
        rem = (-m) % idx
        
        # Find all k in [1, L] such that k ≡ rem (mod idx)
        # If rem == 0, k must be multiple of idx. First is idx (if idx <= L).
        # If rem > 0, first is rem (if rem <= L).
        
        if rem == 0:
            start_k = idx
        else:
            start_k = rem
            
        if start_k > L:
            return False
            
        for k in range(start_k, L + 1, idx):
            adj[i].append(k - 1) # 0-indexed for right side? 
            # Let's use k-1 as the node id for right side to keep 0..L-1
            # But we need to ensure distinctness. 
            # The nodes are k values. 
            # Using k-1 as index is fine as long as it's unique.
            
    # Right side size is L (nodes 0 to L-1 representing k=1 to L)
    # But not all k are connected.
    # We can just use n_right = L
    # And adj[i] contains indices k-1
    
    # Optimization: if any adj[i] is empty, return False
    for i in range(n):
        if not adj[i]:
            return False
            
    return _hopcroft_karp(adj, n, L) == n

def _f(m, n=12):
    lo, hi = 12, 143
    if not _can_match(m, hi, n):
        return 144
        
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    p = reduce(_lcm, range(1, 13))
    
    best_m = 0
    best_f = _f(0)
    
    for m in range(1, p):
        if time.time() > deadline:
            break
        f_val = _f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m

    return int(best_m)
