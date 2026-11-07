import time
import math
from functools import reduce
from collections import deque

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _dinic_maxflow(adj, n_left, n_right):
    """Dinic max-flow for bipartite matching.
    Nodes: 0=source, 1..n_left=left, n_left+1..n_left+n_right=right, n_left+n_right+1=sink
    """
    n_nodes = n_left + n_right + 2
    source = 0
    sink = n_nodes - 1
    
    # Build adjacency list for flow network
    # Each edge: [to, capacity, reverse_index]
    graph = [[] for _ in range(n_nodes)]
    
    def add_edge(u, v, cap):
        graph[u].append([v, cap, len(graph[v])])
        graph[v].append([u, 0, len(graph[u]) - 1])
    
    # Source to left nodes
    for i in range(n_left):
        add_edge(source, 1 + i, 1)
    
    # Left to right nodes
    for i in range(n_left):
        for v_idx in adj[i]:
            add_edge(1 + i, 1 + n_left + v_idx, 1)
    
    # Right nodes to sink
    for j in range(n_right):
        add_edge(1 + n_left + j, sink, 1)
    
    flow = 0
    level = [0] * n_nodes
    ptr = [0] * n_nodes
    
    from collections import deque as dq
    
    def bfs():
        for i in range(n_nodes):
            level[i] = -1
        level[source] = 0
        q = dq([source])
        while q:
            u = q.popleft()
            for e in graph[u]:
                if e[1] > 0 and level[e[0]] == -1:
                    level[e[0]] = level[u] + 1
                    q.append(e[0])
        return level[sink] != -1
    
    def dfs(u, pushed):
        if u == sink:
            return pushed
        while ptr[u] < len(graph[u]):
            e = graph[u][ptr[u]]
            v, cap, rev = e[0], e[1], e[2]
            if cap > 0 and level[v] == level[u] + 1:
                tr = dfs(v, min(pushed, cap))
                if tr > 0:
                    e[1] -= tr
                    graph[v][rev][1] += tr
                    return tr
            ptr[u] += 1
        return 0
    
    while bfs():
        for i in range(n_nodes):
            ptr[i] = 0
        while True:
            pushed = dfs(source, float('inf'))
            if pushed == 0:
                break
            flow += pushed
            if flow == n_left:
                return flow
    
    return flow

def _can_match(m, L, n=12):
    if L < n:
        return False
        
    val_to_idx = {}
    right_vals = []
    adj_idx = [[] for _ in range(n)]
    
    for i in range(n):
        idx = i + 1
        start_k = m // idx + 1
        end_k = (m + L) // idx
        
        if start_k > end_k:
            return False
            
        for k in range(start_k, end_k + 1):
            v = k * idx
            if v not in val_to_idx:
                val_to_idx[v] = len(right_vals)
                right_vals.append(v)
            adj_idx[i].append(val_to_idx[v])
            
    if len(right_vals) < n:
        return False
        
    return _dinic_maxflow(adj_idx, n, len(right_vals)) == n

def _f(m, n=12):
    # Binary search smallest L in [12, 144]
    # L=144 is always feasible
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
    deadline = start_time + 80  # 10s safety margin
    
    # Calculate LCM of 1..12
    p = reduce(_lcm, range(1, 13))
    
    # Check all m in [0, p-1] since f(12, m) is periodic with period p
    best_m = 0
    best_f = _f(0)
    
    for m in range(1, p):
        if time.time() > deadline:
            break
        f_val = _f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
        # Early termination: if we've found the maximum possible value
        if best_f >= 144:
            break

    # The best m in [0, p-1] is equivalent to any m + k*p.
    # We return the smallest such m.
    return int(best_m)
