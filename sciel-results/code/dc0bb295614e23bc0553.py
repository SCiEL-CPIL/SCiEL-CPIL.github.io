import time
import sys

def _dinic_max_flow(capacity, source, sink, n):
    """
    Dinic's algorithm for max flow.
    capacity: 2D list (n x n) of capacities.
    Returns the max flow value.
    """
    # Build adjacency list
    graph = [[] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if capacity[u][v] > 0:
                graph[u].append(v)
                graph[v].append(u)
                
    # We need to track residual capacities and the graph structure.
    # Standard Dinic uses a graph with edges.
    # Let's implement a standard Dinic with edge objects or lists.
    # Since n is small (<= 12 + 144 + 2 = 158), a simple list-based approach is fine.
    
    # Re-initialize for Dinic proper
    # Edge: [to, rev, cap]
    g = [[] for _ in range(n)]
    
    def add_edge(u, v, cap):
        g[u].append([v, len(g[v]), cap])
        g[v].append([u, len(g[u])-1, 0])
        
    for u in range(n):
        for v in range(n):
            if capacity[u][v] > 0:
                add_edge(u, v, capacity[u][v])

    flow = 0
    INF = 10**9
    
    while True:
        # BFS to find levels
        level = [-1] * n
        level[source] = 0
        q = [source]
        for u in q:
            for e in g[u]:
                if e[2] > 0 and level[e[0]] < 0:
                    level[e[0]] = level[u] + 1
                    q.append(e[0])
                    
        if level[sink] < 0:
            break
            
        # DFS to find blocking flow
        it = [0] * n
        
        def dfs(u, f):
            if u == sink:
                return f
            for i in range(it[u], len(g[u])):
                it[u] = i
                e = g[u][i]
                if e[2] > 0 and level[u] < level[e[0]]:
                    d = dfs(e[0], min(f, e[2]))
                    if d > 0:
                        e[2] -= d
                        g[e[0]][e[1]][2] += d
                        return d
            return 0
            
        while True:
            f = dfs(source, INF)
            if f == 0:
                break
            flow += f
            
    return flow

def _can_match(m, L, n=12):
    """
    Check if (m, m+L] contains distinct b_1...b_n with i | b_i.
    Uses Max Flow (Dinic).
    """
    # Nodes:
    # 0: Source
    # 1..12: Indices i
    # 13..13+L-1: Values v in (m, m+L]
    # 13+L: Sink
    
    N = 1 + n + L + 1
    source = 0
    sink = N - 1
    
    # Initialize capacity matrix (sparse graph construction is better, but N is small)
    # However, building a full NxN matrix for every call might be slow if L is large.
    # N is at most 1 + 12 + 144 + 1 = 158. 158x158 is ~25k entries. This is fine.
    # But we can just build the graph directly in _dinic_max_flow if we pass edges.
    # Let's modify _dinic_max_flow to take edges or just build the matrix here.
    # Actually, let's just build the matrix and pass it.
    
    cap = [[0] * N for _ in range(N)]
    
    # Source to indices
    for i in range(1, n + 1):
        cap[source][i] = 1
        
    # Indices to values
    # Value v corresponds to node index: 1 + n + (v - (m+1))
    # Let's precompute node indices for values
    # v ranges from m+1 to m+L
    
    # Optimization: Precompute divisibility?
    # For each i, check which v in range are divisible by i.
    
    for i in range(1, n + 1):
        for v in range(m + 1, m + L + 1):
            if v % i == 0:
                node_v = 1 + n + (v - (m + 1))
                cap[i][node_v] = 1
                
    # Values to sink
    for v in range(m + 1, m + L + 1):
        node_v = 1 + n + (v - (m + 1))
        cap[node_v][sink] = 1
        
    # Run Dinic
    max_flow = _dinic_max_flow(cap, source, sink, N)
    
    return max_flow == n

def _f_val(m, n=12):
    """
    Find the least positive L such that (m, m+L] contains a valid matching.
    """
    # Max L is stated to be at most 144
    for L in range(1, 145):
        if _can_match(m, L, n):
            return L
    return 144 # Fallback, shouldn't happen

def construct():
    """
    Find m in 0..1000000 that maximizes f(12, m).
    """
    # The function f(12, m) is periodic with period lcm(1..12) = 27720
    # So we only need to search m in [0, 27720)
    K = 27720
    best_m = 0
    best_L = 0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds margin (limit is 90s)
    
    for m in range(K):
        if time.time() > deadline:
            break
        L = _f_val(m)
        if L > best_L:
            best_L = L
            best_m = m
            # If we hit the theoretical max, we can stop early
            if best_L == 144:
                break
                
    return int(best_m)
