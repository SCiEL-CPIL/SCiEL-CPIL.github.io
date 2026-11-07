import time
import math
from functools import reduce

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _dinic_max_flow(n_left, n_right, adj, deadline_check=None):
    """
    Dinic's algorithm for bipartite matching.
    n_left: number of left nodes (indices 1..12)
    n_right: number of right nodes (candidate values)
    adj: list of lists, adj[i] = list of right node indices connected to left node i
    Returns max flow (which equals max matching size).
    """
    # Build flow network: source=0, left nodes 1..n_left, right nodes n_left+1..n_left+n_right, sink=n_left+n_right+1
    src = 0
    sink = n_left + n_right + 1
    N = sink + 1
    
    # Adjacency list with edge indices: graph[u] = list of (v, cap, rev_idx)
    graph = [[] for _ in range(N)]
    
    def add_edge(u, v, cap):
        graph[u].append([v, cap, len(graph[v])])
        graph[v].append([u, 0, len(graph[u]) - 1])
    
    # Source to left nodes
    for i in range(n_left):
        add_edge(src, 1 + i, 1)
    
    # Left to right
    for i in range(n_left):
        for j in adj[i]:
            add_edge(1 + i, 1 + n_left + j, 1)
    
    # Right to sink
    for j in range(n_right):
        add_edge(1 + n_left + j, sink, 1)
    
    flow = 0
    while True:
        # BFS: find level graph
        level = [-1] * N
        level[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for edge in graph[u]:
                v, cap, rev = edge
                if cap > 0 and level[v] == -1:
                    level[v] = level[u] + 1
                    queue.append(v)
        
        if level[sink] == -1:
            break
        
        # DFS: find blocking flow
        # Use iterative approach with explicit stack to avoid recursion depth issues
        ptr = [0] * N  # current edge pointer for each node
        
        def dfs(u, pushed):
            if pushed == 0:
                return 0
            if u == sink:
                return pushed
            while ptr[u] < len(graph[u]):
                edge = graph[u][ptr[u]]
                v, cap, rev = edge
                if cap > 0 and level[v] == level[u] + 1:
                    tr = dfs(v, min(pushed, cap))
                    if tr > 0:
                        edge[1] -= tr
                        graph[v][rev][1] += tr
                        return tr
                ptr[u] += 1
            return 0
        
        while True:
            pushed = dfs(src, float('inf'))
            if pushed == 0:
                break
            flow += pushed
            if deadline_check and deadline_check():
                return flow  # Return partial result if deadline exceeded
        
        if deadline_check and deadline_check():
            return flow
    
    return flow

def _can_match(m, L, n=12):
    # Check if (m, m+L] contains distinct b_1..b_n with i | b_i
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        r = ((m // i) + 1) * i
        v = r
        end = m + L
        while v <= end:
            adj[i - 1].append(v)
            v += i
        if not adj[i - 1]:
            return False

    # Map right side values to indices for efficient matching
    val_to_idx = {}
    right_vals = []
    for i in range(n):
        for v in adj[i]:
            if v not in val_to_idx:
                val_to_idx[v] = len(right_vals)
                right_vals.append(v)
    
    # Convert adj to index-based
    adj_idx = [[val_to_idx[v] for v in adj[i]] for i in range(n)]
    
    n_right = len(right_vals)
    
    # Use Dinic's algorithm
    max_flow = _dinic_max_flow(n, n_right, adj_idx)
    
    return max_flow == n

def _f(m, n=12):
    # Binary search smallest L in [1, 144]
    lo, hi = 1, 144
    if not _can_match(m, hi, n):
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Calculate LCM of 1..12
    p = reduce(_lcm, range(1, 13))
    
    # Generate critical m in [0, p-1]
    # m is critical if m+1 is a multiple of some i in 1..12
    critical_ms = set()
    for i in range(1, 13):
        for k in range(1, p // i + 1):
            m_val = k * i - 1
            if 0 <= m_val < p:
                critical_ms.add(m_val)
            
    # Also include 0 just in case
    critical_ms.add(0)
    
    # Sort for deterministic order
    critical_list = sorted(critical_ms)
    
    best_m = 0
    best_f = _f(0)
    
    for m in critical_list:
        if time.time() > deadline:
            break
        f_val = _f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m

    return int(best_m)
