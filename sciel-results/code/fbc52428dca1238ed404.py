"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def _build_edge_index():
    n = P['n']
    edge_idx = {}
    for i, (u, v) in enumerate(P['edge_order']):
        edge_idx[(min(u,v), max(u,v))] = i
    return edge_idx

def _cycle_score(cycle, edge_idx):
    n = len(cycle)
    max_idx = 0
    for i in range(n):
        u, v = cycle[i], cycle[(i+1) % n]
        key = (min(u,v), max(u,v))
        idx = edge_idx[key]
        if idx > max_idx:
            max_idx = idx
    return max_idx

def _find_hamilton_cycle_k(edge_idx, k):
    """Find a Hamilton cycle using only edges with index <= k-1 (0-based).
    Returns the cycle as a list of vertices, or None if not found."""
    n = P['n']
    # Build adjacency list using only edges with 0-based index < k
    adj = [set() for _ in range(n)]
    for i, (u, v) in enumerate(P['edge_order']):
        if i >= k:
            break
        adj[u].add(v)
        adj[v].add(u)
    
    # Check degree constraint: every vertex must have degree >= 2
    for i in range(n):
        if len(adj[i]) < 2:
            return None
    
    # DFS to find Hamilton cycle
    # Fix vertex 0 as start, and to avoid symmetry, fix the second vertex
    # to be the smallest neighbor of 0 (to reduce search space)
    if not adj[0]:
        return None
    
    # Try different second vertices for diversity
    second_candidates = sorted(adj[0])
    
    for second in second_candidates:
        visited = [False] * n
        visited[0] = True
        visited[second] = True
        path = [0, second]
        
        # DFS with pruning
        def dfs():
            if len(path) == n:
                # Check if last vertex connects back to 0
                if 0 in adj[path[-1]]:
                    return True
                return False
            
            last = path[-1]
            # Sort neighbors for deterministic exploration
            neighbors = sorted(adj[last])
            for nxt in neighbors:
                if not visited[nxt]:
                    # Pruning: check if removing nxt would disconnect unvisited
                    # Simple degree check: if a vertex (other than last and nxt)
                    # has only one unvisited neighbor (which is nxt), and nxt is
                    # not the last one to be visited, it might be a problem.
                    # Simpler: just check basic connectivity via degree.
                    
                    # Check: after visiting nxt, does any unvisited vertex (not nxt)
                    # have 0 available neighbors in unvisited set?
                    # This is expensive; use a lighter check.
                    
                    visited[nxt] = True
                    path.append(nxt)
                    
                    if dfs():
                        return True
                    
                    path.pop()
                    visited[nxt] = False
            return False
        
        if dfs():
            return path[:]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    edge_idx = _build_edge_index()
    n = P['n']
    total_edges = n * (n - 1) // 2  # 276
    
    # Binary search on k (1-based max arrival index)
    # We want the smallest k such that a Hamilton cycle exists using edges with 0-based index < k
    # i.e., max edge index in the cycle (0-based) is k-1, so 1-based score is k
    
    lo = n  # minimum possible: n edges, each needs at least... well, at least n
    hi = total_edges
    
    best_cycle = list(range(n))  # fallback: identity
    best_score = _cycle_score(best_cycle, edge_idx)
    
    # First, try to find the minimum k via binary search
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        cycle = _find_hamilton_cycle_k(edge_idx, mid)
        if cycle is not None:
            hi = mid
            # Update best if this is better
            sc = _cycle_score(cycle, edge_idx)
            if sc < best_score:
                best_score = sc
                best_cycle = cycle[:]
        else:
            lo = mid + 1
    
    # Now lo == hi is the minimum k. Let's verify and get the cycle.
    if time.time() < deadline - 1.0:
        cycle = _find_hamilton_cycle_k(edge_idx, lo)
        if cycle is not None:
            sc = _cycle_score(cycle, edge_idx)
            if sc < best_score:
                best_score = sc
                best_cycle = cycle[:]
    
    # Try to improve with 2-opt style local search
    if time.time() < deadline - 2.0:
        cycle = best_cycle[:]
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(n):
                if time.time() > deadline - 2.0:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline - 2.0:
                        break
                    # Reverse segment i..j
                    new_cycle = cycle[:i] + cycle[i:j+1][::-1] + cycle[j+1:]
                    sc = _cycle_score(new_cycle, edge_idx)
                    if sc < best_score:
                        best_score = sc
                        best_cycle = new_cycle[:]
                        cycle = new_cycle[:]
                        improved = True
    
    return best_cycle
# EVOLVE-BLOCK-END
