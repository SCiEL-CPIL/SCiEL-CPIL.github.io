"""Find Hamilton cycle with minimal max edge arrival index in pinned edge order."""
import time
import random
import sys

# EVOLVE-BLOCK-START
_P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = _P['n']
EDGE_ORDER = _P['edge_order']
NUM_EDGES = len(EDGE_ORDER)

# Precompute edge arrival index (1-based)
_edge_arrival = {}
for i, e in enumerate(EDGE_ORDER):
    a, b = e
    key = (min(a, b), max(a, b))
    _edge_arrival[key] = i + 1

def _cycle_score(cycle):
    """Score = max 1-based arrival index of cycle's edges."""
    n = len(cycle)
    max_idx = 0
    for i in range(n):
        a = cycle[i]
        b = cycle[(i + 1) % n]
        key = (min(a, b), max(a, b))
        idx = _edge_arrival.get(key, 9999)
        if idx > max_idx:
            max_idx = idx
    return max_idx

def _find_hamilton_in_prefix(k, deadline):
    """Find a Hamilton cycle using only first k edges. Returns cycle or None."""
    # Build adjacency from first k edges
    adj = [set() for _ in range(N)]
    for i in range(k):
        a, b = EDGE_ORDER[i]
        adj[a].add(b)
        adj[b].add(a)
    
    # Check all vertices have degree >= 2
    for v in range(N):
        if len(adj[v]) < 2:
            return None
    
    # Backtracking with memoization (bitmask)
    # Try all starting vertices, but fix vertex 0 as start to reduce symmetry
    # Actually, let's just try to find any Hamilton cycle
    
    sys.setrecursionlimit(10000)
    
    best_cycle = None
    deadline_check = 0
    
    # Use vertex 0 as fixed start (any Hamilton cycle can be rotated to start at 0)
    start = 0
    # Try each neighbor of 0 as second vertex
    for second in sorted(adj[start]):
        if time.time() > deadline:
            break
        # DFS: path = [start, second, ...], need to visit all N vertices
        # State: (current_vertex, visited_mask)
        # Use iterative DFS with backtracking
        
        path = [start, second]
        visited = {start, second}
        
        def dfs(path, visited):
            nonlocal best_cycle, deadline_check
            deadline_check += 1
            if deadline_check % 10000 == 0 and time.time() > deadline:
                return None
            if len(path) == N:
                # Check if last vertex connects back to start
                last = path[-1]
                if start in adj[last]:
                    return list(path)
                return None
            cur = path[-1]
            # Try neighbors not yet visited, ordered by degree (least first)
            neighbors = sorted(adj[cur], key=lambda x: len(adj[x]))
            for nb in neighbors:
                if nb not in visited:
                    path.append(nb)
                    visited.add(nb)
                    result = dfs(path, visited)
                    if result is not None:
                        return result
                    path.pop()
                    visited.discard(nb)
            return None
        
        result = dfs(path, visited)
        if result is not None:
            best_cycle = result
            break
    
    return best_cycle

def _find_optimal_cycle(deadline):
    """Binary search + DFS to find cycle with minimal max arrival index."""
    # Lower bound: need at least N edges (the cycle itself)
    # Upper bound: NUM_EDGES
    lo, hi = N, NUM_EDGES
    best_cycle = None
    best_score = NUM_EDGES + 1
    
    # First, try to find a good upper bound quickly
    # Try the full graph
    cycle = _find_hamilton_in_prefix(NUM_EDGES, deadline)
    if cycle:
        s = _cycle_score(cycle)
        best_cycle = cycle
        best_score = s
        hi = s
    
    # Binary search for minimal k
    while lo < hi and time.time() < deadline:
        mid = (lo + hi) // 2
        cycle = _find_hamilton_in_prefix(mid, deadline)
        if cycle is not None:
            s = _cycle_score(cycle)
            if s < best_score:
                best_score = s
                best_cycle = cycle
            hi = mid
        else:
            lo = mid + 1
    
    # If binary search didn't find anything, use full graph result
    if best_cycle is None:
        cycle = _find_hamilton_in_prefix(NUM_EDGES, deadline)
        if cycle:
            best_cycle = cycle
            best_score = _cycle_score(cycle)
    
    return best_cycle, best_score

def construct():
    """Return a permutation of vertices 0..23 describing a Hamilton cycle."""
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Quick fallback: identity cycle
    fallback = list(range(N))
    fallback_score = _cycle_score(fallback)
    
    # Try to find better cycle
    best_cycle = None
    best_score = fallback_score
    
    # Strategy 1: Binary search for minimal prefix
    cycle, score = _find_optimal_cycle(deadline)
    if cycle is not None and score < best_score:
        best_cycle = cycle
        best_score = score
    
    # Strategy 2: If we have time, try to improve with local search
    if best_cycle is not None and time.time() < deadline - 5.0:
        # Local search: try swapping adjacent vertices to reduce max arrival
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            cur = best_cycle
            cur_score = _cycle_score(cur)
            for i in range(N - 1):
                if time.time() > deadline - 2.0:
                    break
                # Try swapping cur[i] and cur[i+1]
                new_cycle = cur[:]
                new_cycle[i], new_cycle[i+1] = new_cycle[i+1], new_cycle[i]
                ns = _cycle_score(new_cycle)
                if ns < cur_score:
                    best_cycle = new_cycle
                    best_score = ns
                    cur = new_cycle
                    cur_score = ns
                    improved = True
    
    # Strategy 3: Try multiple random starts with DFS
    if time.time() < deadline - 5.0:
        adj = [set() for _ in range(N)]
        for a, b in EDGE_ORDER:
            adj[a].add(b)
            adj[b].add(a)
        
        # Try different starting vertices
        for start in range(N):
            if time.time() > deadline - 3.0:
                break
            for second in sorted(adj[start], key=lambda x: len(adj[x])):
                if time.time() > deadline - 3.0:
                    break
                path = [start, second]
                visited = {start, second}
                
                def dfs2(path, visited):
                    nonlocal best_cycle, best_score
                    if len(path) == N:
                        if start in adj[path[-1]]:
                            s = _cycle_score(path)
                            if s < best_score:
                                best_score = s
                                best_cycle = list(path)
                            return True
                        return False
                    cur = path[-1]
                    neighbors = sorted(adj[cur], key=lambda x: len(adj[x]))
                    for nb in neighbors:
                        if nb not in visited:
                            path.append(nb)
                            visited.add(nb)
                            if dfs2(path, visited):
                                return True
                            path.pop()
                            visited.discard(nb)
                    return False
                
                dfs2(path, visited)
                if best_score <= N:  # Can't do better than N
                    break
    
    if best_cycle is None:
        return fallback
    return best_cycle
# EVOLVE-BLOCK-END
