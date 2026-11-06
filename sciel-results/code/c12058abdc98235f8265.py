"""Find Hamilton cycle minimizing max edge arrival index."""
import random
import time

def _build_edge_order():
    """Reconstruct the pinned edge order using random.Random(614)."""
    n = 24
    rng = random.Random(614)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges)
    return edges

def _check_hamilton_cycle(n, adjacency, deadline):
    """Check if graph (given by adjacency lists) has a Hamilton cycle.
    Uses DFS with Warnsdorff heuristic. Returns (True, cycle) or (False, None)."""
    # Pre-filter: every vertex must have degree >= 2
    for v in range(n):
        if len(adjacency[v]) < 2:
            return False, None
    
    # Start from vertex 0 (fix to break symmetry)
    start = 0
    path = [start]
    visited = [False] * n
    visited[start] = True
    
    # Use sets for faster membership in adjacency
    adj_sets = [set(adj) for adj in adjacency]
    
    node_count = 0
    MAX_NODES = 1000000  # Limit to prevent hanging
    
    def dfs(path, visited):
        nonlocal node_count
        if time.time() > deadline:
            return None
        node_count += 1
        if node_count > MAX_NODES:
            return None
        if len(path) == n:
            # Check closing edge
            last = path[-1]
            first = path[0]
            if first in adj_sets[last]:
                return list(path)
            return None
        
        cur = path[-1]
        # Get unvisited neighbors, sorted by Warnsdorff (fewest unvisited neighbors first)
        candidates = []
        for nb in adj_sets[cur]:
            if not visited[nb]:
                unvisited_count = sum(1 for x in adj_sets[nb] if not visited[x])
                candidates.append((unvisited_count, nb))
        
        candidates.sort()
        
        for _, nb in candidates:
            visited[nb] = True
            path.append(nb)
            result = dfs(path, visited)
            if result is not None:
                return result
            path.pop()
            visited[nb] = False
        
        return None
    
    result = dfs(path, visited)
    if result is not None:
        return True, result
    return False, None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    n = 24
    edges = _build_edge_order()
    
    # Binary search on k
    lo, hi = n, n * (n - 1) // 2  # 24 to 276
    best_cycle = None
    best_k = hi
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        # Build adjacency for edges with index <= mid
        adjacency = [[] for _ in range(n)]
        for i in range(1, mid + 1):
            u, v = edges[i - 1]
            adjacency[u].append(v)
            adjacency[v].append(u)
        
        found, cycle = _check_hamilton_cycle(n, adjacency, deadline)
        
        if found:
            best_k = mid
            best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is None:
        # Fallback: try to find any Hamilton cycle in complete graph
        return list(range(n))
    
    return best_cycle
