import random
import sys
from collections import deque

def construct(k, g):
    if k != 3 or g != 16:
        return _construct_general(k, g)
    
    # For k=3, g=16: try random 3-regular graphs
    # Start near known best (936) and try a few sizes
    target_sizes = [936, 940, 944, 948, 952, 956, 960, 964, 968, 972, 976, 980, 984, 988, 992, 996, 1000]
    
    for n in target_sizes:
        if n % 2 != 0:
            continue  # 3-regular graphs need even n
        for attempt in range(50):
            edges = _random_3reg(n)
            if edges is None:
                continue
            if _has_girth_at_least(edges, n, g):
                return n, edges
    
    # Fallback: try larger sizes
    for n in range(1004, 1200, 4):
        for attempt in range(20):
            edges = _random_3reg(n)
            if edges is None:
                continue
            if _has_girth_at_least(edges, n, g):
                return n, edges
    
    # Last resort: use a simple construction
    return _construct_general(k, g)

def _random_3reg(n):
    """Generate a random 3-regular graph on n vertices. Returns list of edges or None."""
    if n < 4 or n % 2 != 0:
        return None
    
    # Use configuration model: create 3 stubs per vertex, pair them randomly
    stubs = []
    for v in range(n):
        stubs.extend([v] * 3)
    
    random.shuffle(stubs)
    
    edges = []
    seen = set()
    for i in range(0, 3 * n, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None  # Self-loop, reject
        a, b = min(u, v), max(u, v)
        if (a, b) in seen:
            return None  # Multi-edge, reject
        seen.add((a, b))
        edges.append((u, v))
    
    # Verify it's actually 3-regular and simple
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != 3 for d in deg):
        return None
    
    return edges

def _has_girth_at_least(edges, n, g):
    """Check if graph has girth at least g (no cycles shorter than g)."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # BFS from each vertex to detect cycles
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _construct_general(k, g):
    """Fallback for other parameters - use a simple random approach."""
    # For general case, try random regular graphs
    for n in range(max(2 * (k - 1) + 2, 20), 500):
        if n * k % 2 != 0:
            continue
        for attempt in range(10):
            stubs = []
            for v in range(n):
                stubs.extend([v] * k)
            random.shuffle(stubs)
            edges = []
            seen = set()
            valid = True
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                a, b = min(u, v), max(u, v)
                if (a, b) in seen:
                    valid = False
                    break
                seen.add((a, b))
                edges.append((u, v))
            if not valid:
                continue
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if any(d != k for d in deg):
                continue
            if _has_girth_at_least(edges, n, g):
                return n, edges
    # Return a trivial invalid graph as last resort
    return 2 * (k - 1) + 2, []
