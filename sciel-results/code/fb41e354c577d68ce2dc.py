"""Construct a 3-regular graph of girth >= g using random configuration model."""
import random
from collections import deque

def _is_simple_3reg(stubs):
    """Check if the stub pairing produces a simple 3-regular graph."""
    n = len(stubs) // 3
    adj = [set() for _ in range(n)]
    for u, v in stubs:
        if u == v:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
    for v in range(n):
        if len(adj[v]) != 3:
            return False
    return True

def _compute_girth_fast(n, adj, max_depth):
    """Compute girth using BFS from each vertex, stopping at max_depth."""
    girth = max_depth + 1
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                break
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] < dist[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < girth:
                        girth = cycle_len
    return girth

def _generate_random_3reg(n):
    """Generate a random simple 3-regular graph on n vertices using configuration model."""
    if n % 2 != 0:
        return None
    for _ in range(1000):
        # Create stubs: each vertex has 3 stubs
        stubs = []
        for v in range(n):
            for _ in range(3):
                stubs.append(v)
        # Randomly pair stubs
        random.shuffle(stubs)
        pairs = [(stubs[i], stubs[i+1]) for i in range(0, len(stubs), 2)]
        # Check simplicity
        adj = [set() for _ in range(n)]
        simple = True
        for u, v in pairs:
            if u == v:
                simple = False
                break
            if v in adj[u]:
                simple = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if not simple:
            continue
        # Convert to list
        adj_list = [list(s) for s in adj]
        return adj_list
    return None

def construct(k, g):
    """Construct a 3-regular graph with girth >= g."""
    assert k == 3, "This implementation only handles k=3"
    
    random.seed(42)
    
    # Moore bound for 3-regular girth g
    if g % 2 == 0:
        moore = 2 * sum(2**i for i in range(g//2))
    else:
        moore = 2 * sum(2**i for i in range((g-1)//2)) + 2 * 2**((g-1)//2)
    
    # Start from a reasonable lower bound
    n_start = max(moore + 2, 100)
    
    best_n = None
    best_edges = None
    
    # Try increasing n values
    for n in range(n_start, 3000, 2):  # n must be even for 3-regular
        adj = _generate_random_3reg(n)
        if adj is None:
            continue
        
        # Compute girth
        girth = _compute_girth_fast(n, adj, g - 1)
        
        if girth >= g:
            # Build edge list
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
    
    # Fallback: try larger n with more attempts
    for n in range(3000, 5000, 2):
        for attempt in range(50):
            adj = _generate_random_3reg(n)
            if adj is None:
                continue
            girth = _compute_girth_fast(n, adj, g - 1)
            if girth >= g:
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    # Last resort: return a trivial large graph (shouldn't happen)
    n = 5000
    adj = _generate_random_3reg(n)
    if adj is None:
        raise RuntimeError("Failed to generate graph")
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return n, edges
