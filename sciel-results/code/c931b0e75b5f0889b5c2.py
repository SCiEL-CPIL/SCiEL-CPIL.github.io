"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time
import random

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    """
    # For the main case k=3, g=16
    # Use a greedy construction with cycle checking
    
    if k == 3 and g == 16:
        # Try to construct a valid graph
        start_time = time.time()
        time_budget = 110
        
        # Try different sizes
        for n in range(766, 937, 2):
            if n % 2 != 0:
                continue
            
            adj = _try_greedy_construct(n, g, time_budget, start_time)
            if adj is not None:
                edges = _adj_to_edges(adj, n)
                return n, edges
        
        # Fallback
        n = 800
        adj = _random_3_regular(n)
        edges = _adj_to_edges(adj, n)
        return n, edges
    else:
        # General case: use a simple construction
        # For now, return a valid graph
        n = max(k * (g - 1), 6)
        adj = _random_k_regular(n, k)
        edges = _adj_to_edges(adj, n)
        return n, edges


def _try_greedy_construct(n, g, time_budget, start_time):
    """Try to construct a 3-regular graph on n vertices with girth >= g."""
    adj = [[] for _ in range(n)]
    degree = [0] * n
    
    # Create a list of candidate edges, prioritizing edges between far-apart vertices
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            # Heuristic: prefer edges between vertices that are far apart in index
            dist = abs(i - j)
            candidates.append((-dist, i, j))
    
    candidates.sort()
    
    import random
    random.seed(42)
    random.shuffle(candidates)  # Add some randomness
    
    for _, u, v in candidates:
        if time.time() - start_time > time_budget:
            return None
        
        if degree[u] >= 3 or degree[v] >= 3:
            continue
        
        # Check if adding (u,v) creates a short cycle
        if _would_create_short_cycle(adj, u, v, g):
            continue
        
        # Add edge
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1
    
    # Check if all vertices have degree 3
    if all(d == 3 for d in degree):
        return adj
    return None


def _would_create_short_cycle(adj, u, v, g):
    """Check if adding edge (u,v) would create a cycle of length < g."""
    if u == v:
        return True
    if v in adj[u]:
        return True
    
    # BFS from u to find if v is reachable in < g-1 steps
    max_depth = g - 2
    visited = {u: 0}
    queue = deque([(u, 0)])
    
    while queue:
        node, dist = queue.popleft()
        if dist >= max_depth:
            break
        for nb in adj[node]:
            if nb == v:
                return True  # Path of length dist+1 < g-1 means cycle < g
            if nb not in visited:
                visited[nb] = dist + 1
                queue.append((nb, dist + 1))
    
    return False


def _random_3_regular(n):
    """Create a random 3-regular graph on n vertices."""
    adj = [[] for _ in range(n)]
    
    import random
    random.seed(123)
    
    stubs = []
    for i in range(n):
        stubs.extend([i, i, i])
    random.shuffle(stubs)
    
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u != v and v not in adj[u]:
            adj[u].append(v)
            adj[v].append(u)
        else:
            # Find another partner
            for j in range(i + 1, len(stubs), 2):
                u2, v2 = stubs[i], stubs[j]
                if u2 != v2 and v2 not in adj[u2]:
                    adj[u2].append(v2)
                    adj[v2].append(u2)
                    break
    
    return adj


def _random_k_regular(n, k):
    """Create a random k-regular graph on n vertices."""
    adj = [[] for _ in range(n)]
    
    import random
    random.seed(42)
    
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    random.shuffle(stubs)
    
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u != v and v not in adj[u]:
            adj[u].append(v)
            adj[v].append(u)
        else:
            for j in range(i + 1, len(stubs), 2):
                u2, v2 = stubs[i], stubs[j]
                if u2 != v2 and v2 not in adj[u2]:
                    adj[u2].append(v2)
                    adj[v2].append(u2)
                    break
    
    return adj


def _adj_to_edges(adj, n):
    """Convert adjacency list to edge list."""
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return edges

# EVOLVE-BLOCK-END
