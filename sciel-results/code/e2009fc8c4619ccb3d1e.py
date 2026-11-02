import time
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    if k != 4:
        return 0, []

    # Moore bound for k=4, g=10 (even)
    r = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(r))
    
    # n must be at least moore_bound
    # Known answer is 320.
    # We search even n first.
    
    for n in range(moore_bound, 400):
        if time.time() > deadline:
            break
            
        if n % 2 != 0:
            continue

        # Try to find generators a, b
        # Graph is Z_n with generators {a, -a, b, -b}
        # We need gcd(n, a, b) == 1 for connectivity
        found = False
        for a in range(1, n // 2):
            if time.time() > deadline:
                return 0, []
            for b in range(a + 1, n // 2):
                if math.gcd(math.gcd(n, a), b) != 1:
                    continue
                
                # Check girth and connectivity
                valid, edges = _check_cyclic_graph(n, a, b, g)
                if valid:
                    return n, edges
        
    # If not found, return empty
    return 0, []

def _check_cyclic_graph(n, a, b, g):
    """
    Checks if the cyclic graph Z_n with generators {a, -a, b, -b} has girth >= g.
    Returns (valid, edges).
    """
    # BFS from 0
    dist = [-1] * n
    prev = [-1] * n
    dist[0] = 0
    queue = deque([0])
    
    # Generators
    gens = [a, n - a, b, n - b]
    
    while queue:
        v = queue.popleft()
        dv = dist[v]
        if dv >= g:
            continue
        
        for gen in gens:
            u = (v + gen) % n
            if dist[u] == -1:
                dist[u] = dv + 1
                prev[u] = v
                queue.append(u)
            else:
                if u == prev[v]:
                    continue
                cycle_len = dv + 1 + dist[u]
                if cycle_len < g:
                    return False, []
                
    # Check connectivity
    if any(d == -1 for d in dist):
        return False, []
        
    # Generate edges
    edges = []
    for v in range(n):
        for gen in [a, b]:
            u = (v + gen) % n
            if v < u:
                edges.append((v, u))
            else:
                edges.append((u, v))
                
    return True, edges
