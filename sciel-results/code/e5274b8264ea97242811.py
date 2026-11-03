"""Program for constructing small k-regular graphs of given girth."""
import numpy as np
import random
import time
from collections import deque

# EVOLVE-BLOCK-START
def _luw_construct(k, g):
    """Fallback LUW construction."""
    if k == 2:
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    
    if k == 1:
        return 2, [(0, 1)]
    
    # LUW parameters
    d_odd = g - 5
    if d_odd < 3: d_odd = 3
    if d_odd % 2 == 0: d_odd += 1
    
    d_even = g - 4
    if d_even < 2: d_even = 2
    if d_even % 2 == 1: d_even += 1
    
    d = min(d_odd, d_even)
    q = k
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0: rule[c] = (1, c0 - 2)
        elif r == 1: rule[c] = (c0 - 1, 1)
        elif r == 2: rule[c] = (c0, 1)
        else: rule[c] = (1, c0 + 1)
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges

def _generate_random_bipartite_k_reg(nL, nR, k):
    """Generate a random simple k-regular bipartite graph."""
    if nL * k != nR * k:
        return None
    if nL < k or nR < k:
        return None
        
    for _ in range(1000): # Retry loop for simplicity
        # Configuration model
        left_stubs = [i for i in range(nL) for _ in range(k)]
        right_stubs = [j for j in range(nR) for _ in range(k)]
        random.shuffle(right_stubs)
        
        edges = set()
        valid = True
        for l, r in zip(left_stubs, right_stubs):
            # r is 0..nR-1, map to nL..nL+nR-1
            u = l
            v = nL + r
            if u == v: # Impossible in bipartite
                valid = False
                break
            if (u, v) in edges:
                valid = False
                break
            edges.add((u, v))
        
        if not valid:
            continue
            
        # Check regularity (guaranteed by construction) and simplicity (checked)
        # Convert to list
        return list(edges)
    return None

def _generate_random_k_reg(n, k):
    """Generate a random simple k-regular graph."""
    if (n * k) % 2 != 0:
        return None
    if n < k:
        return None
        
    for _ in range(1000):
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if (u, v) in edges or (v, u) in edges:
                valid = False
                break
            edges.add((u, v))
        
        if not valid:
            continue
        return list(edges)
    return None

def _check_girth(n, edges, g):
    """Check if graph has girth at least g."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            # If we have already explored up to g/2, we can stop early
            if dist[u] >= g // 2:
                continue
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
                        
    return True

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    """
    if k == 2:
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    
    if k == 1:
        return 2, [(0, 1)]
    
    # Moore Bound Estimate
    if g % 2 == 0:
        # Bipartite
        half_g = g // 2
        moore = 2 * sum((k-1)**i for i in range(half_g))
        n_start = moore
        if n_start % 2 != 0: n_start += 1
        bipartite = True
    else:
        half_g = (g-1) // 2
        moore = 1 + k * sum((k-1)**i for i in range(half_g))
        n_start = moore
        if (n_start * k) % 2 != 0: n_start += 1
        bipartite = False
        
    # Ensure n is at least k+1
    if n_start < k + 1:
        n_start = k + 1
        if (n_start * k) % 2 != 0: n_start += 1
        
    start_time = time.time()
    max_time = 110 # Leave 10s for overhead/fallback
    
    # Try random construction
    n = n_start
    attempts = 0
    while time.time() - start_time < max_time:
        if bipartite:
            nL = n // 2
            nR = n // 2
            edges = _generate_random_bipartite_k_reg(nL, nR, k)
        else:
            edges = _generate_random_k_reg(n, k)
            
        if edges is not None:
            if _check_girth(n, edges, g):
                return n, edges
        
        n += 2
        attempts += 1
        if attempts > 50: # Don't try too many sizes
            break
            
    # Fallback to LUW
    return _luw_construct(k, g)
# EVOLVE-BLOCK-END
