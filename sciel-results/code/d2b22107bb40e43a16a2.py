import random
from collections import deque
import time

def construct(k, g):
    """Construct a k-regular graph with girth at least g."""
    start_time = time.time()
    
    if k != 3 or g != 16:
        # Fallback to LUW for other parameters
        return _luw_construct(k, g)
    
    # Try to find a (3,16)-cage
    # Known smallest: 936 vertices
    # Try promising n values first, then expand search
    
    # Time budget: 100 seconds for search, 15 seconds for fallback
    time_budget = 100
    
    # Try n values: start with known cage size, then expand
    n_values = [936]
    # Add other even values around 936
    for offset in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        n_values.append(936 + offset)
        n_values.append(936 - offset)
    # Also try some larger values
    for n in range(960, 1000, 2):
        if n not in n_values:
            n_values.append(n)
    # Sort by distance from 936
    n_values.sort(key=lambda x: abs(x - 936))
    
    for n in n_values:
        if time.time() - start_time > time_budget:
            break
        if n < 766 or n % 2 != 0:
            continue
        half = n // 2
        # Try to build random 3-regular bipartite graphs
        attempts = max(10, 200 - int(10 * (time.time() - start_time)))
        for attempt in range(attempts):
            if time.time() - start_time > time_budget:
                break
            edges = _random_3reg_bipartite(half)
            if edges is None:
                continue
            if _check_girth_fast(edges, n, g):
                return n, edges
    
    # Fallback: return a valid but possibly larger graph
    # Use LUW construction
    return _luw_construct(k, g)

def _random_3reg_bipartite(half):
    """Generate a random 3-regular bipartite graph with half vertices on each side."""
    n = half * 2
    
    # Configuration model with rejection
    for _ in range(100):  # Try up to 100 times to avoid multi-edges
        stubs_a = [i for i in range(half) for _ in range(3)]
        stubs_b = [i + half for i in range(half) for _ in range(3)]
        
        random.shuffle(stubs_a)
        random.shuffle(stubs_b)
        
        edges = set()
        valid = True
        for a, b in zip(stubs_a, stubs_b):
            if a > b:
                a, b = b, a
            if (a, b) in edges:
                valid = False
                break
            edges.add((a, b))
        
        if not valid:
            continue
        
        # Check all vertices have degree 3
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != 3 for d in deg):
            continue
        
        return list(edges)
    
    return None

def _check_girth_fast(edges, n, g):
    """Check if graph has girth at least g using non-backtracking walks."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For girth >= g, no non-backtracking walk of length < g should return to start
    # Check from each vertex, but use early termination
    max_depth = (g - 1) // 2  # For bipartite, max cycle length to check is g-2
    
    for start in range(n):
        # BFS with parent tracking
        # state: (node, parent, depth)
        visited = {}  # (node, parent) -> depth
        queue = deque([(start, -1, 0)])
        visited[(start, -1)] = 0
        
        while queue:
            node, parent, depth = queue.popleft()
            if depth >= max_depth:
                continue
            
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                key = (neighbor, node)
                if key in visited:
                    # Found a cycle
                    cycle_len = visited[key] + depth + 1
                    if cycle_len < g:
                        return False
                else:
                    visited[key] = depth + 1
                    queue.append((neighbor, node, depth + 1))
    
    return True

def _luw_construct(k, g):
    """Fallback to LUW construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
