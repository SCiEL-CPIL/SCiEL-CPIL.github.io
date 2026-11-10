import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses a randomized BFS-based search to find a small cage graph.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds budget (5s safety margin)
    
    # Moore bound
    if g % 2 == 0:
        moore_bound = 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        moore_bound = 1 + (k - 1) * sum((k - 1) ** i for i in range((g - 1) // 2))
    
    best_n = float('inf')
    best_edges = []
    
    # First, get the LUW construction as a baseline
    luw_n, luw_edges = _luw_fallback(k, g)
    best_n = luw_n
    best_edges = luw_edges
    
    # Try to find a smaller graph using randomized BFS construction
    # For k=3, g=16, Moore bound is 512, known best is 936
    # We'll try various target sizes from Moore bound up to best_n - 1
    
    random.seed(42)  # Deterministic for reproducibility
    
    for target_n in range(max(moore_bound, 512), min(best_n, 800)):
        if time.time() > deadline - 10:
            break
        
        # Try a few attempts for each target size
        attempts = max(1, int((deadline - 10 - time.time()) / 5))
        
        for _ in range(attempts):
            if time.time() > deadline - 5:
                break
            
            n, edges = _try_bfs_cage(k, g, target_n, start_time, deadline)
            if n is not None and n < best_n:
                best_n = n
                best_edges = edges
                # Found a smaller graph, try to go even smaller
                # But we're iterating upwards, so this is already the best for this target
                break  # Move to next (larger) target
    
    return best_n, best_edges

def _try_bfs_cage(k, g, target_n, start_time, deadline):
    """
    Attempt to construct a k-regular graph of size target_n with girth >= g
    using a BFS-based layered construction.
    
    Returns (n, edges) or (None, None).
    """
    # Strategy: Build the graph layer by layer from a root vertex.
    # For girth g, we need to ensure no cycles shorter than g.
    # We use a non-backtracking walk approach.
    
    # For even girth g = 2r, the graph has a "center" edge and layers expanding from both ends.
    # For odd girth, similar with a center vertex.
    
    r = g // 2 if g % 2 == 0 else (g + 1) // 2
    
    # We'll try to build a graph with exactly target_n vertices.
    # The Moore bound for k=3, g=16 is 512 = 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 2 * 255 = 510
    # Wait, let me recalculate: for g=16, r=8
    # Moore bound = 2 * sum_{i=0}^{7} 2^i = 2 * (2^8 - 1) = 2 * 255 = 510
    # But the task says Moore bound is 512, so maybe I'm off by one.
    
    # Let's use a different approach: randomized construction with girth checking.
    # Generate random 3-regular graphs and check if they have girth >= g.
    # This is expensive but might work for small n.
    
    # For n up to ~600, we can try this.
    if target_n > 700:
        return None, None  # Too large for this approach
    
    # Try to generate a random 3-regular graph with target_n vertices
    # and check its girth.
    
    for attempt in range(3):
        if time.time() > deadline - 2:
            break
        
        edges = _generate_random_3reg(target_n)
        if edges is None:
            continue
        
        # Check girth
        girth = _compute_girth(target_n, edges)
        if girth >= g:
            return target_n, edges
    
    return None, None

def _generate_random_3reg(n):
    """Generate a random 3-regular graph on n vertices."""
    if n < 4 or n % 2 != 0:
        return None
    
    # Use configuration model
    half_edges = [i for i in range(n) for _ in range(3)]
    random.shuffle(half_edges)
    
    edges = set()
    for i in range(0, len(half_edges), 2):
        u, v = half_edges[i], half_edges[i+1]
        if u == v:
            return None  # Self-loop, retry
        if (min(u,v), max(u,v)) in edges:
            return None  # Multi-edge, retry
        edges.add((min(u,v), max(u,v)))
    
    # Check all vertices have degree 3
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    if any(d != 3 for d in degree):
        return None
    
    return list(edges)

def _compute_girth(n, edges):
    """Compute the girth of a graph using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_girth = float('inf')
    
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
    
    return min_girth if min_girth < float('inf') else float('inf')

def _luw_fallback(k, g):
    """
    Original Lazebnik-Ustimenko-Woldar construction.
    """
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
# EVOLVE-BLOCK-END
