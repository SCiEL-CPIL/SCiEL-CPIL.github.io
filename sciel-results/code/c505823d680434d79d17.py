import random
import time
from collections import deque, defaultdict

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i,j) pairs.
    """
    # Strategy 1: LUW construction (Lazebnik-Ustimenko-Woldar)
    best_n = None
    best_edges = None
    
    # Try LUW construction
    try:
        n_luw, edges_luw = _luw_construct(k, g)
        if n_luw is not None:
            best_n = n_luw
            best_edges = edges_luw
    except Exception:
        pass
    
    # Strategy 2: Try randomized greedy construction for smaller graphs
    try:
        n_rand, edges_rand = _random_greedy(k, g, time_limit=60)
        if n_rand is not None and (best_n is None or n_rand < best_n):
            best_n = n_rand
            best_edges = edges_rand
    except Exception:
        pass
    
    # Strategy 3: Try algebraic constructions
    try:
        n_alg, edges_alg = _algebraic_construct(k, g)
        if n_alg is not None and (best_n is None or n_alg < best_n):
            best_n = n_alg
            best_edges = edges_alg
    except Exception:
        pass
    
    if best_n is None:
        # Fallback: just return LUW
        best_n, best_edges = _luw_construct(k, g)
    
    return best_n, best_edges


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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


def _random_greedy(k, g, time_limit=60):
    """
    Try to build a k-regular graph of girth at least g using greedy randomization.
    Start with a small number of vertices and try to add edges while maintaining girth.
    """
    start_time = time.time()
    
    # Moore bound: n >= 2 * sum_{i=0}^{(g-1)//2} (k-1)^i
    half_g = (g - 1) // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(half_g + 1))
    
    # Try a few different starting sizes
    for start_n in range(moore_bound, moore_bound + 100):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random seeds
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
            
            edges, success = _try_greedy_build(k, g, start_n, seed=attempt * 1000)
            if success and edges is not None:
                # Verify
                if _verify_girth(k, g, start_n, edges):
                    return start_n, edges
    
    return None, None


def _try_greedy_build(k, g, n, seed=0):
    """
    Try to build a k-regular graph on n vertices with girth at least g.
    Returns (edges, success).
    """
    rng = random.Random(seed)
    
    # Initialize with empty edge list
    degree = [0] * n
    edges = []
    edge_set = set()
    adjacency = [set() for _ in range(n)]
    
    # Build graph by adding edges one at a time
    # First, ensure we can complete to k-regular
    # We need to pair up "stubs"
    
    # Random configuration model: create k copies of each vertex, random permutation
    stubs = [v for v in range(n) for _ in range(k)]
    rng.shuffle(stubs)
    
    # Pair up stubs
    candidate_edges = []
    valid = True
    for i in range(0, n * k, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            valid = False
            break
        if (min(u, v), max(u, v)) in edge_set:
            valid = False
            break
        edge_set.add((min(u, v), max(u, v)))
        candidate_edges.append((u, v))
    
    if not valid:
        return None, False
    
    # Check girth
    if _verify_girth(k, g, n, candidate_edges):
        return candidate_edges, True
    
    return None, False


def _algebraic_construct(k, g):
    """
    Try algebraic constructions for specific small parameters.
    For k=5, g=9, try known explicit constructions.
    """
    if k != 5 or g != 9:
        return None, None
    
    # For (5,9): Try to use a different explicit construction
    # One approach: use the incidence graph of a generalized (something)
    # Another: use Cayley graph over a specific group
    
    # Try a simple approach: use the line graph of a known graph
    # Or try to construct via finite field extensions
    
    # For now, return None to indicate no better construction found
    # The LUW construction should give a valid (if not optimal) answer
    return None, None


def _verify_girth(k, g, n, edges):
    """
    Verify that the graph is k-regular and has girth at least g.
    Uses BFS from each vertex to check for short cycles.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check regularity
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check for short cycles using BFS from each vertex
    # For girth g, we need to check that there's no cycle of length < g
    # Equivalent: no two vertices at distance <= (g-1)//2 from each other have a common neighbor
    # More precisely: for each vertex, do BFS up to depth (g-1)//2 and check no collisions
    
    half = (g - 1) // 2
    
    for start in range(n):
        # BFS from start, tracking parent to detect cycles
        # Actually, for girth checking: 
        # A cycle of length L exists iff there are two different paths of length L/2 from some vertex
        # More efficiently: for each vertex v, do BFS and check that no vertex is reached twice within depth half
        
        visited = [-1] * n  # visited[v] = distance from start, -1 if not visited
        parent = [-1] * n
        visited[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if visited[u] >= half:
                continue
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and visited[v] <= visited[u]:
                    # Found a cycle
                    cycle_len = visited[u] + visited[v] + 1
                    if cycle_len < g:
                        return False
    
    return True
