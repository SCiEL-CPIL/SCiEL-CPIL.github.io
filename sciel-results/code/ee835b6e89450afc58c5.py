"""Construct small 3-regular graphs with large girth using BFS-based construction with cycle avoidance."""
import time
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # For the specific case k=3, g=16, use a known construction
    # Moore bound: n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i for even g
    if g % 2 == 0:
        half_g = g // 2
        moore_bound = 2 * sum((k-1)**i for i in range(half_g))
    else:
        half_g = (g - 1) // 2
        moore_bound = 1 + k * sum((k-1)**i for i in range(half_g))
    
    # Try BFS-based construction
    best_n = None
    best_edges = None
    
    # For k=3, g=16, we know the answer is around 936 vertices
    # Let's try to construct it using a systematic approach
    
    # Use a known parametric family or try multiple seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        result = _try_bfs_construct(k, g, seed, deadline)
        if result is not None:
            n, edges = result
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: LUW construction
    return _luw_construct(k, g)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
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


def _try_bfs_construct(k, g, seed, deadline):
    """Try to construct a k-regular graph with girth >= g using BFS with cycle avoidance."""
    import random
    random.seed(seed)
    
    # Build a tree first, then close cycles to make it regular
    # Start with a root vertex
    n_vertices = 0
    adjacency = {}  # vertex -> set of neighbors
    parent = {}     # vertex -> parent in BFS tree
    depth = {}      # vertex -> depth from root
    
    def can_connect(u, v):
        """Check if connecting u and v would create a cycle shorter than g."""
        if u == v:
            return False
        if v in adjacency.get(u, set()):
            return False
        
        # If both are in the tree, check if the path between them plus this edge
        # creates a cycle shorter than g
        if u in depth and v in depth:
            # Find LCA and compute cycle length
            # For simplicity, just check if they're close in the tree
            # This is a heuristic - we need to be more careful
            
            # Get the path from u to v through the tree
            path_u = _get_path_to_root(u, parent)
            path_v = _get_path_to_root(v, parent)
            
            # Find LCA
            lca = None
            for node in path_u:
                if node in path_v:
                    lca = node
                    break
            
            if lca is None:
                return True  # Not in same tree component
            
            # Cycle length = depth[u] + depth[v] - 2*depth[lca] + 1
            cycle_len = depth[u] + depth[v] - 2 * depth[lca] + 1
            return cycle_len >= g
        
        return True
    
    def _get_path_to_root(v, parent):
        path = []
        cur = v
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        return path
    
    # Start with vertex 0
    n_vertices = 1
    adjacency[0] = set()
    parent[0] = None
    depth[0] = 0
    
    # BFS frontier
    frontier = deque([0])
    
    # We need to build enough vertices
    # Target: at least Moore bound
    moore_bound = 2 * sum((k-1)**i for i in range(g // 2)) if g % 2 == 0 else 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    max_attempts = 5000
    attempt = 0
    
    while frontier and attempt < max_attempts:
        if time.time() > deadline:
            return None
        
        u = frontier.popleft()
        attempt += 1
        
        # Check if u needs more connections
        if len(adjacency[u]) >= k:
            continue
        
        # Try to connect u to existing vertices first
        candidates = [v for v in adjacency if v != u and v not in adjacency[u] and can_connect(u, v)]
        
        if candidates:
            # Pick a random candidate
            v = random.choice(candidates)
            adjacency[u].add(v)
            adjacency[v].add(u)
            # If v was not fully connected, add to frontier
            if len(adjacency[v]) < k:
                frontier.append(v)
        else:
            # Create a new vertex
            new_v = n_vertices
            n_vertices += 1
            adjacency[new_v] = set()
            parent[new_v] = u
            depth[new_v] = depth[u] + 1
            adjacency[u].add(new_v)
            adjacency[new_v].add(u)
            frontier.append(new_v)
    
    # Check if we have a valid graph
    if len(adjacency) < moore_bound:
        return None
    
    # Verify all vertices have degree k
    for v in adjacency:
        if len(adjacency[v]) != k:
            return None
    
    # Verify girth
    if not _check_girth(adjacency, g):
        return None
    
    # Convert to edge list
    edges = []
    seen = set()
    for u in adjacency:
        for v in adjacency[u]:
            if (u, v) not in seen and (v, u) not in seen:
                edges.append((u, v))
                seen.add((u, v))
                seen.add((v, u))
    
    return n_vertices, edges


def _check_girth(adjacency, g):
    """Check if the graph has girth at least g using BFS from each vertex."""
    n = len(adjacency)
    
    for start in range(n):
        # BFS to find shortest cycle containing start
        visited = {start: (None, 0)}  # vertex -> (parent, distance)
        queue = deque([(start, None, 0)])
        
        while queue:
            node, par, dist = queue.popleft()
            
            for neighbor in adjacency[node]:
                if neighbor == par:
                    continue
                if neighbor in visited:
                    # Found a cycle
                    cycle_len = dist + visited[neighbor][1] + 1
                    if cycle_len < g:
                        return False
                else:
                    visited[neighbor] = (node, dist + 1)
                    queue.append((neighbor, node, dist + 1))
    
    return True
