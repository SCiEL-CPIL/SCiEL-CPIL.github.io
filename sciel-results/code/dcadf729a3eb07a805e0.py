"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph of girth >= g with as few vertices as possible.
    
    Uses a combination of:
    1. Known explicit constructions (LUW, etc.)
    2. Randomized search with constraint propagation
    3. Deadline guard to ensure we return within time limit
    
    For the specific case k=3, g=16, we use the LUW construction which gives
    a valid graph, then attempt to find smaller graphs via random search.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Moore bound: lower bound on n
    # For k-regular graph with girth g:
    # If g is even: n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
    # If g is odd: n >= 1 + k * sum_{i=0}^{(g-1)/2-1} (k-1)^i
    if g % 2 == 0:
        moore_bound = 2 * sum((k-1)**i for i in range(g//2))
    else:
        moore_bound = 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    # Start with LUW construction as baseline
    best_n, best_edges = _luw_construction(k, g)
    
    # Try to improve with random search if we have time
    if best_n > moore_bound:
        while time.time() < deadline - 5:
            n, edges = _random_search(k, g, time_limit=deadline - time.time() - 2)
            if n is not None and n < best_n:
                best_n, best_edges = n, edges
    
    return best_n, best_edges


def _luw_construction(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
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


def _random_search(k, g, time_limit):
    """
    Randomized search for a k-regular graph of girth >= g.
    Uses a non-backtracking walk based approach to verify girth.
    """
    import time
    
    start_time = time.time()
    
    # For small cases, try random regular graphs
    # Start with reasonable size
    if g % 2 == 0:
        moore_bound = 2 * sum((k-1)**i for i in range(g//2))
    else:
        moore_bound = 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    # Try a few different sizes
    for target_n in [moore_bound, moore_bound + 10, moore_bound + 50, moore_bound + 100]:
        if time.time() - start_time > time_limit - 1:
            break
        if target_n % 2 != 0 and k * target_n % 2 != 0:
            target_n += 1
        
        result = _try_random_regular(k, target_n, g, time_limit - (time.time() - start_time))
        if result is not None:
            return result
    
    return None


def _try_random_regular(k, n, g, time_limit):
    """Try to construct a random k-regular graph with girth >= g."""
    import time
    import random
    
    start_time = time.time()
    max_attempts = 100
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
        
        # Generate random k-regular graph using configuration model
        adj = _random_regular_graph(k, n)
        if adj is None:
            continue
        
        # Check girth using BFS from each vertex (limited depth)
        if _check_girth(adj, g):
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
    
    return None


def _random_regular_graph(k, n):
    """Generate a random k-regular graph using configuration model."""
    import random
    
    if k * n % 2 != 0:
        return None
    
    # Create stubs
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    
    # Random matching
    random.shuffle(stubs)
    adj = {v: [] for v in range(n)}
    
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None  # Self-loop, reject
        if v in adj[u]:
            return None  # Multi-edge, reject
        adj[u].append(v)
        adj[v].append(u)
    
    # Verify regularity
    for v in range(n):
        if len(adj[v]) != k:
            return None
    
    return adj


def _check_girth(adj, g):
    """Check if graph has girth >= g using BFS from each vertex."""
    n = len(adj)
    
    for start in range(n):
        # BFS from start
        dist = {start: 0}
        parent = {start: None}
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= g // 2:
                continue
            
            for v in adj[u]:
                if v == parent.get(u):
                    continue
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif dist[v] == dist[u] + 1:
                    # Found a cycle of length dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True


# EVOLVE-BLOCK-END
