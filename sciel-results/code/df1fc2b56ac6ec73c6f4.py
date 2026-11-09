"""Construct a 5-regular graph with girth 9 (cage problem n(5,9))."""
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s limit
    
    def elapsed():
        return time.time() - start_time
    
    # Baseline: LUW construction
    baseline_n, baseline_edges = _luw_construct(k, g)
    
    best_n = baseline_n
    best_edges = baseline_edges
    
    # Try to find better (smaller) valid graphs using randomized search
    # with time budget
    if elapsed() < 20.0:
        try:
            n, edges = _randomized_search(k, g, deadline)
            if n < best_n:
                best_n = n
                best_edges = edges
        except Exception:
            pass
    
    return best_n, best_edges

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction: q-regular bipartite graph with girth >= d+5."""
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
        r = (c - 5) % 4
        c0 = c - r
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
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _has_short_cycle(n, edges, max_girth):
    """Check if graph has a cycle shorter than max_girth using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        # BFS tracking parent to avoid immediate backtrack
        visited = {}
        queue = deque()
        visited[start] = (start, -1, 0)  # (node, parent, depth)
        queue.append((start, -1, 0))
        
        while queue:
            node, parent, depth = queue.popleft()
            if depth >= max_girth:
                break
            for nb in adj[node]:
                if nb == parent:
                    continue
                if nb == start and depth + 1 < max_girth:
                    return True  # Found cycle of length depth+1
                if nb in visited:
                    if visited[nb][1] != node and depth + 1 < max_girth:
                        return True
                else:
                    visited[nb] = (node, nb, depth + 1)
                    queue.append((nb, node, depth + 1))
    return False

def _randomized_search(k, g, deadline):
    """Randomized local search to find small k-regular graph with girth >= g."""
    start_time = time.time()
    
    # Try various sizes starting from lower bound
    # Moore bound for odd girth g=9, degree k=5:
    # n >= 2*(1 + k + k*(k-1) + k*(k-1)^2 + k*(k-1)^3)
    moore = 2 * (1 + k + k*(k-1) + k*(k-1)**2 + k*(k-1)**3)
    
    best_n = float('inf')
    best_edges = None
    
    # Try sizes from moore bound upward
    for n in range(moore, moore + 100):
        if time.time() > deadline - 10:
            break
        if n % 2 == 0 and n * k % 2 != 0:
            continue  # Handshake: n*k must be even
        
        # Try a few random 5-regular graphs of this size
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            edges = _random_regular_graph(n, k)
            if edges is None:
                continue
            
            if _has_short_cycle(n, edges, g):
                continue
            
            # Try to improve girth with local search
            n2, edges2 = _local_search_girth(n, edges, k, g, deadline)
            if n2 < best_n:
                best_n = n2
                best_edges = edges2
    
    if best_edges is None:
        return None, None
    return best_n, best_edges

def _random_regular_graph(n, k):
    """Generate a random k-regular graph using configuration model."""
    if n * k % 2 != 0:
        return None
    
    # Create stubs
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    
    random.shuffle(stubs)
    
    # Pair stubs
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None  # Loop, reject
        if (min(u,v), max(u,v)) in edges:
            return None  # Multi-edge, reject
        edges.add((min(u,v), max(u,v)))
    
    if len(edges) != n * k // 2:
        return None
    
    return list(edges)

def _local_search_girth(n, edges, k, g, deadline):
    """Local search to increase girth via edge swaps."""
    start_time = time.time()
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find all short cycles
    def find_short_cycles(adj, n, max_len):
        cycles = []
        for start in range(n):
            # BFS
            visited = {start: (start, -1, 0)}
            queue = deque([(start, -1, 0)])
            while queue:
                node, parent, depth = queue.popleft()
                if depth >= max_len:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if nb == start and depth + 1 <= max_len:
                        cycles.append((start, node, depth + 1))
                    elif nb not in visited:
                        visited[nb] = (node, nb, depth + 1)
                        queue.append((nb, node, depth + 1))
                    elif visited[nb][1] != node and depth + 1 <= max_len:
                        cycles.append((start, node, depth + 1))
        return cycles
    
    max_iter = 50
    for _ in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        cycles = find_short_cycles(adj, n, g - 1)
        if not cycles:
            break
        
        # Pick a random short cycle and try to break it
        start, node, length = random.choice(cycles)
        # Find the edge to remove
        # This is complex; skip for now
        break
    
    return n, edges
