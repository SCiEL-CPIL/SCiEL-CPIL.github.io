import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    # Helper: Check Girth
    def check_girth(adjacency, n, g):
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= g - 1:
                    continue
                for v in adjacency[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    # Helper: Verify Graph
    def verify_graph(n, edges, k, g):
        if n < 1:
            return False
        adjacency = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if u >= n or v >= n:
                return False
            if (min(u, v), max(u, v)) in edge_set:
                return False
            edge_set.add((min(u, v), max(u, v)))
            adjacency[u].append(v)
            adjacency[v].append(u)
        
        for i in range(n):
            if len(adjacency[i]) != k:
                return False
        
        if not check_girth(adjacency, n, g):
            return False
            
        return True

    # Petersen Graph (3,5)
    def petersen():
        n = 10
        edges = []
        # Outer pentagon
        for i in range(5):
            edges.append((i, (i + 1) % 5))
        # Inner star
        for i in range(5):
            edges.append((5 + i, 5 + (i + 2) % 5))
        # Spokes
        for i in range(5):
            edges.append((i, 5 + i))
        return n, edges

    # Tutte-Coxeter Graph (3,7)
    # This is the (3,7)-cage. 30 vertices.
    # Bipartite. 15 points, 15 lines.
    # I will use a known edge list or generate via LUW if hardcoding is risky.
    # However, LUW(3,3) gives 54 vertices.
    # 30 is significantly better.
    # I will hardcode the edges of the Tutte-Coxeter graph.
    # Source: Standard cage data.
    def tutte_coxeter():
        n = 30
        edges = []
        # The Tutte-Coxeter graph is the incidence graph of the projective plane of order 2? No, that's 14.
        # It is the (3,7) cage.
        # I will construct it using the LUW(3,3) if I can't recall the exact edges, 
        # but 54 is a valid answer. 
        # To be safe and avoid bugs, I will return LUW(3,3) for (3,7) 
        # UNLESS I can verify a 30-vertex graph quickly.
        # Actually, I can just use the LUW construction for all non-(3,5) cases.
        # The score improvement for (5,9) is the main goal.
        pass
        # Fallback to LUW in the main flow.

    # LUW Construction
    def luw_construct(k, d):
        q = k
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

    # Strategy
    if k == 3 and g <= 5:
        n, edges = petersen()
        if verify_graph(n, edges, k, g):
            return n, edges

    # For (5,9) and similar cases, try randomized search for small regular graphs
    # with sufficient girth, falling back to LUW if needed.
    
    def random_regular_graph(n, k, seed=42):
        """Generate a random k-regular graph on n vertices using configuration model."""
        if n * k % 2 != 0:
            return None
        if k >= n:
            return None
        
        rng = random.Random(seed)
        # Create stubs
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        rng.shuffle(stubs)
        
        # Pair stubs
        edges = []
        edge_set = set()
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                return None  # Self-loop, reject
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return None  # Multi-edge, reject
            edge_set.add(key)
            edges.append((u, v))
        
        # Verify degree
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != k for d in deg):
            return None
        
        return n, edges

    def check_girth_fast(adjacency, n, g):
        """Check girth using BFS from each vertex, early exit."""
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= g - 1:
                    continue
                for v in adjacency[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    # Try randomized search for small graphs
    if k == 5 and g == 9:
        # Moore bound for (5,9): 2 + 5*(1 + 4 + 4^2 + 4^3) = 2 + 5*(1+4+16+64) = 2+5*85 = 427
        # So n must be >= 428. Try random search for n in a reasonable range.
        best_n = None
        best_edges = None
        
        # Try increasing n from Moore bound upward
        moore_bound = 2 + k * sum(k**(i-1) for i in range(1, (g-1)//2 + 1))
        if k == 5 and g == 9:
            moore_bound = 428
        
        # For efficiency, try a few random seeds for small n values
        for n_try in range(moore_bound, moore_bound + 50):
            if time.time() > deadline:
                break
            for seed in range(20):
                if time.time() > deadline:
                    break
                result = random_regular_graph(n_try, k, seed=seed + n_try * 1000)
                if result is None:
                    continue
                n_val, edges_val = result
                # Quick girth check
                adjacency = [[] for _ in range(n_val)]
                for u, v in edges_val:
                    adjacency[u].append(v)
                    adjacency[v].append(u)
                if check_girth_fast(adjacency, n_val, g):
                    best_n = n_val
                    best_edges = edges_val
                    break
            if best_n is not None:
                break
        
        if best_n is not None:
            return best_n, best_edges

    # Calculate minimal depth for LUW
    # Girth of LUW(k, d) is 2d + 2
    # We need 2d + 2 >= g
    # d >= (g - 2) / 2
    d = (g - 1) // 2
    
    # Ensure d is at least 2 for LUW to be non-trivial
    if d < 2:
        d = 2

    luw_n, luw_edges = luw_construct(k, d)
    
    if verify_graph(luw_n, luw_edges, k, g):
        return luw_n, luw_edges
        
    # Fallback: try larger d
    for d_try in range(d + 1, d + 5):
        if time.time() > deadline:
            break
        luw_n, luw_edges = luw_construct(k, d_try)
        if verify_graph(luw_n, luw_edges, k, g):
            return luw_n, luw_edges

    return 0, []
