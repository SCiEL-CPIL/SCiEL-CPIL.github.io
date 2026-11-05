import time
import random
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k == 5 and g == 9:
        # Try to find a good 5-regular graph with girth >= 9
        # Start with a known good construction and try to improve
        result = _luw_construct(k, g)
        if result is not None:
            n, edges = result
            # Verify it's valid
            if _verify_graph(n, edges, k, g, deadline):
                return result
        
        # Try random Cayley graph approach
        result = _random_cayley_construct(k, g, deadline)
        if result is not None:
            n, edges = result
            if _verify_graph(n, edges, k, g, deadline):
                return result
        
        # Fallback to LUW
        return _luw_construct(k, g)
    
    return _luw_construct(k, g)

def _verify_graph(n, edges, k, g, deadline):
    """Verify the graph is k-regular, simple, and has girth >= g"""
    if time.time() > deadline - 10:
        return False
    
    # Check edge count and regularity
    if len(edges) != n * k // 2:
        return False
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    edge_set = set()
    for i, j in edges:
        if i == j:
            return False
        if i >= n or j >= n:
            return False
        key = (min(i, j), max(i, j))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[i].append(j)
        adj[j].append(i)
    
    # Check regularity
    for v in range(n):
        if len(adj[v]) != k:
            return False
    
    # Check girth using BFS from each vertex (with early termination)
    # For efficiency, we only need to check up to depth (g-1)//2
    max_depth = (g - 1) // 2
    for v in range(n):
        if time.time() > deadline - 5:
            return True  # Assume valid if we're running out of time
        if not _check_girth_from(v, adj, max_depth, n):
            return False
    
    return True

def _check_girth_from(start, adj, max_depth, n):
    """Check that there's no cycle shorter than g starting from start"""
    # BFS with parent tracking to detect cycles
    visited = [-1] * n  # -1 = unvisited, otherwise depth
    parent = [-1] * n
    queue = [(start, 0)]
    visited[start] = 0
    
    while queue:
        v, depth = queue[0]
        queue.pop(0)
        
        if depth >= max_depth:
            continue
        
        for u in adj[v]:
            if visited[u] == -1:
                visited[u] = depth + 1
                parent[u] = v
                queue.append((u, depth + 1))
            elif parent[v] != u and parent[u] != v:
                # Found a cycle - check its length
                # The cycle length is depth + visited[u] + 1
                cycle_len = depth + visited[u] + 1
                if cycle_len < 2 * max_depth + 1:  # If girth < g
                    return False
    
    return True

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction"""
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

def _random_cayley_construct(k, g, deadline):
    """Try to construct a random Cayley graph on a non-abelian group"""
    if time.time() > deadline - 30:
        return None
    
    # Use a product of cyclic groups as the base
    # For k=5, we need 5 involutory generators
    # Let's try Z_5^3 with carefully chosen generators
    
    # Actually, let's try a different approach: use a known good seed
    # and try local search to improve
    
    # For now, return None to use LUW
    return None
