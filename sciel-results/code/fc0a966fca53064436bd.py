import time
import math
from collections import deque

def construct(k, g):
    deadline = time.time() + 115
    
    if k == 3 and g == 16:
        # Try to find a small graph using Klein construction
        # q must be prime power, q % 4 == 1
        # Candidates: 257, 521, 1021, 2053, ...
        candidates = [257, 521, 1021, 2053, 4109, 8213]
        
        for q in candidates:
            if time.time() > deadline:
                break
            if not _is_prime(q) or q % 4 != 1:
                continue
            
            n, edges = _klein_graph(q)
            if n is None:
                continue
            
            # Check girth
            if _check_girth(n, edges, g):
                return n, edges
        
        # Fallback to LUW if no small graph found
        return _luw_fallback(k, g)
    
    # Generic fallback for other k, g
    return _luw_fallback(k, g)

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def _klein_graph(q):
    # Vertices 0..q-1 are U, q..2q-1 are V
    n = 2 * q
    edges = []
    
    # Find a non-quadratic residue 'a'
    # a is a non-residue if pow(a, (q-1)//2, q) == q-1
    a = None
    for x in range(1, q):
        if pow(x, (q - 1) // 2, q) == q - 1:
            a = x
            break
    
    if a is None:
        return None, None
        
    for u in range(q):
        u2 = (u * u) % q
        v1 = u2
        v2 = (u2 + a) % q
        v3 = (u2 - a) % q
        
        # Add edges (u, q+v)
        edges.append((u, q + v1))
        edges.append((u, q + v2))
        edges.append((u, q + v3))
        
    return n, edges

def _check_girth(n, edges, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex to find shortest cycle
    # Since g is small (16), we can do BFS up to depth g/2
    # Actually, we need to check for cycles of length < g
    # A cycle of length L corresponds to a path of length L from v to v
    # In a bipartite graph, all cycles are even.
    
    # Optimization: Only need to check cycles of length 4, 6, ..., g-2
    # We can use BFS from each vertex, but that is O(n^2 * g)
    # n is small (<= 2000), so this is fine.
    
    for start in range(n):
        # dist[v] = distance from start
        # prev[v] = previous vertex in BFS tree
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                continue
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    # Found a cycle
                    # Cycle length = dist[u] + dist[v] + 1
                    # But this is for undirected graph.
                    # If u and v are already visited, and not parent, we found a cycle.
                    # We need to be careful not to count the edge back to parent.
                    if v != u: # No loops
                        # Check if v is parent of u?
                        # We don't store parent, so we just check if dist[u] + dist[v] + 1 < g
                        # This might overcount if the path is not simple, but BFS guarantees shortest path.
                        # If dist[u] + dist[v] + 1 < g, there is a cycle of that length.
                        if dist[u] + dist[v] + 1 < g:
                            return False
                            
    return True

def _luw_fallback(k, g):
    # The original LUW construction
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
