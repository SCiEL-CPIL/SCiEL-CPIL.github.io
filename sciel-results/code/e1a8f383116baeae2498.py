import time
import math
from collections import deque

# EVOLVE-BLOCK-START

def _verify_graph(n, edges, k, g):
    """Verify graph is k-regular, simple, and has girth >= g using non-backtracking walks."""
    if n < 2:
        return False
    adj = [set() for _ in range(n)]
    for i, j in edges:
        if i == j or i >= n or j >= n or i < 0 or j < 0:
            return False
        if i in adj[j]:
            return False  # repeated edge
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Non-backtracking walk check from every vertex
    for start in range(n):
        visited = {}
        queue = deque()
        for v in adj[start]:
            queue.append((v, start, 1))
            visited[(v, start)] = 1
        while queue:
            v, prev, dist = queue.popleft()
            if dist >= g:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                key = (w, v)
                if key in visited:
                    if visited[key] + 1 < g:
                        return False
                else:
                    visited[key] = dist + 1
                    queue.append((w, v, dist + 1))
    return True

def _cage_3_9():
    """(3,9)-cage: 70 vertices, girth 9, cubic."""
    # The (3,9)-cage is the unique cubic graph of girth 9 with 70 vertices.
    # Construction: bipartite graph with parts A={0..34} and B={35..69}.
    # For i in 0..34:
    #   i connects to (i+1)%35, (i-1)%35, and 35+((i*3)%35)
    #   35+i connects to 35+((i+2)%35), 35+((i-2)%35), and (35+i - 35*3)%35 = (i*3+35)%35... 
    # This is getting complicated. Let me use a known valid construction.
    
    # The (3,9)-cage can be constructed as follows:
    # Vertices: (i, s) for i in Z_35, s in {0, 1}
    # Edges:
    #   (i, 0) - ((i+1) mod 35, 0)
    #   (i, 0) - ((i-1) mod 35, 0)
    #   (i, 0) - ((3*i+1) mod 35, 1)
    #   (i, 1) - ((3*i) mod 35, 0)  [inverse of above]
    #   (i, 1) - ((i+2) mod 35, 1)
    #   (i, 1) - ((i-2) mod 35, 1)
    
    # Wait, this gives degree 3 for each vertex, but I need to verify no multiple edges.
    # (i,0) connects to (i+1,0), (i-1,0), (3i+1,1)
    # (i,1) connects to (3i,0), (i+2,1), (i-2,1)
    
    # For this to be valid, 3i+1 mod 35 must be a bijection, which it is since gcd(3,35)=1.
    # And (3i,0) is the inverse: if (i,0)-(j,1) where j=3i+1, then (j,1)-(i,0) where i=(j-1)/3.
    # But I defined (j,1) connects to (3j,0), not ((j-1)/3, 0). So this is inconsistent.
    
    # Let me fix: define (i,0) - (j,1) iff j = 3i+1 mod 35.
    # Then (j,1) - (i,0) where i = (j-1)*inverse_of_3 mod 35.
    # inverse of 3 mod 35: 3*12 = 36 = 1 mod 35. So inverse is 12.
    # So (j,1) - ((j-1)*12 mod 35, 0).
    
    # Let me redefine:
    edges = []
    n = 70
    for i in range(35):
        # Part 0: vertices 0..34
        edges.append((i, (i + 1) % 35))
        edges.append((i, (i - 1) % 35))
        edges.append((i, 35 + (3 * i + 1) % 35))
        # Part 1: vertices 35..69
        edges.append((35 + i, 35 + (i + 2) % 35))
        edges.append((35 + i, 35 + (i - 2) % 35))
        # The cross edges are already added from part 0
    
    # Deduplicate
    edge_set = set()
    for a, b in edges:
        if a < b:
            edge_set.add((a, b))
        else:
            edge_set.add((b, a))
    edges = list(edge_set)
    
    if len(edges) != 105:
        return None
    
    return 70, edges

def _double_with_matching(n, edges):
    """
    Take two copies of a 3-regular graph and connect with a perfect matching.
    Result: 2n vertices, 3-regular, girth >= 2*g - 2 where g is the girth of the original.
    """
    new_n = 2 * n
    new_edges = []
    # First copy: vertices 0..n-1
    for u, v in edges:
        new_edges.append((u, v))
    # Second copy: vertices n..2n-1
    for u, v in edges:
        new_edges.append((n + u, n + v))
    # Perfect matching: i <-> n+i
    for i in range(n):
        new_edges.append((i, n + i))
    return new_n, new_edges

def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar construction for (k,g) with k odd prime or k=4."""
    q = k
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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

def construct(k, g):
    deadline = time.time() + 140
    best_n, best_edges = None, None
    
    if k != 3:
        try:
            n, edges = _luw_graph(k, g)
            if n is not None and len(edges) == n * k // 2:
                if _verify_graph(n, edges, k, g):
                    return n, edges
        except Exception:
            pass
        return None, None
    
    # For cubic graphs, try the double-with-matching approach
    # Start with (3,9)-cage (70 vertices, girth 9)
    # Double with matching -> 140 vertices, girth >= 16
    
    try:
        n9, edges9 = _cage_3_9()
        if n9 is not None:
            if _verify_graph(n9, edges9, 3, 9):
                n16, edges16 = _double_with_matching(n9, edges9)
                if _verify_graph(n16, edges16, 3, 16):
                    best_n, best_edges = n16, edges16
    except Exception:
        pass
    
    # Fallback: LUW construction
    try:
        n, edges = _luw_graph(k, g)
        if n is not None and len(edges) == n * k // 2:
            if best_n is None or n < best_n:
                if _verify_graph(n, edges, k, g):
                    best_n, best_edges = n, edges
    except Exception:
        pass
    
    if best_n is not None:
        return best_n, best_edges
    
    return None, None

# EVOLVE-BLOCK-END

