"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
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


def _girth_check(n, edges, k, g):
    """Check that the graph is k-regular and has girth >= g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check regularity
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check girth via BFS from each vertex
    for start in range(n):
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
                elif v != parent[u] and parent[v] != u:
                    # Found a cycle
                    a, b = u, v
                    da, db = dist[u], dist[v]
                    while a != b:
                        if da > db:
                            a = parent[a]
                            da -= 1
                        elif db > da:
                            b = parent[b]
                            db -= 1
                        else:
                            a = parent[a]
                            b = parent[b]
                            da -= 1
                            db -= 1
                    cycle_len = dist[u] + dist[v] - 2 * da + 1
                    if cycle_len < g:
                        return False
    return True


def _cayley_girth_check(n, mult, g1, g2, g):
    """Check girth of Cayley graph with generators g1, g2 (and their inverses)."""
    e_id = 0
    g1_inv = -1
    g2_inv = -1
    for i in range(n):
        if mult[g1][i] == e_id:
            g1_inv = i
        if mult[g2][i] == e_id:
            g2_inv = i
    
    gens = (g1, g1_inv, g2, g2_inv)
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for s in gens:
                v = mult[u][s]
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and parent[v] != u:
                    a, b = u, v
                    da, db = dist[u], dist[v]
                    while a != b:
                        if da > db:
                            a = parent[a]
                            da -= 1
                        elif db > da:
                            b = parent[b]
                            db -= 1
                        else:
                            a = parent[a]
                            b = parent[b]
                            da -= 1
                            db -= 1
                    cycle_len = dist[u] + dist[v] - 2 * da + 1
                    if cycle_len < g:
                        return False
    return True


def _build_cayley_edges(n, mult, g1, g2):
    """Build edge list for Cayley graph."""
    e_id = 0
    g1_inv = -1
    g2_inv = -1
    for i in range(n):
        if mult[g1][i] == e_id:
            g1_inv = i
        if mult[g2][i] == e_id:
            g2_inv = i
    
    edge_set = set()
    edges = []
    for u in range(n):
        for s in (g1, g1_inv, g2, g2_inv):
            v = mult[u][s]
            if u < v:
                edge_set.add((u, v))
    edges = list(edge_set)
    return edges


def _try_cayley_search(k, g, deadline, max_n=400):
    """Search for Cayley graphs with desired girth."""
    # Try groups of various orders
    # For k=4, we need 2 generators (plus inverses)
    
    # Try cyclic groups Z_n
    for n in range(240, min(max_n + 1, 350)):
        if time.time() > deadline:
            break
        
        # Build multiplication table for Z_n
        mult = [[(i + j) % n for j in range(n)] for i in range(n)]
        
        # Find candidate generators
        # Need elements with order > 2, and g1, g2 not inverses of each other
        candidates = []
        for i in range(1, n):
            # Check if i has order > 2
            if i == n - i:  # order 2
                continue
            candidates.append(i)
        
        # Try pairs
        best_result = None
        for i in range(len(candidates)):
            if time.time() > deadline:
                break
            g1 = candidates[i]
            for j in range(i + 1, len(candidates)):
                if time.time() > deadline:
                    break
                g2 = candidates[j]
                
                # Check distinctness
                if g1 == g2 or g1 == n - g2 or (n - g1) == g2 or (n - g1) == (n - g2):
                    continue
                
                # Check girth
                if _cayley_girth_check(n, mult, g1, g2, g):
                    edges = _build_cayley_edges(n, mult, g1, g2)
                    if len(edges) == n * k // 2:
                        return n, edges
    
    return None


def construct(k, g):
    deadline = time.time() + 120  # 120s total, leave margin
    
    # Start with LUW construction as guaranteed valid baseline
    n_luw, edges_luw = _luw_construct(k, g)
    
    # Verify it's valid
    if not _girth_check(n_luw, edges_luw, k, g):
        return n_luw, edges_luw
    
    best_n = n_luw
    best_edges = edges_luw[:]
    
    # Try Cayley graph search to potentially find smaller valid graphs
    cayley_result = _try_cayley_search(k, g, deadline, max_n=best_n - 1)
    if cayley_result is not None:
        n_cayley, edges_cayley = cayley_result
        if n_cayley < best_n:
            # Verify Cayley graph is valid
            if _girth_check(n_cayley, edges_cayley, k, g):
                best_n = n_cayley
                best_edges = edges_cayley
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
