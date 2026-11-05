"""Deterministic greedy tree-folding search for the (4,10)-cage."""
import sys
import time
from collections import deque

# --- Finite-field helpers for GF(4) (char-2) and GF(k) (prime) ---
def _gf4_ops():
    add = lambda a, b: a ^ b
    table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
    ]
    mul = lambda a, b: table[a][b]
    return add, mul

def _gfp_ops(p):
    add = lambda a, b: (a + b) % p
    mul = lambda a, b: (a * b) % p
    return add, mul

# --- LUW / Lazebnik–Ustimenko–Woldar construction (fallback) ---
def _luw_construct(k, g):
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1

    if q == 4:
        add, mul = _gf4_ops()
        sub = add
    else:
        add, mul = _gfp_ops(q)
        sub = lambda a, b: (a - b) % q

    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

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
            yield (1 - side), tuple(y)

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

# --- Greedy tree-folding search using depth-5 tree ---
def _greedy_fold_search(k, g, time_limit):
    """Attempt to find a k-regular graph of girth >= g via greedy folding of the infinite tree.
    Uses a tree of depth D=(g-1)//2 + 1 so that boundary vertices can be at distance >= g from each other.
    Returns (n, edges) or None if no valid graph found within time_limit."""
    start_time = time.time()
    
    # Use depth D = (g+1)//2 so max boundary distance is 2*D >= g
    # For g=10, D=5, max distance=10
    D = (g + 1) // 2
    
    # Generate tree vertices as tuples (path from root)
    vertices = [()]
    for depth in range(1, D + 1):
        current_len = len(vertices)
        for i in range(current_len):
            v = vertices[i]
            for c in range(k):
                vertices.append(v + (c,))
    
    n_tree = len(vertices)
    vindex = {v: i for i, v in enumerate(vertices)}
    
    # Build tree adjacency
    tree_adj = [[] for _ in range(n_tree)]
    for i, v in enumerate(vertices):
        if len(v) == 0:
            continue
        parent = v[:-1]
        pidx = vindex[parent]
        tree_adj[i].append(pidx)
        tree_adj[pidx].append(i)
    
    boundary = [i for i, v in enumerate(vertices) if len(v) == D]
    
    def lca_depth(u, v):
        d = 0
        for i in range(min(len(u), len(v))):
            if u[i] == v[i]:
                d = i + 1
            else:
                break
        return d
    
    def tree_dist(u, v):
        ld = lca_depth(u, v)
        return len(u) + len(v) - 2 * ld
    
    # Each boundary vertex needs k-1 more edges
    degree_needed = {i: k - 1 for i in boundary}
    new_edges = []
    
    boundary_sorted = sorted(boundary, key=lambda i: vertices[i])
    
    for i in boundary_sorted:
        if degree_needed[i] <= 0:
            continue
        vi = vertices[i]
        candidates = []
        for j in boundary:
            if j == i or degree_needed[j] <= 0:
                continue
            vj = vertices[j]
            d = tree_dist(vi, vj)
            if d + 1 >= g:  # cycle length would be at least d+1
                candidates.append((d, j))
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        taken = 0
        for d, j in candidates:
            if taken >= degree_needed[i]:
                break
            new_edges.append((i, j))
            degree_needed[i] -= 1
            degree_needed[j] -= 1
            taken += 1
        
        if degree_needed[i] > 0:
            return None
    
    for i in boundary:
        if degree_needed[i] != 0:
            return None
    
    edges = []
    for u in range(n_tree):
        for v in tree_adj[u]:
            if u < v:
                edges.append((u, v))
    edges.extend(new_edges)
    
    if not _validate_graph(n_tree, edges, k, g):
        return None
    
    return n_tree, edges

def construct(k, g):
    """Construct a k-regular graph with girth >= g."""
    start_time = time.time()
    
    result = _greedy_fold_search(k, g, time_limit=5.0)
    
    if result is not None:
        n, edges = result
        return n, edges
    
    return _luw_construct(k, g)

def _validate_graph(n, edges, k, g):
    """Validate that the graph is k-regular, simple, and has girth >= g."""
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        e = (min(u, v), max(u, v))
        if e in edge_set:
            return False
        edge_set.add(e)
    
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    if any(d != k for d in degree):
        return False
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                elif abs(dist[v] - dist[u]) <= 1 and dist[v] != dist[u] + 1:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
