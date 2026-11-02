import time
import math
import random
import numpy as np
from collections import deque

def construct(k, g):
    deadline = time.time() + 115
    
    if k != 3:
        return _luw_fallback(k, g)
        
    r = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(r))
    if moore_bound % 2 != 0:
        moore_bound += 1
        
    # Phase 1: Circulant Search with expanded step sets
    # Try {a, n-a, h} and {a, b, n-a-b} type circulants
    max_circ_n = 1500
    for n in range(moore_bound, max_circ_n + 1, 2):
        if time.time() > deadline - 30: # Reserve time for random phase
            break
        h = n // 2
        # Try step set {a, n-a, h}
        for a in range(1, h):
            if math.gcd(h, a) != 1:
                continue
            if _check_girth_circulant_numpy(n, [a, n - a, h], g):
                edges = []
                for u in range(n):
                    for s in (a, n - a, h):
                        v = (u + s) % n
                        if u < v:
                            edges.append((u, v))
                return n, edges
        # Try step set {a, b, n-a-b} where a+b < n/2
        # This gives another family of 3-regular circulants
        for a in range(1, n // 3):
            for b in range(a + 1, n // 2 - a):
                c = n - a - b
                if c <= b:
                    break
                steps = [a, b, c]
                if _check_girth_circulant_numpy(n, steps, g):
                    edges = []
                    for u in range(n):
                        for s in steps:
                            v = (u + s) % n
                            if u < v:
                                edges.append((u, v))
                    return n, edges
                    
    # Phase 2: Random Graph Search
    # Use a better girth check with proper LCA computation
    n = moore_bound
    while n < 1500:
        if time.time() > deadline - 10:
            break
            
        # Try random graphs for this n
        attempts = 0
        max_attempts = 30
        
        while attempts < max_attempts:
            if time.time() > deadline - 10:
                break
                
            adj = _generate_random_3_reg(n)
            if adj is None:
                attempts += 1
                continue
                
            if _check_girth_general(adj, g):
                edges = _get_edges_from_adj(adj)
                return n, edges
                
            attempts += 1
            
        n += 2
        
    return _luu_fallback(k, g)

def _check_girth_circulant_numpy(n, steps, g):
    """
    Check if circulant graph C(n; steps) has girth >= g.
    Uses non-backtracking walk enumeration from vertex 0.
    """
    # For a 3-regular circulant with steps S = [s1, s2, s3],
    # the inverse of s_i is (n - s_i) % n.
    # We need to track which step was used to avoid backtracking.
    
    # Map each step to its inverse index
    step_map = {}
    for i, s in enumerate(steps):
        inv = (n - s) % n
        step_map[inv] = i
    
    # R[d, last_step_type, vertex] = True if vertex is reachable
    # from 0 in d non-backtracking steps, with last step of type last_step_type
    # Initialize: depth 1
    R = np.zeros((g, 3, n), dtype=bool)
    for i, s in enumerate(steps):
        R[1, i, s % n] = True
    
    for d in range(2, g):
        for i in range(3):  # last step type from previous depth
            for j in range(3):  # current step type
                if step_map.get(steps[j], -1) == i:
                    continue  # backtracking
                # From all vertices reachable at depth d-1 with last step i,
                # take step j
                R[d, j] |= np.roll(R[d-1, i], steps[j])
        # Check if we've returned to vertex 0
        if R[d, :, 0].any():
            return False
    
    return True

def _generate_random_3_reg(n):
    """
    Generate a random 3-regular simple graph using configuration model
    with retries to avoid loops and multi-edges.
    """
    for _ in range(20):  # Try up to 20 times
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        
        random.shuffle(stubs)
        
        adj = [set() for _ in range(n)]
        valid = True
        
        for i in range(0, 3*n, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        
        if not valid:
            continue
            
        # Verify degree
        ok = True
        for i in range(n):
            if len(adj[i]) != 3:
                ok = False
                break
        if ok:
            return adj
            
    return None

def _check_girth_general(adj, g):
    """
    Check if graph has girth >= g using BFS from each vertex.
    For each edge (u,v) not in the BFS tree, compute the actual cycle length
    by finding the LCA of u and v in the BFS tree.
    """
    n = len(adj)
    max_depth = (g - 1) // 2  # We only need to explore this deep
    
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        q = deque([start])
        dist[start] = 0
        parent[start] = start
        
        while q:
            u = q.popleft()
            du = dist[u]
            
            if du >= max_depth:
                continue  # Don't expand beyond max_depth
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < max_depth:
                        q.append(v)
                else:
                    # v is already visited
                    # Skip if it's the tree edge (parent-child in BFS tree)
                    if v == parent[u]:
                        continue
                    if u == parent[v]:
                        continue  # Already checked when processing v
                    
                    # Compute LCA of u and v to get actual cycle length
                    # Walk up from u and v to find common ancestor
                    u_anc = u
                    v_anc = v
                    dist_u = dist[u]
                    dist_v = dist[v]
                    
                    # Bring to same depth
                    while dist_u > dist_v:
                        u_anc = parent[u_anc]
                        dist_u -= 1
                    while dist_v > dist_u:
                        v_anc = parent[v_anc]
                        dist_v -= 1
                    
                    # Walk up together
                    while u_anc != v_anc:
                        u_anc = parent[u_anc]
                        v_anc = parent[v_anc]
                    
                    lca = u_anc
                    cycle_len = (dist[u] - dist[lca]) + (dist[v] - dist[lca]) + 1
                    
                    if cycle_len < g:
                        return False
                        
    return True

def _get_edges_from_adj(adj):
    edges = []
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges

def _luu_fallback(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for k=3, girth >= g.
    Returns a 3-regular graph with girth at least g.
    """
    q = k  # q = 3
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
