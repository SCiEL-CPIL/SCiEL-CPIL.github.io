"""Approach: voltage graph lifting + random 4-regular graph search + local repair."""
import random
import numpy as np
from collections import deque

def _is_simple_4reg(n, edges):
    """Check simple and 4-regular."""
    deg = [0] * n
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        a, b = min(u, v), max(u, v)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[u] += 1
        deg[v] += 1
    return all(d == 4 for d in deg)

def _girth_check(n, edges, min_girth):
    """Return True if no cycle shorter than min_girth. Uses BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for start in range(n):
        # BFS tracking (node, parent, depth)
        visited = [(-1, -1) for _ in range(n)]  # (parent, depth)
        visited[start] = (-1, 0)
        queue = deque([(start, -1, 0)])
        while queue:
            node, parent, depth = queue.popleft()
            if depth >= min_girth - 1:
                continue
            for nb in adj[node]:
                if nb == parent:
                    continue
                if visited[nb][0] == -1 and visited[nb][1] == -1:
                    visited[nb] = (node, depth + 1)
                    queue.append((nb, node, depth + 1))
                else:
                    p, d = visited[nb]
                    if p != -1 and d != -1:
                        cycle_len = depth + 1 + d
                        if cycle_len < min_girth:
                            return False
    return True

def _random_4reg(n, seed=0):
    """Generate a random 4-regular graph on n vertices. Returns edges or None."""
    if n < 4 or n % 2 == 1:
        return None
    rng = random.Random(seed)
    # Half-edge method: create 2n half-edges, pair them randomly
    # Each vertex has 4 stubs
    stubs = []
    for v in range(n):
        for _ in range(4):
            stubs.append(v)
    edges = []
    used = [False] * len(stubs)
    attempts = 0
    while attempts < 1000:
        attempts += 1
        perm = list(range(len(stubs)))
        rng.shuffle(perm)
        edges = []
        valid = True
        for i in range(0, len(perm), 2):
            a, b = perm[i], perm[i + 1]
            u, v = stubs[a], stubs[b]
            if u == v:
                valid = False
                break
            edges.append((u, v))
        if not valid:
            continue
        # Check for multi-edges
        seen = set()
        valid = True
        for u, v in edges:
            a, b = min(u, v), max(u, v)
            if (a, b) in seen:
                valid = False
                break
            seen.add((a, b))
        if valid:
            return edges
    return None

def _voltage_lift(base_adj, group_size, voltages):
    """
    Lift a base graph via voltage assignment.
    base_adj: list of lists of (neighbor, voltage)
    group_size: size of abelian group (Z_m)
    voltages: not used, base_adj already contains voltages
    Returns (n, edges) for the lift.
    """
    n_base = len(base_adj)
    n = n_base * group_size
    edges = []
    for u in range(n_base):
        for v, volt in base_adj[u]:
            for g in range(group_size):
                u_lift = u * group_size + g
                v_lift = v * group_size + (g + volt) % group_size
                if u_lift < v_lift:
                    edges.append((u_lift, v_lift))
    return n, edges

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g on as few vertices as possible.
    For k=4, g=10, we try multiple strategies.
    """
    if k != 4:
        # Fallback: use the original approach for other k
        return _construct_general(k, g)
    
    # Moore bound for 4-regular girth 10: 2 + 4*sum_{i=0}^{4} 3^i = 2 + 4*(1+3+9+27+81) = 2+4*121 = 486? 
    # Actually Moore bound for even girth g=10: 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2*(1+3+9+27+81) = 2*121 = 242
    # So lower bound is 242.
    
    best_n = float('inf')
    best_edges = None
    
    # Strategy 1: Voltage graph with small base
    # Try base = K_5 (5 vertices, 4-regular), lift with group of size m
    # K_5 has triangles, so we need careful voltage assignment to avoid short cycles.
    # Actually, K_5 has girth 3, so lifting may not help directly.
    
    # Better: use a base graph with larger girth.
    # For 4-regular, the smallest base with girth >= 5 is the Petersen graph (10 vertices).
    # Lift Petersen with group of size m: n = 10*m.
    # Petersen has girth 5. If we assign voltages carefully, the lift can have girth >= 10.
    
    # Petersen graph adjacency with voltage assignment
    # Petersen: vertices 0..9, edges:
    # Outer cycle: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),  # outer
        (5,7), (7,9), (9,6), (6,8), (8,5),  # inner
        (0,5), (1,6), (2,7), (3,8), (4,9)   # spokes
    ]
    
    # Try various group sizes for Petersen lift
    rng = random.Random(42)
    for m in [24, 25, 26, 28, 30, 32, 35, 40, 48, 50, 60, 80, 100, 120, 160, 200, 240, 320]:
        n = 10 * m
        if n >= best_n:
            continue
        # Random voltage assignment in Z_m
        for trial in range(20):
            voltages = {}
            for u, v in petersen_edges:
                voltages[(u, v)] = rng.randint(0, m - 1)
                voltages[(v, u)] = (-voltages[(u, v)]) % m
            # Build base_adj with voltages
            base_adj = [[] for _ in range(10)]
            for u, v in petersen_edges:
                base_adj[u].append((v, voltages[(u, v)]))
            n_lift, edges = _voltage_lift(base_adj, m, None)
            # Check girth
            if _girth_check(n_lift, edges, g):
                if n_lift < best_n:
                    best_n = n_lift
                    best_edges = edges
                    break
    
    # Strategy 2: Random 4-regular graphs with local repair
    if best_edges is None:
        for n_try in range(242, 350, 2):  # even numbers, 4-reg requires n even
            if n_try % 2 != 0:
                continue
            for seed in range(50):
                edges = _random_4reg(n_try, seed)
                if edges is None:
                    continue
                if _girth_check(n_try, edges, g):
                    if n_try < best_n:
                        best_n = n_try
                        best_edges = edges
                    break
    
    # Strategy 3: Local repair from random graph
    if best_edges is None:
        for n_try in [244, 246, 248, 250, 252, 254, 256, 260, 264, 270, 280, 300, 320]:
            for seed in range(30):
                edges = _random_4reg(n_try, seed)
                if edges is None:
                    continue
                # Try to repair short cycles
                repaired = _repair_girth(n_try, edges, g, max_iter=200)
                if repaired is not None:
                    if n_try < best_n:
                        best_n = n_try
                        best_edges = repaired
                    break
    
    # Fallback: use a known construction if nothing worked
    if best_edges is None:
        # Use the original Lazebnik-Ustimenko-Woldar as fallback
        return _construct_luw(k, g)
    
    return best_n, best_edges

def _repair_girth(n, edges, target_girth, max_iter=100):
    """Try to repair short cycles by rewiring."""
    edges = list(edges)
    for _ in range(max_iter):
        # Find a short cycle
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        short_cycle = None
        for start in range(n):
            # BFS to find short cycle
            visited = [(-1, -1) for _ in range(n)]
            visited[start] = (-1, 0)
            queue = deque([(start, -1, 0, [start])])
            while queue:
                node, parent, depth, path = queue.popleft()
                if depth >= target_girth - 1:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if visited[nb][0] == -1 and visited[nb][1] == -1:
                        visited[nb] = (node, depth + 1)
                        queue.append((nb, node, depth + 1, path + [nb]))
                    else:
                        p, d = visited[nb]
                        if p != -1 and d != -1:
                            cycle_len = depth + 1 + d
                            if cycle_len < target_girth:
                                short_cycle = path + [nb]
                                break
                if short_cycle:
                    break
            if short_cycle:
                break
        
        if short_cycle is None:
            return edges
        
        # Try to break the cycle by rewiring
        # Find an edge in the cycle and try to replace it
        cycle_edges = set()
        for i in range(len(short_cycle)):
            a, b = short_cycle[i], short_cycle[(i + 1) % len(short_cycle)]
            cycle_edges.add((min(a, b), max(a, b)))
        
        improved = False
        for a, b in list(cycle_edges):
            # Find alternative neighbors
            adj_a = [x for x in adj[a] if x != b]
            adj_b = [x for x in adj[b] if x != a]
            for c in adj_a:
                for d in adj_b:
                    if c == d:
                        continue
                    # Check if adding (a,c) and (b,d) creates multi-edges or loops
                    if a == c or b == d:
                        continue
                    new_edges = [e for e in edges if e != (a, b) and e != (b, a)]
                    new_edges.append((a, c))
                    new_edges.append((b, d))
                    # Check simplicity
                    seen = set()
                    valid = True
                    for u, v in new_edges:
                        if u == v:
                            valid = False
                            break
                        x, y = min(u, v), max(u, v)
                        if (x, y) in seen:
                            valid = False
                            break
                        seen.add((x, y))
                    if not valid:
                        continue
                    # Check degree
                    deg = [0] * n
                    for u, v in new_edges:
                        deg[u] += 1
                        deg[v] += 1
                    if any(d != 4 for d in deg):
                        continue
                    # Check girth
                    if _girth_check(n, new_edges, target_girth):
                        edges = new_edges
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    if _girth_check(n, edges, target_girth):
        return edges
    return None

def _construct_luw(k, g):
    """Original Lazebnik-Ustimenko-Woldar fallback."""
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

def _construct_general(k, g):
    """General fallback for k != 4."""
    return _construct_luw(k, g)
