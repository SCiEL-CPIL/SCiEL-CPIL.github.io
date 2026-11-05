import random
import time
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit

    # Known small cages and Moore graphs
    if k == 2:
        # Cycle graph C_g
        n = g
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges

    if k == 3:
        # Known (3,g) cages
        if g == 3:
            n = 3
            edges = [(0,1),(1,2),(0,2)]
            return n, edges
        if g == 4:
            n = 4
            edges = [(0,1),(1,2),(2,3),(3,0)]
            return n, edges
        if g == 5:
            n = 5
            edges = [(0,1),(1,2),(2,3),(3,4),(4,0)]
            return n, edges
        if g == 6:
            n = 10
            # Petersen graph
            edges = [(0,1),(1,2),(2,3),(3,4),(4,0),
                     (5,6),(6,7),(7,8),(8,9),(9,5),
                     (0,5),(1,6),(2,7),(3,8),(4,9)]
            return n, edges
        if g == 7:
            n = 24
            # (3,7) cage has 24 vertices
            # Use a known construction or LUW
            pass

    # For k=4, g=10: try voltage graph first, then LUW as fallback
    if k == 4 and g == 10:
        # Try voltage graph construction
        voltage_result = _voltage_graph_construct(k, g, deadline)
        if voltage_result is not None:
            n, edges = voltage_result
            if _verify(n, edges, k, g):
                return n, edges

        # Fallback: LUW
        return _luw_graph(k, g)

    # For other (k,g), use LUW
    return _luw_graph(k, g)


def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 5:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    elif q == 7:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    elif q == 3:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    else:
        # For other primes, use modular arithmetic
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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
    
    n = len(order)
    edges = sorted(set(edges))
    return n, edges


def _voltage_graph_construct(k, g, deadline):
    """Construct a 4-regular girth-g graph using voltage graph (lift) over Z_m."""
    if k != 4:
        return None

    best_result = None
    best_n = float('inf')

    # Base graph: K_{4,4} (8 vertices, 4-regular, girth 4)
    base_edges = _get_k44_edges()
    base_n = 8

    # Precompute non-backtracking closed walks of length 3 to g-1
    closed_walks = _enumerate_closed_walks(base_edges, g)

    if not closed_walks:
        # No short closed walks, so any m works
        m = 2
        n = m * base_n
        voltages = [0] * len(base_edges)
        lift_edges = _build_lift(base_edges, voltages, m)
        if _verify(n, lift_edges, k, g):
            return n, lift_edges
        return None

    # Try different m values
    for m in range(2, 100):
        if time.time() > deadline - 10:
            break
        n = m * base_n
        if n >= best_n:
            continue

        # Try to find voltage assignment with girth >= g
        voltages = _find_voltages(closed_walks, m, g, deadline - 5)
        if voltages is not None:
            lift_edges = _build_lift(base_edges, voltages, m)
            if _verify(n, lift_edges, k, g):
                if n < best_n:
                    best_n = n
                    best_result = (n, lift_edges)

    return best_result


def _get_k44_edges():
    """Get edges of K_{4,4}: vertices 0-3 on one side, 4-7 on other."""
    edges = []
    for i in range(4):
        for j in range(4):
            edges.append((i, j + 4))
    return edges


def _enumerate_closed_walks(base_edges, g):
    """Enumerate all non-backtracking closed walks of length 3 to g-1 in the base graph."""
    base_n = len(set(v for e in base_edges for v in e))
    
    # Build adjacency with edge indices
    adj = [[] for _ in range(base_n)]
    for idx, (u, v) in enumerate(base_edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    closed_walks = []
    
    for L in range(3, g):
        # BFS from each directed edge
        for start_u, start_v in base_edges:
            # Find the edge index for (start_u, start_v)
            start_eidx = None
            for idx, (u, v) in enumerate(base_edges):
                if (u == start_u and v == start_v) or (u == start_v and v == start_u):
                    start_eidx = idx
                    break
            if start_eidx is None:
                continue
            
            # Determine sign: if base_edges[start_eidx] = (start_u, start_v), sign=+1, else -1
            u0, v0 = base_edges[start_eidx]
            if u0 == start_u and v0 == start_v:
                start_sign = 1
            else:
                start_sign = -1
            
            # BFS: state is (prev, cur, path)
            # path is list of (eidx, sign)
            queue = deque()
            queue.append((start_u, start_v, [(start_eidx, start_sign)]))
            
            while queue:
                prev, cur, path = queue.popleft()
                if len(path) == L:
                    if cur == start_u:
                        # Closed walk!
                        closed_walks.append(path[:])
                    continue
                for (next_v, eidx) in adj[cur]:
                    if next_v == prev:
                        continue  # No backtracking
                    # Determine sign for this edge
                    u1, v1 = base_edges[eidx]
                    if u1 == cur and v1 == next_v:
                        sign = 1
                    else:
                        sign = -1
                    queue.append((cur, next_v, path + [(eidx, sign)]))
    
    return closed_walks


def _find_voltages(closed_walks, m, g, deadline):
    """Find voltage assignment so that no closed walk has sum 0 mod m."""
    n_edges = len(closed_walks[0]) if closed_walks else 0
    # Actually, n_edges should be the number of base edges
    # Let's get it from the walks
    if not closed_walks:
        return None
    
    # Get number of edges from the first walk
    # The walks contain (eidx, sign) pairs, so eidx ranges over base edges
    max_eidx = max(eidx for walk in closed_walks for eidx, sign in walk)
    n_edges = max_eidx + 1
    
    # Initialize voltages randomly
    random.seed(42)  # Deterministic for reproducibility
    voltages = [random.randint(0, m - 1) for _ in range(n_edges)]
    
    def count_violations(vols):
        count = 0
        for walk in closed_walks:
            s = 0
            for eidx, sign in walk:
                s = (s + sign * vols[eidx]) % m
            if s == 0:
                count += 1
        return count
    
    current_violations = count_violations(voltages)
    if current_violations == 0:
        return voltages[:]
    
    # Simulated annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.99
    best_voltages = voltages[:]
    best_violations = current_violations
    
    while T > T_min and time.time() < deadline:
        # Pick a random edge to change
        eidx = random.randint(0, n_edges - 1)
        old_val = voltages[eidx]
        new_val = random.randint(0, m - 1)
        voltages[eidx] = new_val
        
        new_violations = count_violations(voltages)
        
        if new_violations < current_violations or random.random() < pow(2.718281828, (current_violations - new_violations) / T):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_voltages = voltages[:]
            if new_violations == 0:
                return voltages[:]
        else:
            voltages[eidx] = old_val
        
        T *= cooling
    
    if best_violations == 0:
        return best_voltages
    
    return None


def _build_lift(base_edges, voltages, m):
    """Build the lifted graph from base edges and voltages."""
    base_n = len(set(v for e in base_edges for v in e))
    n = m * base_n
    edges = []
    
    for (u, v), volt in zip(base_edges, voltages):
        for i in range(m):
            u_lift = u * m + i
            v_lift = v * m + ((i + volt) % m)
            edges.append((u_lift, v_lift))
    
    return edges


def _verify(n, edges, k, g):
    """Verify that the graph is k-regular and has girth at least g."""
    # Check regularity
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if (min(u, v), max(u, v)) in edge_set:
            return False
        edge_set.add((min(u, v), max(u, v)))
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check girth using BFS from each vertex
    for start in range(n):
        # BFS to find shortest cycle through start
        # Use non-backtracking BFS
        visited = {}
        queue = deque()
        
        for neighbor in adj[start]:
            visited[(start, neighbor)] = 1
            queue.append((start, neighbor, 1))
        
        while queue:
            prev, cur, dist = queue.popleft()
            if dist >= g:
                continue
            for next_v in adj[cur]:
                if next_v == prev:
                    continue
                if next_v == start:
                    return False  # Found cycle of length dist + 1 < g
                if (cur, next_v) not in visited:
                    visited[(cur, next_v)] = dist + 1
                    queue.append((cur, next_v, dist + 1))
    
    return True
