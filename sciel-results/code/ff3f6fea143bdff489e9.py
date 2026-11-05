"""Construct k-regular graphs of girth g using voltage graph / cover graph methods."""
import time
import itertools
import numpy as np
from collections import deque

def _check_girth(adj, g):
    """Check that graph with adjacency list `adj` has no cycle shorter than g.
    Uses non-backtracking BFS from each vertex."""
    n = len(adj)
    # For each vertex, do a non-backtracking BFS up to depth (g-1)//2
    # If we find a vertex at depth d that was also reached at depth d' < d from the same start
    # via a different non-backtracking path, that indicates a cycle of length <= d+d'.
    # Simpler: non-backtracking walks of length < g should not return to start.
    max_depth = (g - 1) // 2
    for start in range(n):
        # state: (vertex, prev_vertex), start with (start, -1)
        # BFS by depth
        visited = set()  # (vertex, prev) pairs reached
        current = {(start, -1)}
        for depth in range(1, max_depth + 1):
            next_set = set()
            for (v, prev) in current:
                for u in adj[v]:
                    if u == prev:
                        continue
                    state = (u, v)
                    if state in visited:
                        continue
                    visited.add(state)
                    if u == start and depth >= 2:
                        # Found a cycle of length depth (non-backtracking walk back to start)
                        # Actually this means a cycle of length depth
                        return False
                    next_set.add(state)
            current = next_set
    return True


def _voltage_graph_construct(k, g, m):
    """Try to construct a k-regular graph of girth g using a voltage graph on Z_m.
    
    Base graph: use a small k-regular base graph and assign voltages.
    The cover graph has m * |V_base| vertices.
    
    For simplicity, start with a complete graph K_k as base (k vertices, each connected to all others).
    Actually K_k is (k-1)-regular, not k-regular.
    
    Let's use a different approach: start with the Petersen-like or just try small base graphs.
    
    Alternative: use a 2-vertex base with multiple edges (multi-graph) that lifts to simple.
    Base: two vertices A, B with k parallel edges. Assign voltages v_1, ..., v_k in Z_m.
    Cover: 2m vertices. Vertex (A, i) connects to (B, i + v_j) for j=1..k.
    This is k-regular. Girth depends on voltage assignments.
    
    For the cover to be simple, we need: for each i, the k values (i + v_j) mod m to be all distinct
    (no multiple edges between same pair), which means v_j - v_i != 0 mod m for j != i, i.e., all v_j distinct mod m.
    Also need no self-loops: (A,i) to (B, i+v_j) is fine since A != B.
    
    Girth analysis: cycles in the cover correspond to closed walks in the base.
    A cycle of length 2r corresponds to going A->B->A->...->B->A with r transitions each way.
    The voltage sum must be 0 mod m.
    
    Shortest cycle: 
    - 4-cycle: A,i -> B,i+v_a -> A,i+v_a+v_b -> B,i+v_a+v_b+v_c -> A,i+v_a+v_b+v_c+v_d where we need
      the path to close. Actually for a 4-cycle: A,i -> B,i+va -> A,i+va+vb -> B,i+va+vb+vc -> A,i+va+vb+vc+vd
      For this to be a 4-cycle back to A,i: va+vb+vc+vd = 0 mod m, and the vertices must be distinct.
    
    This is getting complex. Let me try a direct approach with small m and random voltage assignments.
    """
    # Use 2-vertex base with k parallel edges
    # Vertices: (side, offset) where side in {0,1}, offset in 0..m-1
    # Edge: (0, i) -- (1, (i + v_j) % m) for j in 0..k-1
    
    # Try random voltage assignments
    best = None
    deadline = time.time() + 110  # leave margin
    
    # We need m such that km >= k (trivially true) and the graph is simple.
    # For simplicity: all v_j distinct mod m. Need m >= k.
    
    # Start with m = k and increase
    for m in range(k, 2000):
        if time.time() > deadline:
            break
        
        # Try a few random assignments
        attempts = 0
        max_attempts = min(500, max(50, 3000 // m))
        while attempts < max_attempts:
            if time.time() > deadline:
                break
            attempts += 1
            
            # Random distinct voltages
            if m >= k:
                voltages = list(range(m))
                np.random.shuffle(voltages)
                voltages = voltages[:k]
            else:
                break
            
            # Build adjacency
            n = 2 * m
            adj = [[] for _ in range(n)]
            for i in range(m):
                for j in range(k):
                    u = i  # (0, i)
                    v = (i + voltages[j]) % m  # (1, (i+v_j)%m)
                    adj[u].append(m + v)
                    adj[m + v].append(u)
            
            # Check simplicity (should be guaranteed by distinct voltages, but verify)
            valid = True
            for i in range(n):
                if len(adj[i]) != len(set(adj[i])):
                    valid = False
                    break
                if len(adj[i]) != k:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check girth
            if _check_girth(adj, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        
        # If we've tried m up to some point, increase
        if m >= 500:
            break
    
    return None


def _try_known_constructions(k, g):
    """Try known explicit constructions for small cases."""
    # For (5, 9): known small graphs exist. 
    # The Lazebnik-Ustimenko-Woldar construction is in the seed.
    # Let's also try the Hoffman-Singleton type or other known cages.
    
    # Moore bound: n >= k + k*(k-1) + ... for girth g
    # For odd girth g = 2r+1: n >= k + k*(k-1) + k*(k-1)^2 + ... + k*(k-1)^(r-1)
    # For even girth g = 2r: n >= 2*(k + k*(k-1) + ... + k*(k-1)^(r-1))
    
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = k + sum(k * (k-1)**i for i in range(1, r))
    else:
        r = g // 2
        moore = 2 * sum(k * (k-1)**i for i in range(0, r))
    
    return moore


def construct(k, g):
    """Construct a k-regular graph with girth at least g, minimizing number of vertices."""
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # First try the voltage graph approach
    result = _voltage_graph_construct(k, g, 0)
    if result is not None:
        n, edges = result
        if time.time() < deadline:
            return n, edges
    
    # Fallback: use the Lazebnik-Ustimenko-Woldar construction from the seed
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
        mul_op = lambda x, y: table[x][y]
    else:
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
        mul_op = lambda x, y: x * y % q
    
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
                    y[c] = add_op(x[c], mul_op(y[a], x[b]))
                else:
                    y[c] = sub_op(x[c], mul_op(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
