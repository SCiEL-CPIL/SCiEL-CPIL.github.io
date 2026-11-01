"""Voltage-graph lift construction for small 5-regular graph of girth 9."""
import random
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    # We focus on the task's specific parameters: k=5, g=9.
    # Use a voltage graph (lift) over the Clebsch graph (16 vertices, 5-regular, girth 5).
    # This base graph has fewer short cycles than K_6, making the voltage search easier.
    
    deadline = time.time() + 115  # 115 seconds max (safety margin)
    
    if k != 5 or g != 9:
        return _luw_construct(k, g)
    
    # Generate Clebsch Graph
    base_n = 16
    base_edges = []
    edge_index = {}
    for u in range(base_n):
        for v in range(u + 1, base_n):
            diff = u ^ v
            if bin(diff).count('1') in [1, 4]:
                edge_index[(u, v)] = len(base_edges)
                base_edges.append((u, v))
    n_base_edges = len(base_edges) # 40 edges
    
    # Generate neighbors for fast walk generation
    neighbors = [[] for _ in range(base_n)]
    for u, v in base_edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
    
    def gen_nonbacktracking_closed_walks(length):
        """Generate non-backtracking closed walks of given length in Clebsch, starting at 0."""
        if length < 3:
            return
        
        def walk_gen(pos, cur, prv, path):
            if pos == length - 1:
                # Next must be 0
                if 0 in neighbors[cur] and prv != 0:
                    yield list(path) + [0]
                return
            
            for nv in neighbors[cur]:
                if nv == prv:
                    continue
                path.append(nv)
                yield from walk_gen(pos + 1, nv, cur, path)
                path.pop()
        
        for v1 in neighbors[0]:
            yield from walk_gen(1, v1, 0, [0, v1])
    
    # Collect constraints for lengths 5 to 8 (girth of base is 5, so lengths 3,4 don't exist as non-backtracking closed walks)
    constraints = []
    for L in range(5, 9):
        walks = list(gen_nonbacktracking_closed_walks(L))
        for walk in walks:
            coeff_dict = {}
            for i in range(L):
                u, v = walk[i], walk[i + 1]
                eu, ev = min(u, v), max(u, v)
                eidx = edge_index[(eu, ev)]
                sign = 1 if u < v else -1
                coeff_dict[eidx] = coeff_dict.get(eidx, 0) + sign
            terms = tuple(sorted((eidx, c) for eidx, c in coeff_dict.items() if c != 0))
            if terms:
                constraints.append(terms)
    
    constraints = list(set(constraints))
    n_constraints = len(constraints)
    
    constraint_matrix = np.zeros((n_constraints, n_base_edges), dtype=np.int32)
    for i, terms in enumerate(constraints):
        for eidx, coeff in terms:
            constraint_matrix[i, eidx] = coeff
    
    best_result = None
    best_n = float('inf')
    
    def _try_find_voltages(m, constraint_matrix, n_edges, deadline):
        # Heuristic search
        for attempt in range(1000):
            if time.time() > deadline:
                break
            voltages = np.random.randint(0, m, size=n_edges, dtype=np.int32)
            sums = constraint_matrix @ voltages
            violated = np.where((sums % m) == 0)[0]
            
            for step in range(1000):
                if time.time() > deadline:
                    break
                if len(violated) == 0:
                    return voltages
                
                c = violated[np.random.randint(len(violated))]
                terms = constraint_matrix[c, :]
                nonzero_edges = np.where(terms != 0)[0]
                if len(nonzero_edges) == 0:
                    break
                e = nonzero_edges[np.random.randint(len(nonzero_edges))]
                voltages[e] = (voltages[e] + 1) % m
                
                # Update sums for affected constraints
                # This is slow if done naively, but n_constraints is small (~100-1000?)
                # Let's just recompute all sums for simplicity if n_constraints < 10000
                sums = constraint_matrix @ voltages
                violated = np.where((sums % m) == 0)[0]
        return None
    
    # Moore bound for k=5, g=9 is 425.
    # Base has 16 vertices. Need 16*m >= 425 => m >= 27.
    for m in range(27, 500):
        if time.time() > deadline - 10:
            break
        n_vertices = base_n * m
        if n_vertices >= best_n:
            continue
            
        voltages = _try_find_voltages(m, constraint_matrix, n_base_edges, deadline - 5)
        
        if voltages is not None:
            graph_edges = []
            for u, v in base_edges:
                eidx = edge_index[(u, v)]
                vol = int(voltages[eidx])
                for i in range(m):
                    vid1 = u * m + i
                    # u < v is always true in base_edges generation
                    j = (i + vol) % m
                    vid2 = v * m + j
                    if vid1 < vid2:
                        graph_edges.append((vid1, vid2))
                    else:
                        graph_edges.append((vid2, vid1))
            
            best_result = (n_vertices, graph_edges)
            best_n = n_vertices
            break
    
    if best_result is not None:
        return best_result
    
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction from the seed program."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
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

# EVOLVE-BLOCK-END
