"""Voltage-graph lift construction for small 5-regular graph of girth 9."""
import random
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    # We focus on the task's specific parameters: k=5, g=9.
    # Use a voltage graph (lift) over a small base graph with abelian voltage
    # assignments from a cyclic group Z_m.
    #
    # Base graph: K_6 (complete graph on 6 vertices) is 5-regular.
    # A voltage graph over K_6 with voltages in Z_m gives a 5-regular graph
    # on 6*m vertices.
    #
    # For girth >= 9, we need no non-backtracking closed walk in the lifted
    # graph of length < 9. This is equivalent to: for every non-backtracking
    # closed walk W in the base graph of length L < 9, the sum of voltages along W
    # (with appropriate signs for edge direction) is not 0 mod m.
    
    deadline = time.time() + 115  # 115 seconds max (safety margin)
    
    # For the specific case k=5, g=9:
    if k != 5 or g != 9:
        # Fallback to L-U-W construction for other parameters
        return _luw_construct(k, g)
    
    # Try voltage graph over K_6
    base_n = 6
    base_edges = []
    edge_index = {}
    for u in range(base_n):
        for v in range(u + 1, base_n):
            edge_index[(u, v)] = len(base_edges)
            base_edges.append((u, v))
    n_base_edges = len(base_edges)  # 15 edges
    
    # Enumerate non-backtracking closed walks of length 3..8 in K_6
    # A walk is a sequence of vertices v_0, ..., v_L where v_L = v_0,
    # consecutive vertices differ, and v_{i+1} != v_{i-1} (no backtracking).
    # We fix v_0 = 0 to avoid counting rotations.
    
    def gen_nonbacktracking_closed_walks(length):
        """Generate non-backtracking closed walks of given length in K_6, starting at 0."""
        if length < 3:
            return
        
        def walk_gen(pos, cur, prv, path):
            if pos == length - 1:
                # Next must be 0
                # Conditions: cur != 0 (edge cur->0 exists), prv != 0 (no backtrack: 0 != prv)
                if cur != 0 and prv != 0:
                    yield list(path) + [0]
                return
            
            for nv in range(base_n):
                if nv == cur:
                    continue
                if nv == prv:
                    continue
                path.append(nv)
                yield from walk_gen(pos + 1, nv, cur, path)
                path.pop()
        
        for v1 in range(1, base_n):
            yield from walk_gen(1, v1, 0, [0, v1])
    
    # Collect all constraints: for each walk, we have a linear form in voltages
    # that must be non-zero mod m.
    # Represent each constraint as a tuple of (edge_idx, net_coefficient)
    
    constraints = []  # list of tuples (eidx, coeff)
    for L in range(3, 9):
        walks = list(gen_nonbacktracking_closed_walks(L))
        for walk in walks:
            # walk is [v_0, v_1, ..., v_L] with v_0 = v_L = 0
            # voltage sum = sum_{i=0}^{L-1} s_i * phi_{min(v_i,v_{i+1}), max(v_i,v_{i+1})}
            # where s_i = +1 if v_i < v_{i+1}, else -1
            coeff_dict = {}
            for i in range(L):
                u, v = walk[i], walk[i + 1]
                if u == v:
                    continue
                eu, ev = min(u, v), max(u, v)
                eidx = edge_index[(eu, ev)]
                sign = 1 if u < v else -1
                coeff_dict[eidx] = coeff_dict.get(eidx, 0) + sign
            # Convert to tuple, only non-zero coefficients
            terms = tuple(sorted((eidx, c) for eidx, c in coeff_dict.items() if c != 0))
            if terms:  # non-empty constraint
                constraints.append(terms)
    
    # Deduplicate constraints
    constraints = list(set(constraints))
    n_constraints = len(constraints)
    
    # Build constraint matrix for fast evaluation
    # Each row is a constraint, columns are edge indices
    # constraint_matrix[i][j] = coefficient of edge j in constraint i
    constraint_matrix = np.zeros((n_constraints, n_base_edges), dtype=np.int32)
    for i, terms in enumerate(constraints):
        for eidx, coeff in terms:
            constraint_matrix[i, eidx] = coeff
    
    best_result = None
    best_n = float('inf')
    
    # Try different m values, starting from small
    for m in range(2, 500):
        if time.time() > deadline - 20:
            break
        
        n_vertices = base_n * m
        if n_vertices >= best_n:
            continue  # We want to minimize n, so skip if not smaller
        
        # Try random voltage assignments
        max_attempts = 20000
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            voltages = np.random.randint(0, m, size=n_base_edges, dtype=np.int32)
            
            # Evaluate all constraints: check if any evaluates to 0 mod m
            # constraint_matrix @ voltages gives the sum for each constraint
            sums = constraint_matrix @ voltages
            # Check if any sum is 0 mod m
            if not np.any((sums % m) == 0):
                found = True
                break
        
        if found:
            # Build the lifted graph
            graph_edges = []
            for u in range(base_n):
                for v in range(base_n):
                    if u == v:
                        continue
                    eu, ev = min(u, v), max(u, v)
                    eidx = edge_index[(eu, ev)]
                    vol = int(voltages[eidx])
                    for i in range(m):
                        vid1 = u * m + i
                        if u < v:
                            j = (i + vol) % m
                        else:
                            j = (i - vol) % m
                        vid2 = v * m + j
                        if vid1 < vid2:
                            graph_edges.append((vid1, vid2))
            
            best_result = (n_vertices, graph_edges)
            best_n = n_vertices
            break  # Found a valid one for this m, go to next m (smaller n)
    
    if best_result is not None:
        return best_result
    
    # Fallback: use L-U-W construction
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
