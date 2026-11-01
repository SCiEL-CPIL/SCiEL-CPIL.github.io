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
    
    # Precompute edge-to-constraint mapping for efficient updates
    edge_to_constraints = [[] for _ in range(n_base_edges)]
    for c_idx in range(n_constraints):
        for e_idx in range(n_base_edges):
            if constraint_matrix[c_idx, e_idx] != 0:
                edge_to_constraints[e_idx].append(c_idx)
    
    best_result = None
    best_n = float('inf')
    
    def _try_find_voltages(m, constraint_matrix, n_edges, deadline):
        """Improved heuristic search with multi-step updates and better initialization."""
        n_cons = constraint_matrix.shape[0]
        
        for attempt in range(2000):
            if time.time() > deadline:
                break
            
            # Better initial voltage: try to minimize violations from the start
            # Use a greedy approach: assign voltages one by one to minimize expected violations
            voltages = np.zeros(n_edges, dtype=np.int32)
            for e in range(n_edges):
                best_vol = 0
                best_score = float('inf')
                # Try a few candidate values
                candidates = [0, 1, m // 2, m // 3, 2 * m // 3]
                for cand in candidates:
                    voltages[e] = cand
                    # Quick score: how many constraints are now satisfied?
                    partial_sums = constraint_matrix[:, e] * cand
                    # For already-assigned edges, compute full sum
                    assigned_mask = np.zeros(n_edges, dtype=bool)
                    assigned_mask[:e+1] = True
                    full_sums = constraint_matrix[:, :e+1] @ voltages[:e+1]
                    satisfied = np.sum((full_sums % m) != 0)
                    score = -satisfied  # minimize negative satisfaction = maximize satisfaction
                    if score < best_score:
                        best_score = score
                        best_vol = cand
                voltages[e] = best_vol
            
            sums = constraint_matrix @ voltages
            violated = np.where((sums % m) == 0)[0]
            
            # Local search with multi-step updates
            for step in range(5000):
                if time.time() > deadline:
                    break
                if len(violated) == 0:
                    return voltages
                
                # Pick a violated constraint
                c = violated[np.random.randint(len(violated))]
                terms = constraint_matrix[c, :]
                nonzero_edges = np.where(terms != 0)[0]
                if len(nonzero_edges) == 0:
                    break
                
                # Try multiple delta values for the chosen edge
                best_e = None
                best_delta = None
                best_new_violated = None
                best_violation_count = len(violated)
                
                # Try a few edges and deltas
                num_tries = min(10, len(nonzero_edges))
                try_edges = nonzero_edges[np.random.choice(len(nonzero_edges), size=num_tries, replace=False)]
                for e in try_edges:
                    coeff = terms[e]
                    current_vol = voltages[e]
                    # Try several delta values
                    for delta in [1, -1, 2, -2, m // 2, m // 3]:
                        new_vol = (current_vol + delta) % m
                        if new_vol == current_vol:
                            continue
                        
                        # Compute new sums for affected constraints
                        new_voltages = voltages.copy()
                        new_voltages[e] = new_vol
                        new_sums = sums + terms[e] * delta
                        new_sums = new_sums % m
                        new_violated_count = np.sum(new_sums == 0)
                        
                        if new_violated_count < best_violation_count:
                            best_violation_count = new_violated_count
                            best_e = e
                            best_delta = delta
                            best_new_violated = np.where(new_sums == 0)[0]
                
                if best_e is not None:
                    voltages[best_e] = (voltages[best_e] + best_delta) % m
                    sums = constraint_matrix @ voltages
                    violated = np.where((sums % m) == 0)[0]
                else:
                    # No improvement found; try a random move to escape local minimum
                    e = nonzero_edges[np.random.randint(len(nonzero_edges))]
                    voltages[e] = (voltages[e] + 1) % m
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
