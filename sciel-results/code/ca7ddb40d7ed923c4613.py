import random
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 115  # Safety margin
    
    if k != 5 or g != 9:
        return _luw_construct(k, g)
    
    # Base graph: K_6
    base_n = 6
    base_edges = []
    edge_index = {}
    for u in range(base_n):
        for v in range(u + 1, base_n):
            edge_index[(u, v)] = len(base_edges)
            base_edges.append((u, v))
    n_base_edges = len(base_edges)  # 15 edges
    
    # Generate non-backtracking closed walks of length 3..8 in K_6 starting at 0
    def gen_walks(length):
        def walk_gen(pos, cur, prv, path):
            if pos == length - 1:
                if cur != 0 and prv != 0:
                    yield list(path) + [0]
                return
            for nv in range(base_n):
                if nv != cur and nv != prv:
                    path.append(nv)
                    yield from walk_gen(pos + 1, nv, cur, path)
                    path.pop()
        for v1 in range(1, base_n):
            yield from walk_gen(1, v1, 0, [0, v1])
    
    # Build constraints
    constraints = set()
    for L in range(3, 9):
        for walk in gen_walks(L):
            coeff_dict = {}
            for i in range(L):
                u, v = walk[i], walk[i + 1]
                eu, ev = min(u, v), max(u, v)
                eidx = edge_index[(eu, ev)]
                sign = 1 if u < v else -1
                coeff_dict[eidx] = coeff_dict.get(eidx, 0) + sign
            terms = tuple(sorted((e, c) for e, c in coeff_dict.items() if c != 0))
            if terms:
                constraints.add(terms)
    
    constraints_list = list(constraints)
    n_constraints = len(constraints_list)
    
    # Build matrix
    A = np.zeros((n_constraints, n_base_edges), dtype=np.int32)
    for i, terms in enumerate(constraints_list):
        for eidx, coeff in terms:
            A[i, eidx] = coeff
    
    best_result = None
    best_n = float('inf')
    
    # Try m from small to large
    # We know m=186 works (1116 vertices). We try to find smaller m.
    # But random search might not find small m.
    # Let's try a range of m values.
    
    for m in range(2, 200):
        if time.time() > deadline - 20:
            break
            
        n_vertices = base_n * m
        if n_vertices >= best_n:
            continue
            
        # Local Search / Random Restart
        # We will try several random restarts with local improvement
        found = False
        max_restarts = 50
        steps_per_restart = 2000
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            v = np.random.randint(0, m, size=n_base_edges, dtype=np.int32)
            
            # Check initial
            sums = A @ v
            violations = (sums % m) == 0
            if not np.any(violations):
                found = True
                break
                
            # Local search
            for step in range(steps_per_restart):
                if time.time() > deadline - 5:
                    break
                    
                # Find violated constraints
                viol_idx = np.where(violations)[0]
                if len(viol_idx) == 0:
                    found = True
                    break
                    
                # Pick a random violated constraint
                c_idx = viol_idx[random.randint(0, len(viol_idx) - 1)]
                c_row = A[c_idx]
                
                # Pick a random edge in this constraint to change
                nonzero_edges = np.where(c_row != 0)[0]
                if len(nonzero_edges) == 0:
                    continue
                e_idx = nonzero_edges[random.randint(0, len(nonzero_edges) - 1)]
                
                # Try changing v[e_idx] to a new random value
                old_val = v[e_idx]
                new_val = np.random.randint(0, m)
                if new_val == old_val:
                    continue
                
                v[e_idx] = new_val
                
                # Update sums efficiently?
                # A @ v is O(N*M). M=15. N=80k.
                # This is fast enough in NumPy.
                sums = A @ v
                violations = (sums % m) == 0
                
                # If this change increased violations significantly, revert?
                # Simple hill climbing: accept if violations count doesn't increase too much?
                # Or just accept and hope?
                # Let's accept if it reduces the number of violations.
                # But we don't track the count easily without extra work.
                # Let's just accept if the specific constraint is fixed.
                # Actually, let's just do random walk with acceptance if total violations decreases.
                
        if found:
            # Build graph
            graph_edges = []
            for u in range(base_n):
                for v_base in range(base_n):
                    if u == v_base:
                        continue
                    eu, ev = min(u, v_base), max(u, v_base)
                    eidx = edge_index[(eu, ev)]
                    vol = int(v[eidx])
                    for i in range(m):
                        vid1 = u * m + i
                        if u < v_base:
                            j = (i + vol) % m
                        else:
                            j = (i - vol) % m
                        vid2 = v_base * m + j
                        if vid1 < vid2:
                            graph_edges.append((vid1, vid2))
            best_result = (n_vertices, graph_edges)
            best_n = n_vertices
            break # Found for this m, try smaller m? No, m is increasing.
            # Wait, we want MINIMUM n. So we should try small m first.
            # If we find a solution for m=10, n=60, that's great.
            # If we find for m=186, n=1116, that's good.
            # Since we iterate m from 2 upwards, the first solution found is the smallest m (and thus smallest n) for this strategy.
            # So we can break the outer loop once we find ANY solution?
            # Yes, because we are iterating m in increasing order.
            break
    
    if best_result is not None:
        return best_result
        
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
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
