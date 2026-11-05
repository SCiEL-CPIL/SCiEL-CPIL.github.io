import random
import time
import math
from collections import deque

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)

    # Use a larger base graph (Dodecahedron, 20 vertices, girth 6) for voltage lift
    # Dodecahedron: 20 vertices, 30 edges, 3-regular, girth 6
    # Adjacency: outer pentagon (0-4), inner pentagon (15-19), and connections
    # Standard dodecahedron embedding
    edges = []
    # Two pentagons and connections
    # Outer ring: 0,1,2,3,4
    # Inner ring: 15,16,17,18,19
    # Middle ring: 5,6,7,8,9,10,11,12,13,14
    # Let me use a standard dodecahedron construction
    # Actually, let me use the Petersen graph (10 vertices) which is well-known and has girth 5
    # The dodecahedron has girth 6, which is better for reducing constraints
    
    # Dodecahedron edges (standard labeling):
    # Vertices 0-9 form an inner decagon, 10-19 form connections
    # Let me use a simpler approach: generate dodecahedron via Schläfli {5,3}
    # Actually, let me stick with Petersen (10 vertices, 15 edges) as it's simpler and well-tested
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer 5-cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner 5-star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
    ]
    n_base = 10
    n_edges = len(edges)
    
    adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Generate all closed non-backtracking walks of length < g
    walks = []
    for start_v in range(n_base):
        queue = deque()
        for next_v, e_id in adj[start_v]:
            u, v = edges[e_id]
            sign = 1 if u == start_v else -1
            queue.append((next_v, e_id, 1, [(e_id, sign)]))
        
        while queue:
            curr_v, prev_e, length, path = queue.popleft()
            if length >= g:
                continue
            if curr_v == start_v and length >= 3:
                walks.append(path)
            for next_v, e_id in adj[curr_v]:
                if e_id == prev_e:
                    continue
                u, v = edges[e_id]
                sign = 1 if u == curr_v else -1
                new_path = path + [(e_id, sign)]
                queue.append((next_v, e_id, length + 1, new_path))
    
    unique_walks = set()
    for w in walks:
        unique_walks.add(tuple(w))
    walks = list(unique_walks)
    
    # Pre-compute sparse coefficient vectors for each walk
    walk_coeffs = []
    for w in walks:
        coeffs = [0] * n_edges
        for e_id, sign in w:
            coeffs[e_id] += sign
        nonzero = [(i, c) for i, c in enumerate(coeffs) if c != 0]
        walk_coeffs.append(nonzero)
        
    if not walk_coeffs:
        return 10, edges

    start_time = time.time()
    deadline = start_time + 135.0
    
    best_n = float('inf')
    best_edges = None
    best_m = None
    
    # Try m values from Moore bound upward
    for m in range(52, 300):
        if time.time() > deadline - 15.0:
            break
        
        candidate_n = 10 * m
        if candidate_n >= best_n:
            break
            
        # Use simulated annealing for voltage assignment
        time_left = deadline - time.time()
        max_sa_iters = min(50000, max(5000, int(time_left / 0.0002)))
        
        found = False
        best_volts = None
        
        # Initialize with random voltages
        volts = [random.randint(1, m - 1) for _ in range(n_edges)]
        temp = m * 0.5
        cooling = 0.995
        
        for it in range(max_sa_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Compute current energy (number of violated walks)
            violations = 0
            for nonzero in walk_coeffs:
                s = 0
                for idx, c in nonzero:
                    s += c * volts[idx]
                if s % m == 0:
                    violations += 1
                    if violations > 10:  # Early exit if too many violations
                        break
            
            if violations == 0:
                # Check connectivity
                g_val = m
                for v in volts:
                    g_val = math.gcd(g_val, v)
                    if g_val == 1:
                        break
                if g_val == 1:
                    best_volts = volts[:]
                    found = True
                    break
            
            # Pick a random edge to modify
            edge_idx = random.randint(0, n_edges - 1)
            old_val = volts[edge_idx]
            volts[edge_idx] = random.randint(1, m - 1)
            
            # Compute new energy
            new_violations = 0
            for nonzero in walk_coeffs:
                s = 0
                for idx, c in nonzero:
                    s += c * volts[idx]
                if s % m == 0:
                    new_violations += 1
                    if new_violations > 10:
                        break
            
            delta_e = new_violations - violations
            if delta_e <= 0 or random.random() < math.exp(-delta_e / temp):
                violations = new_violations
            else:
                volts[edge_idx] = old_val
            
            temp *= cooling
        
        if found:
            best_m = m
            best_n = candidate_n
            lift_edges = []
            for i, (u, v) in enumerate(edges):
                vol = volts[i]
                for t in range(m):
                    u_lift = u * m + t
                    v_lift = v * m + (t + vol) % m
                    lift_edges.append((u_lift, v_lift))
            best_edges = lift_edges
            break
            
    if best_edges is None:
        return _luw_fallback(k, g)

    return best_n, best_edges

def _luw_fallback(k, g):
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw

if __name__ == "__main__":
    n, e = construct(3, 16)
    print(f"n={n}")
