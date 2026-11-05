import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Strategy:
      1. Try deterministic algebraic constructions (Cayley graphs of small groups)
      2. Fallback to voltage lift of Heawood/Petersen
      3. Final fallback: LUW construction
    """
    start_time = time.time()
    deadline = start_time + 135.0

    if k == 3:
        # Try algebraic constructions first (deterministic, fast)
        result = _construct_algebraic(k, g, deadline, start_time)
        if result is not None:
            return result
        
        # Try Heawood voltage lift
        n_heawood, edges_heawood = _construct_lift_base(_get_heawood_edges(), g, deadline, start_time)
        if n_heawood is not None:
            return n_heawood, edges_heawood
        
        # Fallback to Petersen voltage lift
        n_petersen, edges_petersen = _construct_lift_base(_get_petersen_edges(), g, deadline, start_time)
        if n_petersen is not None:
            return n_petersen, edges_petersen

        # Final fallback
        return _construct_luw(k, g, d=11)
    else:
        return _construct_luw(k, g, d=7)


def _construct_algebraic(k, g, deadline, start_time):
    """
    Try to construct a k-regular graph with girth >= g using Cayley graphs
    of small finite groups with carefully chosen generating sets.
    """
    if k != 3:
        return None
    
    # PSL(2,11) has order 660. We need a symmetric generating set of size 3.
    # PSL(2,11) can be represented as 3x3 matrices over GF(11) with det=1.
    # We'll enumerate candidate generating sets and check girth via BFS on words.
    
    # Represent PSL(2,11) elements as (a,b,c,d) representing matrix [[a,b],[c,d]] mod 11
    # with ad - bc = 1.
    
    def mat_mul(m1, m2):
        a1,b1,c1,d1 = m1
        a2,b2,c2,d2 = m2
        return ((a1*a2 + b1*c2) % 11, (a1*b2 + b1*d2) % 11,
                (c1*a2 + d1*c2) % 11, (c1*b2 + d1*d2) % 11)
    
    def mat_inv(m):
        a,b,c,d = m
        det = (a*d - b*c) % 11
        # Inverse is [[d,-b],[-c,a]] / det
        inv_det = pow(det, 10, 11)  # det^(11-2) mod 11 = det^-1 mod 11
        return (d*inv_det % 11, -b*inv_det % 11, -c*inv_det % 11, a*inv_det % 11)
    
    identity = (1, 0, 0, 1)
    
    # Generate all elements of PSL(2,11) by BFS from identity using standard generators
    # Standard presentation: PSL(2,11) = <S, T | S^2 = T^3 = (ST)^11 = 1>
    # S = [[0,-1],[1,0]], T = [[0,-1],[1,-1]]
    S = (0, 10, 1, 0)  # [[0,-1],[1,0]] mod 11 = [[0,10],[1,0]]
    T = (0, 10, 1, 10)  # [[0,-1],[1,-1]] mod 11 = [[0,10],[1,10]]
    
    # BFS to generate all 660 elements
    elements = [identity]
    elem_set = {identity: 0}
    queue = deque([identity])
    
    while queue:
        if time.time() > deadline:
            return None
        e = queue.popleft()
        for gen in [S, T, mat_inv(S), mat_inv(T)]:
            ne = mat_mul(e, gen)
            if ne not in elem_set:
                elem_set[ne] = len(elements)
                elements.append(ne)
                queue.append(ne)
    
    n_group = len(elements)
    # PSL(2,11) should have 660 elements
    if n_group != 660:
        return None
    
    # Now try various symmetric generating sets of size 3
    # A symmetric generating set {A, B, C} means we use edges for multiplication by A, A^-1, B, B^-1, C, C^-1
    # But for a 3-regular graph, we need exactly 3 neighbors per vertex.
    # So the generating set should be {A, B, C} where A=A^-1, B=B^-1, C=C^-1 (involutions)
    # OR {A, B} where one is an involution and the other pairs with its inverse... 
    # Actually for 3-regular: each vertex connects to g1*v, g2*v, g3*v for 3 generators.
    # The graph is undirected iff each generator is self-inverse (involution) or we pair them.
    # For simplicity, use 3 involutions.
    
    # Find involutions in PSL(2,11)
    involutions = []
    for i, e in enumerate(elements):
        if mat_mul(e, e) == identity and e != identity:
            involutions.append(i)
    
    if time.time() > deadline:
        return None
    
    # Try random triples of involutions as generating sets
    random.seed(42)
    best_result = None
    best_n = float('inf')
    
    max_tries = 200
    tries = 0
    while tries < max_tries and time.time() < deadline - 5:
        tries += 1
        if len(involutions) < 3:
            break
        
        # Pick 3 random distinct involutions
        indices = random.sample(involutions, 3)
        gens = [elements[indices[0]], elements[indices[1]], elements[indices[2]]]
        
        # Check if these generate the whole group
        generated = {identity}
        gen_queue = deque([identity])
        while gen_queue:
            e = gen_queue.popleft()
            for g in gens:
                ne = mat_mul(e, g)
                if ne not in generated:
                    generated.add(ne)
                    gen_queue.append(ne)
        
        if len(generated) != n_group:
            continue
        
        # Build the Cayley graph
        # Vertex i corresponds to elements[i]
        # Edge from i to i*g_j for each generator j
        adj = [set() for _ in range(n_group)]
        for i, e in enumerate(elements):
            for g in gens:
                ne = mat_mul(e, g)
                j = elem_set[ne]
                adj[i].add(j)
        
        # Verify 3-regular
        if any(len(adj[i]) != 3 for i in range(n_group)):
            continue
        
        # Check girth using BFS (non-backtracking)
        girth_ok = True
        for start in range(n_group):
            # BFS with non-backtracking constraint
            # State: (current, prev, length)
            visited = {}  # (node, prev) -> min length
            bfs_queue = deque()
            for nb in adj[start]:
                bfs_queue.append((nb, start, 1))
            
            while bfs_queue:
                node, prev, length = bfs_queue.popleft()
                
                if node == start and length >= 1:
                    # Found a cycle of length 'length'
                    if length < g:
                        girth_ok = False
                        break
                    continue
                
                if length >= g - 1:
                    continue
                
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    key = (nb, node)
                    if key not in visited or visited[key] > length + 1:
                        visited[key] = length + 1
                        bfs_queue.append((nb, node, length + 1))
            
            if not girth_ok:
                break
        
        if girth_ok:
            # Build edge list
            edges = []
            for i in range(n_group):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            
            if n_group < best_n:
                best_n = n_group
                best_result = (n_group, edges)
            
            if time.time() > deadline - 10:
                break
    
    if best_result is not None:
        return best_result
    
    return None

def _get_petersen_edges():
    return [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

def _get_heawood_edges():
    # Fano plane incidence graph
    # Points 0-6, Lines 7-13
    edges = []
    # L0: 0,1,2 -> 7
    edges += [(0,7), (1,7), (2,7)]
    # L1: 0,3,4 -> 8
    edges += [(0,8), (3,8), (4,8)]
    # L2: 0,5,6 -> 9
    edges += [(0,9), (5,9), (6,9)]
    # L3: 1,3,5 -> 10
    edges += [(1,10), (3,10), (5,10)]
    # L4: 1,4,6 -> 11
    edges += [(1,11), (4,11), (6,11)]
    # L5: 2,3,6 -> 12
    edges += [(2,12), (3,12), (6,12)]
    # L6: 2,4,5 -> 13
    edges += [(2,13), (4,13), (5,13)]
    return edges

def _construct_lift_base(base_edges, g, deadline, start_time):
    """
    Constructs a girth-g graph by lifting the base graph with voltages in Z_m.
    """
    n_base = len(set(v for e in base_edges for v in e))
    edge_index = {}
    for i, (u, v) in enumerate(base_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Precompute all non-backtracking closed walks of length < g in the base graph
    coeff_vectors = set()
    max_walk_len = g - 1
    
    for start_v in range(n_base):
        # DFS to find all non-backtracking closed walks of length < g
        stack = [(start_v, -1, 0, [0]*len(base_edges))]
        
        while stack:
            v, prev_v, l, counts = stack.pop()
            
            if l >= max_walk_len:
                continue
            
            for next_v in range(n_base):
                if next_v == prev_v:
                    continue
                if next_v == v:
                    continue
                
                e_idx = edge_index.get((v, next_v))
                if e_idx is None:
                    continue
                
                new_counts = counts.copy()
                u_edge, v_edge = base_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    new_counts[e_idx] += 1
                else:
                    new_counts[e_idx] -= 1
                
                new_l = l + 1
                if next_v == start_v and new_l > 0:
                    coeff_vectors.add(tuple(new_counts))
                
                stack.append((next_v, v, new_l, new_counts))
    
    # Normalize constraints to remove redundancy
    unique_coeffs = set()
    for c in coeff_vectors:
        if all(x == 0 for x in c):
            continue
        for i in range(len(base_edges)):
            if c[i] != 0:
                if c[i] < 0:
                    c = tuple(-x for x in c)
                unique_coeffs.add(c)
                break
    coeff_list = list(unique_coeffs)
    
    if not coeff_list:
        # No constraints, any voltages work
        voltages = [1] * len(base_edges)
        m = 2
        n_candidate = n_base * m
        edges = _build_lifted_graph(base_edges, voltages, m)
        return n_candidate, edges
    
    # Build coefficient matrix for vectorized checking
    coeff_matrix = np.array(coeff_list, dtype=np.int64)
    n_constraints = len(coeff_matrix)
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    primes = []
    candidate = 2
    while len(primes) < 200:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    
    best_n = float('inf')
    best_edges = None
    
    for m in primes:
        if time.time() > deadline - 5:
            break
        
        n_candidate = n_base * m
        if n_candidate >= best_n:
            continue
        
        found = False
        # Scale attempts based on remaining time
        remaining = deadline - time.time()
        max_attempts = min(300, max(50, int(remaining * 20)))
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            voltages = np.random.randint(1, m, size=len(base_edges)).astype(np.int64)
            
            # Local Repair Loop
            max_repairs = min(80, max(20, int((deadline - time.time()) * 10)))
            for repair_step in range(max_repairs):
                if time.time() > deadline - 5:
                    break
                
                # Vectorized constraint checking
                products = coeff_matrix @ voltages  # shape: (n_constraints,)
                residues = np.mod(products, m)
                
                # Find violated constraints (residue == 0)
                violated_indices = np.where(residues == 0)[0]
                
                if len(violated_indices) == 0:
                    # All constraints satisfied
                    voltages_list = voltages.tolist()
                    edges = _build_lifted_graph(base_edges, voltages_list, m)
                    return n_candidate, edges
                
                # Pick a random violated constraint
                violated_idx = violated_indices[random.randint(0, len(violated_indices)-1)]
                violated_c = coeff_matrix[violated_idx]
                
                # Pick a random edge with non-zero coefficient in this constraint
                nonzero_indices = np.where(violated_c != 0)[0]
                if len(nonzero_indices) == 0:
                    break
                j = nonzero_indices[random.randint(0, len(nonzero_indices)-1)]
                
                # Change voltage j to a random new value
                voltages[j] = random.randint(1, m-1)
        
        # If we haven't found a solution for this m, try next m
    
    return None, None

def _build_lifted_graph(base_edges, voltages, m):
    n_base = len(set(v for e in base_edges for v in e))
    edge_index = {}
    for i, (u, v) in enumerate(base_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    edges = []
    for u in range(n_base):
        for v in range(n_base):
            if u == v:
                continue
            e_idx = edge_index.get((u, v))
            if e_idx is None:
                continue
            
            vol = voltages[e_idx]
            u_edge, v_edge = base_edges[e_idx]
            if u == u_edge and v == v_edge:
                sign = 1
            else:
                sign = -1
            
            for layer in range(m):
                u_node = layer * n_base + u
                target_layer = (layer + sign * vol) % m
                v_node = target_layer * n_base + v
                
                if u_node < v_node:
                    edges.append((u_node, v_node))
                else:
                    edges.append((v_node, u_node))
                
    edges = list(set(edges))
    return edges

def _construct_luw(k, g, d=None, variant=0):
    """
    LUW construction with specified depth d and variant for rule selection.
    For k=3, uses q=3 arithmetic.
    """
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    q = k
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
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
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
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _has_girth_at_least(n, edges, g):
    """
    Check if graph has girth at least g using non-backtracking walks.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_len = g - 1
    
    for start in range(n):
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
        
        while queue:
            u, prev, l = queue.popleft()
            
            if u == start and l > 0:
                return False
            
            if l >= max_len:
                continue
            
            for v in adj[u]:
                if v != prev:
                    queue.append((v, u, l + 1))
    
    return True
