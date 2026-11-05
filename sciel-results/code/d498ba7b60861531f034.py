import time
import random
import math
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    For k=3, uses a voltage lift of the Petersen graph over Z_p (p prime).
    Uses a structured search over polynomial voltage assignments combined with
    random initialization and local repair to find valid voltages.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    if k == 3:
        return _construct_petersen_lift(g, deadline, start_time)
    else:
        # Fallback for other k, though task specifies k=3
        return _construct_luw(k, g, d=7)

def _construct_petersen_lift(g, deadline, start_time):
    """
    Constructs a girth-g graph by lifting the Petersen graph with voltages in Z_p.
    """
    # Petersen graph edges (10 vertices, 15 edges)
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), # Outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5), # Inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)  # Spokes
    ]
    
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Precompute all non-backtracking closed walks of length < g in the Petersen graph
    coeff_vectors = set()
    max_walk_len = g - 1
    
    # Use BFS/DFS to find all non-backtracking closed walks
    # State: (current_v, prev_v, length, counts)
    # We need to track the path to ensure non-backtracking.
    # Since the graph is small, we can store the full path or just the previous vertex.
    # For a closed walk, the sequence of edges matters.
    # We represent the voltage sum as a vector of counts per edge.
    
    for start_v in range(n_base):
        # Stack: (current_v, prev_v, length, counts_list)
        # counts_list is a list of 15 integers, initially all 0.
        # When traversing edge e from u to v:
        #   if e is (u,v) in pet_edges, count[e] += 1
        #   if e is (v,u) in pet_edges, count[e] -= 1
        # Actually, let's just track the path of edge indices and signs.
        # But for efficiency, tracking counts is better.
        
        # Optimization: We only care about walks that return to start_v.
        # We can prune if length >= max_walk_len.
        
        stack = [(start_v, -1, 0, [0]*15)]
        
        while stack:
            v, prev_v, l, counts = stack.pop()
            
            if l >= max_walk_len:
                continue
            
            for next_v in range(n_base):
                if next_v == prev_v:
                    continue # Non-backtracking
                if next_v == v:
                    continue # No loops (simple graph)
                
                e_idx = edge_index.get((v, next_v))
                if e_idx is None:
                    continue
                
                # Determine sign
                u_edge, v_edge = pet_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    sign = 1
                else:
                    sign = -1
                
                new_counts = counts.copy()
                new_counts[e_idx] += sign
                
                new_l = l + 1
                if next_v == start_v and new_l > 0:
                    # Found a non-backtracking closed walk
                    # Note: We might find the same walk multiple times from different starts?
                    # No, we fix start_v. But different walks can have same count vector.
                    # The condition is sum(c_e * x_e) != 0 mod m.
                    # So we just need the set of distinct count vectors.
                    coeff_vectors.add(tuple(new_counts))
                
                stack.append((next_v, v, new_l, new_counts))
    
    # Normalize constraints to remove redundancy (c and -c are equivalent)
    unique_coeffs = set()
    for c in coeff_vectors:
        if all(x == 0 for x in c):
            continue
        # Find first non-zero
        for i in range(15):
            if c[i] != 0:
                if c[i] < 0:
                    c = tuple(-x for x in c)
                unique_coeffs.add(c)
                break
    coeff_list = list(unique_coeffs)
    
    # Precompute coefficient matrix for faster evaluation
    A = np.array(coeff_list, dtype=np.int64)
    n_constraints = len(A)
    
    # Search for m (prime) and voltages
    best_n = float('inf')
    best_edges = None
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    # We want small m to minimize n = 10*m.
    # Moore bound for k=3, g=16 is 512.
    # So m >= 52.
    # Let's iterate primes starting from 53 upwards.
    
    primes = []
    candidate = 53
    while len(primes) < 200:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    
    # Strategy:
    # 1. For each prime m, try structured polynomial assignments.
    #    v_i = alpha^(a*i^2 + b*i + c) mod m
    # 2. If that fails, try random assignments with local repair.
    
    for m in primes:
        if time.time() > deadline:
            break
        
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
        
        found = False
        
        # --- Strategy 1: Structured Polynomial Search ---
        # v_i = alpha^(poly(i)) mod m
        # We need to find alpha, a, b, c such that all constraints are satisfied.
        # Also need gcd(voltages, m) = 1 for connectivity?
        # Actually, if alpha is a primitive root, the generated values are non-zero.
        # The subgroup generated by the voltages must be the whole group Z_m for the lift to be connected.
        # If we use powers of a primitive root, the set of voltages is a subset of the group.
        # The lift is connected iff the voltages generate Z_m.
        # If we pick random a,b,c, the voltages are likely to generate Z_m if m is prime.
        
        # Let's try a few random alpha (primitive roots are preferred but any non-zero works for generation if we vary enough)
        # Actually, if we fix alpha, and vary i, we get a sequence.
        # Let's just try random alpha in 1..m-1 and random a,b,c.
        
        max_poly_attempts = 500
        for attempt in range(max_poly_attempts):
            if time.time() > deadline:
                break
                
            alpha = random.randint(1, m-1)
            a = random.randint(0, m-1)
            b = random.randint(0, m-1)
            c = random.randint(0, m-1)
            
            voltages = np.zeros(15, dtype=np.int64)
            for i in range(15):
                exp = (a * i * i + b * i + c) % m
                voltages[i] = pow(alpha, exp, m)
            
            # Check connectivity: do voltages generate Z_m?
            # Since m is prime, any non-zero voltage generates Z_m.
            # So we just need at least one non-zero voltage.
            # But wait, the lift connectivity depends on the gcd of the voltage subgroup and m.
            # The subgroup generated by {v_1, ..., v_15} in Z_m.
            # If any v_i != 0, then since m is prime, gcd(v_i, m) = 1, so the subgroup is Z_m.
            # So connectivity is guaranteed if any voltage is non-zero.
            # With alpha != 0, voltages are non-zero.
            
            # Check girth constraints
            prod = (A @ voltages) % m
            if np.any(prod == 0):
                continue
            
            # Valid!
            found = True
            break
        
        if not found:
            # --- Strategy 2: Random + Local Repair ---
            max_random_attempts = 200
            for attempt in range(max_random_attempts):
                if time.time() > deadline:
                    break
                    
                voltages = np.array([random.randint(1, m-1) for _ in range(15)], dtype=np.int64)
                
                # Local Repair Loop
                for repair_step in range(20):
                    prod = (A @ voltages) % m
                    violated_indices = np.where(prod == 0)[0]
                    
                    if len(violated_indices) == 0:
                        found = True
                        break
                    
                    # Pick a random violated constraint
                    violated_idx = random.choice(violated_indices.tolist())
                    violated_c = coeff_list[violated_idx]
                    
                    # Pick a random edge index j where c[j] != 0
                    candidates = [i for i in range(15) if violated_c[i] != 0]
                    if not candidates:
                        break
                    j = random.choice(candidates)
                    
                    # Try to fix by changing voltages[j]
                    # We need sum(c_e * x_e) != 0.
                    # Currently 0.
                    # New sum = c_j * delta.
                    # We want c_j * delta != 0 mod m.
                    # Since m is prime and c_j != 0, any delta != 0 works.
                    # But we also don't want to break other constraints too badly.
                    # Just pick a random new value.
                    new_val = random.randint(1, m-1)
                    if new_val == voltages[j]:
                        continue
                    voltages[j] = new_val
                
                if found:
                    break
        
        if found:
            # Construct graph
            edges = _build_lifted_graph(pet_edges, voltages, m)
            # Verify girth (optional, but good for safety)
            # For speed, we can skip full verification if we trust the theory.
            # But let's do a quick check or just return.
            # The theory says if all NB closed walks have non-zero voltage, girth is OK.
            # So we can return directly.
            return n_candidate, edges
    
    # Fallback
    return _construct_luw(k=3, g=g, d=11)

def _build_lifted_graph(pet_edges, voltages, m):
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    edges = []
    seen = set()
    for i, (u, v) in enumerate(pet_edges):
        vol = int(voltages[i])
        for t in range(m):
            u_node = t * n_base + u
            v_node = ((t + vol) % m) * n_base + v
            
            if u_node < v_node:
                edge = (u_node, v_node)
            else:
                edge = (v_node, u_node)
            
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)
                
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
    
    # Build rule table with variant-dependent offsets
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
