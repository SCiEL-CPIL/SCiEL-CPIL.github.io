import time
import random
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses voltage lift of Petersen graph for k=3.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    if k == 3:
        return _construct_petersen_lift(g, deadline, start_time)
    else:
        # Fallback for other k, though task specifies k=3
        return _construct_luw(k, g, d=7)

def _construct_petersen_lift(g, deadline, start_time):
    """
    Constructs a girth-g graph by lifting the Petersen graph with voltages in Z_m.
    Uses Simulated Annealing to find valid voltages.
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
    
    for start_v in range(n_base):
        # DFS to find all non-backtracking closed walks of length < g
        # State: (current_v, prev_v, length, counts)
        stack = [(start_v, -1, 0, [0]*15)]
        
        while stack:
            v, prev_v, l, counts = stack.pop()
            
            if l >= max_walk_len:
                continue
            
            for next_v in range(n_base):
                if next_v == prev_v:
                    continue # Non-backtracking
                if next_v == v:
                    continue # No loops
                
                e_idx = edge_index.get((v, next_v))
                if e_idx is None:
                    continue
                
                new_counts = counts.copy()
                u_edge, v_edge = pet_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    new_counts[e_idx] += 1
                else:
                    new_counts[e_idx] -= 1
                
                new_l = l + 1
                if next_v == start_v and new_l > 0:
                    # Found a non-backtracking closed walk
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
    # C is a numpy array of shape (num_constraints, 15)
    if not coeff_list:
        # If no constraints, any voltage works, but we need m >= 2
        m = 2
        voltages = [1] * 15
        n_candidate = 10 * m
        edges = _build_lifted_graph(pet_edges, voltages, m)
        return n_candidate, edges

    C = np.array(coeff_list, dtype=np.int64)
    num_constraints = len(C)
    
    # Search for m and voltages
    best_n = float('inf')
    best_edges = None
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = []
    candidate = 2
    while len(primes) < 200:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    
    for m in primes:
        if time.time() > deadline:
            break
        
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
        
        # Simulated Annealing
        # Energy: number of constraints violated (sum % m == 0)
        # We want to minimize energy to 0.
        
        def calculate_energy(voltages_np):
            # voltages_np is shape (15,)
            # C is shape (num_constraints, 15)
            # sums = C @ voltages_np
            sums = C @ voltages_np
            # Check which sums are 0 mod m
            # Note: numpy modulo can be negative? No, Python's % is always non-negative for positive modulus.
            # Numpy's % follows C standard, which for positive modulus is non-negative.
            violated = (sums % m == 0)
            return np.sum(violated)
        
        found = False
        # Parameters for SA
        T_start = 10.0
        T_end = 0.01
        alpha = 0.99
        max_iters = 5000
        restarts = 5
        
        for restart in range(restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            voltages = [random.randint(1, m-1) for _ in range(15)]
            v_np = np.array(voltages, dtype=np.int64)
            current_energy = calculate_energy(v_np)
            
            if current_energy == 0:
                found = True
                break
                
            T = T_start
            
            for i in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Pick a random edge to change
                j = random.randint(0, 14)
                old_val = v_np[j]
                new_val = random.randint(1, m-1)
                if new_val == old_val:
                    continue
                    
                v_np[j] = new_val
                new_energy = calculate_energy(v_np)
                
                delta_e = new_energy - current_energy
                
                if delta_e < 0:
                    # Accept
                    current_energy = new_energy
                else:
                    # Accept with probability exp(-delta_e / T)
                    if random.random() < np.exp(-delta_e / T):
                        current_energy = new_energy
                    else:
                        # Reject
                        v_np[j] = old_val
                        
                T *= alpha
                
                if current_energy == 0:
                    found = True
                    break
                if T < T_end:
                    break
            
            if found:
                break
        
        if found:
            # Final verification
            voltages_final = v_np.tolist()
            edges = _build_lifted_graph(pet_edges, voltages_final, m)
            if _has_girth_at_least(n_candidate, edges, g):
                return n_candidate, edges
            # If verification fails, it's a bug, but we assume SA found a valid energy=0 state
            # which implies girth >= g theoretically.
            # Let's just return it if energy is 0, trusting the math.
            # But to be safe, we already checked _has_girth_at_least.
            # If it fails, we continue searching.
            
    # Fallback
    return _construct_luw(k=3, g=g, d=11)

def _build_lifted_graph(pet_edges, voltages, m):
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
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
            u_edge, v_edge = pet_edges[e_idx]
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

def _has_girth_at_least(n, edges, g):
    """
    Check if graph has girth at least g using non-backtracking walks.
    A graph has girth >= g iff there are no non-backtracking closed walks of length < g.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Non-backtracking closed walks of length < g
    max_len = g - 1
    
    for start in range(n):
        # State: (current_vertex, prev_vertex, length)
        # Start with all neighbors of start as length-1 walks
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
        
        while queue:
            u, prev, l = queue.popleft()
            
            if u == start and l > 0:
                # Found a non-backtracking closed walk of length l < g
                return False
            
            if l >= max_len:
                continue
            
            for v in adj[u]:
                if v != prev:
                    queue.append((v, u, l + 1))
    
    return True
