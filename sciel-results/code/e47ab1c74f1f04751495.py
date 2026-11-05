import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses voltage lift of Petersen graph for k=3.
    """
    start_time = time.time()
    # Leave 5 seconds for final verification and return
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
    adj_base = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        adj_base[u].append(v)
        adj_base[v].append(u)
    
    # Precompute all non-backtracking closed walks of length < g in the Petersen graph
    # We need to ensure no such walk has a voltage sum of 0 mod m.
    # Walks of length < 5 in Petersen are impossible (girth 5).
    # So we check lengths 5 to g-1.
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
            
            for next_v in adj_base[v]:
                if next_v == prev_v:
                    continue # Non-backtracking
                
                e_idx = edge_index[(v, next_v)]
                
                new_counts = counts.copy()
                u_edge, v_edge = pet_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    new_counts[e_idx] += 1
                else:
                    new_counts[e_idx] -= 1
                
                new_l = l + 1
                if next_v == start_v and new_l > 0:
                    # Found a non-backtracking closed walk
                    # Only add if length is >= 5 (though Petersen girth is 5, 
                    # so all closed non-backtracking walks are >= 5)
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
    
    if not coeff_list:
        # Should not happen for Petersen, but safe
        return _construct_luw(3, g, d=7)

    # Precompute coefficient matrix for faster evaluation
    # We want to check sum(c_e * x_e) != 0 mod m
    # This is a linear system of inequalities over Z_m.
    A = np.array(coeff_list, dtype=np.int64)
    n_constraints = len(A)
    
    # Search for m and voltages
    # We need m large enough. 
    # The number of constraints grows with g. 
    # For g=16, we have walks of length 5..15.
    # Let's start with primes that are reasonably large to allow enough "space" for voltages.
    # Smaller m might not have a solution.
    
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

    # Start from 53. 
    # Why 53? 
    # The Petersen graph has 10 vertices. 
    # Lifts of size 10*m. 
    # Known best is 936, so m ~ 93.6. 
    # Primes around 100 are good candidates. 
    # But maybe a smaller one works? 
    # Let's try primes from 53 upwards.
    
    candidate_m = 53
    best_n = float('inf')
    best_edges = None
    
    while time.time() < deadline - 10:
        if is_prime(candidate_m):
            m = candidate_m
            n_candidate = 10 * m
            
            if n_candidate >= best_n:
                candidate_m += 1
                continue
            
            # Try to find a valid voltage assignment for this m
            # Use Simulated Annealing
            
            found = False
            # Limit attempts per m to avoid getting stuck
            max_restarts = 10 
            # If we spend too much time on one m, move to next?
            # Or just try a few restarts.
            
            for restart in range(max_restarts):
                if time.time() > deadline - 10:
                    break
                    
                # Random initialization
                voltages = np.array([random.randint(1, m-1) for _ in range(15)], dtype=np.int64)
                
                # Simulated Annealing
                T = 1.0
                cooling = 0.999
                min_T = 0.01
                max_iters = 5000 # Adjust based on time
                
                def cost(v):
                    residues = np.mod(A @ v, m)
                    return np.sum(residues == 0)
                
                current_cost = cost(voltages)
                best_v = voltages.copy()
                best_cost = current_cost
                
                if best_cost == 0:
                    found = True
                    break
                
                for it in range(max_iters):
                    if time.time() > deadline - 10:
                        break
                    
                    # Pick a random variable to change
                    j = random.randint(0, 14)
                    old_val = voltages[j]
                    voltages[j] = random.randint(1, m - 1)
                    
                    new_cost = cost(voltages)
                    delta = new_cost - current_cost
                    
                    if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                        current_cost = new_cost
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_v = voltages.copy()
                        if new_cost == 0:
                            found = True
                            break
                    else:
                        voltages[j] = old_val
                    
                    T *= cooling
                
                if found:
                    break
            
            if found:
                # Construct graph
                edges = _build_lifted_graph(pet_edges, best_v, m)
                
                # Verify girth
                if _has_girth_at_least(n_candidate, edges, g):
                    return n_candidate, edges
                else:
                    # If verification fails, it's a bug in theory or implementation.
                    # But theoretically, if all non-backtracking walks have non-zero voltage, girth is OK.
                    # Let's assume it passes if valid.
                    # If it fails, we might have a bug in _has_girth_at_least or the walk generation.
                    # For safety, if it fails, we don't return it, but we should investigate.
                    # However, in a competition, returning a potentially invalid graph is bad.
                    # But if the voltage condition is met, it MUST be valid.
                    # Let's trust the voltage condition for speed, but the task says "certified exactly".
                    # So we must return the graph only if _has_girth_at_least passes.
                    # If it fails, we treat it as a failure for this m.
                    pass
            
        candidate_m += 1
    
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
            
            vol = int(voltages[e_idx])
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
