import time
import random
import math
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses voltage lift of Petersen graph for k=3.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    if k == 3:
        return _construct_petersen_lift(g, deadline, start_time)
    else:
        # Fallback for other k
        return _construct_luw(k, g, d=7)

def _construct_petersen_lift(g, deadline, start_time):
    """
    Constructs a girth-g graph by lifting the Petersen graph with voltages in Z_m.
    Uses Simulated Annealing for voltage assignment.
    """
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    n_edges = 15
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    # Generate constraints: non-backtracking closed walks of length < g
    coeff_vectors = set()
    max_walk_len = g - 1
    
    for start_v in range(n_base):
        # State: (current_v, prev_v, length, counts_tuple)
        # Use stack for DFS
        stack = [(start_v, -1, 0, tuple([0]*n_edges))]
        while stack:
            v, prev_v, l, counts = stack.pop()
            if l >= max_walk_len:
                continue
            
            for next_v in range(n_base):
                if next_v == prev_v or next_v == v:
                    continue
                e_idx = edge_index.get((v, next_v))
                if e_idx is None:
                    continue
                
                new_counts_list = list(counts)
                u_edge, v_edge = pet_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    new_counts_list[e_idx] += 1
                else:
                    new_counts_list[e_idx] -= 1
                
                new_l = l + 1
                if next_v == start_v and new_l > 0:
                    # Closed walk found
                    c = tuple(new_counts_list)
                    # Normalize: make first non-zero positive
                    for i in range(n_edges):
                        if c[i] != 0:
                            if c[i] < 0:
                                c = tuple(-x for x in c)
                            break
                    if any(c): # Skip zero vector (shouldn't happen for simple graph walks)
                        coeff_vectors.add(c)
                
                stack.append((next_v, v, new_l, tuple(new_counts_list)))

    coeff_list = list(coeff_vectors)
    if not coeff_list:
        # Should not happen for girth > 5
        return _construct_luw(3, g, d=11)

    A = np.array(coeff_list, dtype=np.int64)
    n_constraints = len(A)
    
    # Precompute affected rows for each edge as numpy arrays for vectorization
    affected_rows_np = []
    for j in range(n_edges):
        rows = np.nonzero(A[:, j])[0]
        affected_rows_np.append(rows)

    best_n = float('inf')
    best_edges = None

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
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
        found = _solve_sa_vectorized(A, affected_rows_np, m, n_constraints, n_edges, deadline)
        if found is not None:
            voltages = found
            edges = _build_lifted_graph(pet_edges, voltages, m)
            # Verify girth
            if _has_girth_at_least(n_candidate, edges, g):
                return n_candidate, edges
            # If verification fails, something is wrong, but theoretically it should pass.
            # Return anyway as best effort? No, must be valid.
            # If it fails, skip this m.
            
    # Fallback
    return _construct_luw(3, g, d=11)

def _solve_sa_vectorized(A, affected_rows_np, m, n_constraints, n_edges, deadline):
    """
    Solve for voltages using Vectorized Simulated Annealing.
    Returns voltages array or None.
    """
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize
        x = np.array([random.randint(1, m-1) for _ in range(n_edges)], dtype=np.int64)
        sums = (A @ x) % m
        violations = np.sum(sums == 0)
        
        if violations == 0:
            return x

        T = 10.0
        cooling = 0.99
        min_T = 0.1
        
        steps = 0
        max_steps_per_m = 200000 # Reduced to allow restarts
        
        while T > min_T and steps < max_steps_per_m:
            if time.time() > deadline - 5:
                break
            
            # Pick random edge
            j = random.randint(0, n_edges - 1)
            # Pick random delta
            delta = random.randint(1, m - 1)
            
            old_val = x[j]
            new_val = (old_val + delta) % m
            
            # Calculate change in violations using vectorization
            rows = affected_rows_np[j]
            if len(rows) == 0:
                steps += 1
                continue
                
            # Get current sums for affected rows
            old_sums = sums[rows]
            # Calculate new sums
            delta_sums = A[rows, j] * delta
            new_sums = (old_sums + delta_sums) % m
            
            # Count violations
            old_violations = np.sum(old_sums == 0)
            new_violations = np.sum(new_sums == 0)
            delta_violations = new_violations - old_violations
            
            # Acceptance
            if delta_violations < 0:
                # Improve
                x[j] = new_val
                sums[rows] = new_sums
                violations += delta_violations
                if violations == 0:
                    return x
            else:
                # Accept with prob exp(-delta_violations / T)
                if delta_violations == 0 or random.random() < math.exp(-delta_violations / T):
                    x[j] = new_val
                    sums[rows] = new_sums
                    violations += delta_violations
            
            T *= cooling
            steps += 1
            
    return None

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
