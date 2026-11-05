import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    def check_girth_fast(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Verify 3-regularity and simplicity
        for i in range(n):
            if len(adj[i]) != k:
                return -1
            if len(set(adj[i])) != k:
                return -1
        
        # Use non-backtracking walk enumeration from each vertex
        # A cycle of length < g exists iff there's a non-backtracking closed walk of length < g
        # that isn't just going back and forth (which is prevented by non-backtracking)
        
        for s in range(n):
            if time.time() > time_limit:
                return -1
            
            # BFS with parent tracking to find shortest cycle
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            
            while queue:
                node = queue.popleft()
                d = dist[node]
                
                if d >= g - 1:
                    continue

                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] == -1:
                        dist[nb] = d + 1
                        parent[nb] = node
                        queue.append(nb)
                    else:
                        # Found a back edge or cross edge
                        # The cycle length is d + dist[nb] + 1
                        # But we need to be careful: this only works if the paths are disjoint
                        # For a proper cycle detection, we need to check if s is on the cycle
                        # Actually, for girth checking, any cycle found is valid
                        cycle_len = d + dist[nb] + 1
                        if cycle_len < g:
                            return cycle_len
                        
        return g  # No cycle shorter than g found

    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0: return False
            i += 6
        return True

    # Petersen Graph Definition
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    adj = [[] for _ in range(n_base)]
    for u, v in pet_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Generate constraints: non-backtracking closed walks of length < g
    # We need C @ v != 0 (mod m) for all such walks.
    # Walk vector w is a 15-dim vector.
    # If walk is closed, sum of edge traversals is 0 in the vertex space, but we track edge counts.
    # Actually, for a closed walk, the net displacement in the voltage graph is sum(v_e * dir_e).
    # We want this sum to be non-zero mod m.
    
    walk_vectors = []
    max_L = g - 1
    
    # DFS to find all closed walks of length 3 to max_L
    # State: (node, prev, length, vec)
    # Optimization: We can stop if length > max_L.
    # We only care about closed walks starting at s.
    
    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0]*15)]
        while stack:
            node, prev, length, vec = stack.pop()
            if length > max_L:
                continue
            
            for nb in adj[node]:
                if nb == prev:
                    continue
                e_idx = edge_index[(node, nb)]
                u, v = pet_edges[e_idx]
                d = 1 if node == u else -1
                
                new_vec = vec.copy()
                new_vec[e_idx] += d
                new_l = length + 1
                
                if nb == start_v and new_l >= 3:
                    # Check if vector is non-zero
                    if any(x != 0 for x in new_vec):
                        walk_vectors.append(tuple(new_vec))
                
                if new_l < max_L:
                    stack.append((nb, node, new_l, new_vec))

    if not walk_vectors:
        return (0, [])

    # Normalize and deduplicate
    unique_constraints = set()
    for c in walk_vectors:
        # c and -c are equivalent constraints (c.v=0 iff -c.v=0)
        # Normalize by making first non-zero element positive
        first_nz = -1
        for i in range(15):
            if c[i] != 0:
                first_nz = i
                break
        if first_nz != -1:
            if c[first_nz] < 0:
                c = tuple(-x for x in c)
            unique_constraints.add(c)
            
    C = np.array(list(unique_constraints), dtype=np.int64)
    if C.size == 0:
        return (0, [])
    
    # Precompute row indices for non-zero entries to speed up cost calculation?
    # C is (M, 15). M can be large.
    # C @ v is (M,).
    
    # Search for voltage assignment
    # We want v in (Z_m)^15 such that (C @ v) % m != 0 for all rows.
    
    # Try primes starting from small values.
    # Theoretical lower bound for n(3,16) is 512.
    # n = 10 * m. So m >= 52.
    # Smallest prime >= 52 is 53. n=530.
    
    min_m = 53
    m = min_m
    best_result = None
    
    # Heuristic: Local search / Simulated Annealing on the voltage assignment.
    # Objective: Minimize number of violated constraints (residues == 0).
    
    def solve_voltage(m, time_limit):
        # Use local search with restarts
        # Objective: minimize number of violated constraints
        best_v = None
        best_cost = float('inf')
        
        num_restarts = 20
        
        for restart in range(num_restarts):
            if time.time() > time_limit:
                break
            
            v = np.random.randint(1, m, size=15).astype(np.int64)
            
            def get_cost(v):
                residues = np.mod(C @ v, m)
                return np.sum(residues == 0)
            
            current_cost = get_cost(v)
            
            if current_cost == 0:
                return v, 0
            
            # Local search: hill climbing with random restarts within
            max_steps = 500
            for step in range(max_steps):
                if time.time() > time_limit:
                    break
                
                # Try all possible single-coordinate changes, pick best
                best_delta = 0
                best_j = -1
                best_val = -1
                
                for j in range(15):
                    old_val = v[j]
                    # Try a few random values for this coordinate
                    for trial in range(3):
                        new_val = random.randint(1, m - 1)
                        if new_val == old_val:
                            continue
                        v[j] = new_val
                        delta = get_cost(v) - current_cost
                        if delta < best_delta:
                            best_delta = delta
                            best_j = j
                            best_val = new_val
                    v[j] = old_val
                
                if best_j == -1 or best_delta == 0:
                    break  # Local minimum
                
                v[best_j] = best_val
                current_cost += best_delta
                
                if current_cost == 0:
                    return v, 0
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_v = v.copy()
        
        return best_v, best_cost

    # Iterate over primes
    # We want to find the smallest m (and thus n) such that a valid voltage assignment exists.
    # If we fail for a small m, it might be due to search failure, not impossibility.
    # But for small m, the space is small, so search is more likely to succeed if a solution exists.
    
    current_m = min_m
    while current_m < 1000 and remaining() > 10:
        if is_prime(current_m):
            # Try to solve for this m
            v, cost = solve_voltage(current_m, time_limit=deadline - 5)
            if cost == 0 and v is not None:
                # Construct graph
                n = 10 * current_m
                edges = []
                for u, v_base in pet_edges:
                    e_idx = edge_index[(u, v_base)]
                    vol = v[e_idx]
                    for layer in range(current_m):
                        u_lift = u * current_m + layer
                        v_lift = v_base * current_m + (layer + vol) % current_m
                        if u_lift < v_lift:
                            edges.append((u_lift, v_lift))
                        else:
                            edges.append((v_lift, u_lift))
                
                # Verify
                gc = check_girth_fast(n, edges, g, deadline - 2)
                if gc >= g:
                    return (n, edges)
                # If verification fails, try next m
        current_m += 1
    
    # Fallback: return a valid but larger graph if we can find one
    # Try a few more m values with less time
    for m_try in [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if remaining() < 3:
            break
        if is_prime(m_try):
            v, cost = solve_voltage(m_try, time_limit=deadline - 1)
            if cost == 0 and v is not None:
                n = 10 * m_try
                edges = []
                for u, v_base in pet_edges:
                    e_idx = edge_index[(u, v_base)]
                    vol = v[e_idx]
                    for layer in range(m_try):
                        u_lift = u * m_try + layer
                        v_lift = v_base * m_try + (layer + vol) % m_try
                        if u_lift < v_lift:
                            edges.append((u_lift, v_lift))
                        else:
                            edges.append((v_lift, u_lift))
                
                gc = check_girth_fast(n, edges, g, deadline - 0.5)
                if gc >= g:
                    return (n, edges)
    
    return (0, [])

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
