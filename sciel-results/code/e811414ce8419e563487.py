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
        
        min_cycle = float('inf')
        
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            
            while queue:
                node = queue.popleft()
                d = dist[node]
                
                if d >= g // 2:
                    continue

                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = d + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < g:
                            return min_cycle
                    else:
                        dist[nb] = d + 1
                        parent[nb] = node
                        queue.append(nb)
                        
        return min_cycle

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
    walk_vectors = []
    max_L = g - 1
    
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
                    if any(x != 0 for x in new_vec):
                        walk_vectors.append(tuple(new_vec))
                
                if new_l < max_L:
                    stack.append((nb, node, new_l, new_vec))

    if not walk_vectors:
        return (0, [])

    # Normalize and deduplicate
    unique_constraints = set()
    for c in walk_vectors:
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
    
    # Precompute C for fast matrix-vector products
    C_rows = C.shape[0]
    
    def solve_voltage(m, max_iters, time_limit, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**31 - 1))
        
        # Greedy initialization: try to satisfy constraints one by one
        v = np.zeros(15, dtype=np.int64)
        
        # First pass: random initialization
        v = np.random.randint(1, m, size=15).astype(np.int64)
        
        def get_cost(v):
            residues = np.mod(C @ v, m)
            return np.sum(residues == 0)
            
        best_v = v.copy()
        best_cost = get_cost(v)
        
        if best_cost == 0:
            return v, 0

        # Local search with simulated annealing
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
                
            # Pick random index to change
            j = random.randint(0, 14)
            old_val = v[j]
            new_val = random.randint(1, m - 1)
            if new_val == old_val:
                continue
            
            v[j] = new_val
            new_cost = get_cost(v)
            delta = new_cost - best_cost
            
            if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_v = v.copy()
                    if new_cost == 0:
                        return v, 0
            else:
                v[j] = old_val
                
            T *= cooling

        return best_v, best_cost

    # Try to find smallest m where a valid voltage assignment exists
    # Start from theoretical minimum and go up
    min_m = 53  # n = 10*m >= 512 => m >= 51.2, so m >= 52, smallest prime is 53
    
    # Try primes and some composites
    candidates = []
    m = min_m
    while m < 200:
        candidates.append(m)
        m += 1
    
    best_result = None
    
    for current_m in candidates:
        if remaining() < 3:
            break
            
        # Try multiple restarts for this m
        attempts = 0
        max_attempts = 20
        found = False
        
        while attempts < max_attempts and remaining() > 3 and not found:
            v, cost = solve_voltage(current_m, max_iters=3000, time_limit=deadline - 2, seed=attempts * 7919 + 12345)
            if cost == 0:
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
                gc = check_girth_fast(n, edges, g, deadline - 1)
                if gc >= g:
                    return (n, edges)
                else:
                    # Verification failed
                    break
            attempts += 1
    
    # Fallback: return empty if nothing found
    return (0, [])

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
