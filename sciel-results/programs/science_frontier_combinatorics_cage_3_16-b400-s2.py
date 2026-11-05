import time
import numpy as np
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    
    def remaining():
        return deadline - time.time()
    
    # ---------------- girth verification ----------------
    def check_girth(n, edges, g, time_limit):
        if n == 0:
            return -1
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return -1
        
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1 # Timeout in check
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- Voltage Lift on Petersen Graph ----------------
    def voltage_lift_search():
        if k != 3 or g != 16:
            return None
        
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

        # Generate constraints
        # Non-backtracking closed walks of length 5 to 15
        max_L = g - 1
        walk_vectors = set()
        
        for start_v in range(n_base):
            # DFS for non-backtracking walks
            stack = [(start_v, -1, 0, [0]*15)]
            while stack:
                current, prev, length, counts = stack.pop()
                
                if length >= max_L:
                    continue
                
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = pet_edges[e_idx]
                    d = 1 if current == u else -1
                    
                    new_counts = counts.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1
                    
                    if next_v == start_v and new_l >= 5:
                        # Normalize vector to avoid duplicates
                        # A vector and its negative represent the same constraint (sum=0 vs sum=0)
                        # Actually, sum=0 is the condition.
                        # If c is a solution, -c is a solution.
                        # So we can fix the sign of the first non-zero element to be positive.
                        vec = tuple(new_counts)
                        if any(x != 0 for x in vec):
                            first_nz = -1
                            for i in range(15):
                                if vec[i] != 0:
                                    first_nz = i
                                    break
                            if first_nz != -1 and vec[first_nz] < 0:
                                vec = tuple(-x for x in vec)
                            walk_vectors.add(vec)
                    
                    stack.append((next_v, current, new_l, new_counts))
        
        if not walk_vectors:
            return None
            
        C = np.array(list(walk_vectors), dtype=np.int64)
        num_constraints = C.shape[0]
        
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True

        def solve_voltage(m, time_limit):
            """
            Find v in (Z_m*)^15 such that C @ v != 0 mod m for all rows.
            """
            v = np.random.randint(1, m, size=15).astype(np.int64)
            
            def cost(v):
                residues = np.mod(C @ v, m)
                return np.sum(residues == 0)
            
            best_v = v.copy()
            best_cost = cost(v)
            
            if best_cost == 0:
                return best_v
            
            T = 1.0
            cooling = 0.999
            start_t = time.time()
            
            while time.time() - start_t < time_limit:
                if remaining() < 5:
                    break
                
                # Pick random variable
                j = random.randint(0, 14)
                old_val = v[j]
                v[j] = random.randint(1, m-1)
                
                new_cost = cost(v)
                delta = new_cost - best_cost
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                        if best_cost == 0:
                            return best_v
                else:
                    v[j] = old_val
                
                T *= cooling
            
            return best_v if best_cost == 0 else None
        
        best_result = None
        
        # Try primes in decreasing order of likelihood of success?
        # Or increasing order to find smallest n?
        # We want smallest n, so smallest m.
        # But we need a valid graph.
        # Start with a large m to ensure we get a valid graph quickly.
        # Then try smaller m.
        
        # Let's try m from 101 down to 11.
        # If we find a solution for m, we update best_result.
        # We continue to smaller m to potentially improve.
        
        # First, ensure we have a valid graph.
        # Try m=101, 97, 89, ...
        primes = [p for p in range(11, 200) if is_prime(p)]
        primes.reverse() # Start with larger primes to ensure success
        
        for m in primes:
            if remaining() < 10:
                break
            
            # Time limit for this m
            t_limit = min(5.0, remaining() / 10.0)
            
            v = solve_voltage(m, t_limit)
            
            if v is not None:
                # Construct graph
                n = 10 * m
                edges = []
                for u, v_base in pet_edges:
                    e_idx = edge_index[(u, v_base)]
                    vol = v[e_idx]
                    for layer in range(m):
                        u_lift = u * m + layer
                        v_lift = v_base * m + (layer + vol) % m
                        if u_lift < v_lift:
                            edges.append((u_lift, v_lift))
                        else:
                            edges.append((v_lift, u_lift))
                
                # Verify
                gc = check_girth(n, edges, g, deadline - 2)
                if gc >= g:
                    best_result = (n, edges)
                    # We found a valid graph. 
                    # If we are iterating downwards, we might find a smaller one.
                    # But if we are iterating upwards, we stop here.
                    # Since we are iterating downwards from 101, we continue to find smaller m.
                    # But we must be careful with time.
        
        return best_result

    res = voltage_lift_search()
    if res is not None:
        return res
    
    # Fallback: return a trivial valid graph if possible?
    # If we fail, we return 0, [] which is invalid.
    # We should try to guarantee a result.
    # The loop above should find one for large m.
    
    return (0, [])

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
