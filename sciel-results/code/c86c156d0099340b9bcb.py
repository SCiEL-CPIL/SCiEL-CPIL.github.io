import time
import numpy as np
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # ---------------- Girth Verification ----------------
    def check_girth(n, edges, g, time_limit):
        """
        Checks if the graph has girth at least g.
        Uses non-backtracking BFS from every vertex.
        Returns the girth if < g, else g (or larger).
        """
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        min_cycle = float('inf')
        
        for s in range(n):
            if time.time() > time_limit:
                return -1
            
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                d = dist[node]
                
                if d >= g - 1:
                    continue
                    
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < g:
                            return min_cycle
                    else:
                        dist[nb] = d + 1
                        parent[nb] = node
                        if d + 1 < g:
                            queue.append(nb)
        
        if min_cycle < g:
            return min_cycle
        return min_cycle if min_cycle != float('inf') else g

    # ---------------- Voltage Lift of Petersen ----------------
    def voltage_lift_petersen():
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

        # Generate all non-backtracking closed walks of length [5, 15]
        max_L = g - 1
        walk_vectors = set()
        
        for start_v in range(n_base):
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
                        walk_vectors.add(tuple(new_counts))
                    
                    stack.append((next_v, current, new_l, new_counts))
                    
        if not walk_vectors:
            return None
            
        unique_constraints = set()
        for c in walk_vectors:
            if all(x == 0 for x in c):
                continue
            first_non_zero = -1
            for i in range(15):
                if c[i] != 0:
                    first_non_zero = i
                    break
            if first_non_zero != -1:
                if c[first_non_zero] < 0:
                    c = tuple(-x for x in c)
                unique_constraints.add(c)
                
        C = np.array(list(unique_constraints), dtype=np.int64)
        if C.size == 0:
            return None

        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True

        # Solve voltage assignment using gradient descent on a smooth surrogate
        # Cost = sum( cos(2*pi * (C @ v) / m) ) + regularization
        # We want to minimize this, ideally getting negative values (close to -1 per term)
        # Actually, we want (C @ v) % m != 0. 
        # If we use exp(-t * ((C@v % m) / m)^2) it's not smooth.
        
        # Let's stick to Simulated Annealing but with a better cost function 
        # that accounts for "closeness" to 0 to allow better annealing.
        
        def solve_voltage(m, max_iters=10000):
            v = np.random.randint(0, m, size=15).astype(np.int64)
            
            def cost(v):
                residues = np.mod(C @ v, m)
                # Penalty for being 0. 
                # Smooth approximation: sum(1 - cos(2*pi*residues/m)) is 0 at 0, max at m/2.
                # We want to MAXIMIZE this (distance from 0).
                # So MINIMIZE negative of it.
                return -np.sum(1 - np.cos(2 * np.pi * residues / m))

            best_v = v.copy()
            best_cost = cost(v)

            T = 5.0
            cooling = 0.99
            min_T = 0.01
            
            for it in range(max_iters):
                if remaining() < 5:
                    break

                j = random.randint(0, 14)
                old_val = v[j]
                v[j] = random.randint(0, m - 1)

                new_cost = cost(v)
                delta = new_cost - best_cost

                if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                else:
                    v[j] = old_val

                T *= cooling

            # Check if solution is valid (no residues 0)
            residues = np.mod(C @ best_v, m)
            if np.any(residues == 0):
                return None, 1
            return best_v, 0

        # Try primes in increasing order
        m = 53
        best_result = None
        
        while m < 1000 and remaining() > 20:
            if is_prime(m):
                num_restarts = 8
                for restart in range(num_restarts):
                    if remaining() < 10:
                        break

                    v, cost = solve_voltage(m, max_iters=8000)

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
                        gc = check_girth(n, edges, g, deadline - 5)
                        if gc >= g:
                            return (n, edges)
            
            m += 1
            
        return None

    res = voltage_lift_petersen()
    if res is not None:
        return res
        
    return (0, [])

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
