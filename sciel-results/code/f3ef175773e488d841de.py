import time
import math
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    def remaining():
        return deadline - time.time()

    # ---------------- Fast Girth Verification ----------------
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return -1 # Invalid

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

    # ---------------- LUW Fallback (Guaranteed, but large) ----------------
    def _luw_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def luw_construct():
        q = k
        # d is roughly half the girth
        d = max(3, (g - 2) // 2)
        add, sub, mul = _luw_ops(q)
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
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # ---------------- Voltage Lift on Petersen Graph ----------------
    def voltage_lift_search():
        if k != 3 or g != 16:
            return None

        # Petersen Graph Definition
        # Vertices 0-4 form outer cycle, 5-9 form inner star
        base_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # 0-4
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # 5-9
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # 10-14
        ]
        n_base = 10
        n_edges_base = len(base_edges)
        
        # Build adjacency with edge indices
        base_adj = [[] for _ in range(n_base)]
        for i, (u, v) in enumerate(base_edges):
            base_adj[u].append((v, i))
            base_adj[v].append((u, i))

        # Generate all closed non-backtracking walks of length 3 to g-1
        # We represent a walk by the net voltage sum for each edge.
        # If a walk traverses edge e forward, it contributes +1 to e's coefficient.
        # If backward, -1.
        # The voltage of the lifted walk is sum(coeff[e] * voltage[e]).
        # We need this sum NOT to be 0 mod m.
        
        def get_closed_nb_walks(max_len):
            walks = set()
            for start_v in range(n_base):
                stack = []
                for next_v, e_id in base_adj[start_v]:
                    u, v = base_edges[e_id]
                    sign = 1 if u == start_v else -1
                    stack.append((next_v, e_id, 1, [(e_id, sign)]))
                
                while stack:
                    curr_v, prev_e, length, path = stack.pop()
                    if length >= max_len:
                        continue
                    if curr_v == start_v and length >= 3:
                        # Convert path to coefficient vector
                        coeff = [0] * n_edges_base
                        for eid, sgn in path:
                            coeff[eid] += sgn
                        walks.add(tuple(coeff))
                    
                    for next_v, e_id in base_adj[curr_v]:
                        if e_id == prev_e:
                            continue
                        u, v = base_edges[e_id]
                        sign = 1 if u == curr_v else -1
                        new_path = path + [(e_id, sign)]
                        stack.append((next_v, e_id, length + 1, new_path))
            return list(walks)

        closed_walks = get_closed_nb_walks(g)
        
        # Convert to sparse format for faster dot products if needed, 
        # but NumPy vectorized dot is fast enough for ~100-200 walks and 15 edges.
        # Let's use a dense matrix C where C[i, j] is coeff of edge j in walk i.
        if not closed_walks:
            return None
            
        C = np.array(closed_walks, dtype=np.int64)
        
        # Moore bound: n >= 512. n = 10 * m. So m >= 52.
        min_m = 52
        
        # Search for m and voltage assignment x
        # We try small m first.
        # For each m, we run simulated annealing / random restarts.
        
        rng = np.random.default_rng(42)
        
        # Precompute a list of primes or just iterate integers?
        # Integers are fine, but primes often yield better distribution? 
        # Actually, any m works. Let's iterate m from 52 upwards.
        
        # To speed up, we can filter walks that are linearly dependent? 
        # Not necessary for small numbers.
        
        best_n = float('inf')
        best_edges = None
        
        # We can parallelize or just do sequential with tight loop.
        # Time limit is ~140s.
        
        for m in range(min_m, 200): # Cap at 200 for safety, 2000 vertices is small enough
            if time.time() > deadline - 20:
                break
                
            # If 10*m is already worse than our best found, skip
            if 10 * m >= best_n:
                continue
                
            # Try to find a valid voltage assignment for this m
            # Score: number of walks with sum 0 mod m. We want 0.
            
            found = False
            # Number of restarts depends on time
            max_restarts = 5000 if m < 60 else 2000
            
            for restart in range(max_restarts):
                if time.time() > deadline - 10:
                    break
                    
                # Initialize random voltage
                x = rng.integers(0, m, size=n_edges_base)
                
                # Simulated Annealing
                T = float(m)
                cooling = 0.995
                best_local = x.copy()
                
                # Calculate initial score
                S = (C @ x) % m
                current_score = np.sum(S == 0)
                best_local_score = current_score
                
                if current_score == 0:
                    found = True
                    break
                    
                for step in range(2000):
                    if time.time() > deadline - 5:
                        break
                    if best_local_score == 0:
                        break
                        
                    # Pick a random edge to change
                    idx = rng.integers(0, n_edges_base)
                    old_val = x[idx]
                    # Change to a random other value
                    new_val = rng.integers(0, m)
                    if new_val == old_val:
                        continue
                    x[idx] = new_val
                    
                    # Calculate delta score
                    # Only need to check walks that involve edge idx
                    # But C @ x is fast (15 * ~200 ops)
                    S = (C @ x) % m
                    new_score = np.sum(S == 0)
                    delta = new_score - best_local_score
                    
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        if new_score < best_local_score:
                            best_local_score = new_score
                            best_local = x.copy()
                        # x is already updated
                    else:
                        x[idx] = old_val
                        
                    T *= cooling
                
                if best_local_score == 0:
                    found = True
                    x = best_local
                    break
            
            if found:
                # Construct the graph
                n = 10 * m
                edges_set = set()
                for i, (u, v) in enumerate(base_edges):
                    shift = x[i] % m
                    for t in range(m):
                        u_lift = u * m + t
                        v_lift = v * m + (t + shift) % m
                        if u_lift < v_lift:
                            edges_set.add((u_lift, v_lift))
                        else:
                            edges_set.add((v_lift, u_lift))
                
                final_edges = list(edges_set)
                
                # Verify
                gc = check_girth(n, final_edges, g, deadline - 2)
                if gc == -1:
                    # Timeout or error during check, ignore
                    continue
                if gc >= g:
                    best_n = n
                    best_edges = final_edges
                    break # Found a valid graph for this m, try smaller m? 
                    # Actually, we iterate m from small to large. 
                    # So the first valid m is the smallest n.
                    # We can break the outer loop too.
                
        if best_edges is not None:
            return best_n, best_edges
        return None

    # Try voltage lift first
    result = voltage_lift_search()
    if result is not None:
        return result

    # Fallback to LUW
    return luw_construct()

if __name__ == "__main__":
    # Quick self-test
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices and {len(edges)} edges")
