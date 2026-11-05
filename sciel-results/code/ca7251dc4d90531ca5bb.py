import time
import numpy as np
import random
from math import gcd

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Strategy: Voltage lift of Petersen graph with optimized constraint solving.
    """
    start_time = time.time()
    deadline = start_time + 118.0
    
    def remaining():
        return deadline - time.time()
    
    # ---------------- Girth verification ----------------
    def check_girth(n, edges, g, time_limit):
        """Verify no cycle shorter than g using non-backtracking BFS from every vertex."""
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
    
    # ---------------- Voltage lift of Petersen graph ----------------
    def voltage_lift_petersen():
        if k != 3 or g != 16:
            return None
        
        # Petersen graph edges (10 vertices, 15 edges)
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer cycle
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner star
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
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
        
        # Generate all non-backtracking closed walks of length < g
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
                    
                    if next_v == start_v and new_l >= 3:
                        vec = tuple(new_counts)
                        if any(x != 0 for x in vec):
                            first_nz = -1
                            for i in range(15):
                                if vec[i] != 0:
                                    first_nz = i
                                    break
                            if first_nz != -1:
                                if vec[first_nz] < 0:
                                    vec = tuple(-x for x in vec)
                                walk_vectors.add(vec)
                    
                    stack.append((next_v, current, new_l, new_counts))
        
        if not walk_vectors:
            return None
            
        unique_constraints = list(walk_vectors)
        C = np.array(unique_constraints, dtype=np.int64)
        
        if C.size == 0:
            return None
        
        # Precompute sparse representation for fast incremental cost updates
        # For each edge index, store list of (row_idx, coeff)
        edge_to_rows = [[] for _ in range(15)]
        for row_idx in range(len(C)):
            for col_idx in range(15):
                if C[row_idx, col_idx] != 0:
                    edge_to_rows[col_idx].append((row_idx, C[row_idx, col_idx]))
        
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True
        
        def solve_voltage(m, max_iters=5000, seed=None):
            """Fast simulated annealing with incremental cost updates."""
            rng = np.random.RandomState(seed)
            v = rng.randint(1, m, size=15).astype(np.int64)
            
            # Initialize cost
            prod = C @ v
            residues = np.mod(prod, m)
            cost = int(np.sum(residues == 0))
            
            best_v = v.copy()
            best_cost = cost
            
            T = 2.0
            cooling = 0.99995
            no_improve = 0
            
            for it in range(max_iters):
                if remaining() < 8:
                    break
                
                # Pick a random variable to change
                j = rng.randint(0, 15)
                old_val = v[j]
                new_val = rng.randint(1, m)
                if new_val == old_val:
                    continue
                v[j] = new_val
                
                # Incremental cost update
                delta_cost = 0
                for row_idx, coeff in edge_to_rows[j]:
                    old_res = prod[row_idx] % m
                    new_res = (prod[row_idx] + coeff * (new_val - old_val)) % m
                    prod[row_idx] += coeff * (new_val - old_val)
                    old_zero = 1 if old_res == 0 else 0
                    new_zero = 1 if new_res == 0 else 0
                    delta_cost += new_zero - old_zero
                
                cost += delta_cost
                
                if cost < best_cost:
                    best_cost = cost
                    best_v = v.copy()
                    no_improve = 0
                    if cost == 0:
                        return v, 0
                else:
                    no_improve += 1
                    if no_improve > 500:
                        # Restart with new random initialization
                        v = rng.randint(1, m, size=15).astype(np.int64)
                        prod = C @ v
                        residues = np.mod(prod, m)
                        cost = int(np.sum(residues == 0))
                        no_improve = 0
                        T = 2.0
                        continue
                
                # Accept/reject
                if cost > best_cost:
                    delta = cost - best_cost
                    if delta > 0 and rng.random() < np.exp(-delta / T):
                        pass  # Accept
                    else:
                        v[j] = old_val
                        # Undo cost change
                        for row_idx, coeff in edge_to_rows[j]:
                            old_res = prod[row_idx] % m
                            new_res = (prod[row_idx] - coeff * (new_val - old_val)) % m
                            prod[row_idx] -= coeff * (new_val - old_val)
                            old_zero = 1 if old_res == 0 else 0
                            new_zero = 1 if new_res == 0 else 0
                            delta_cost -= new_zero - old_zero
                        cost -= delta_cost
                
                T *= cooling
                if T < 0.01:
                    T = 0.01
            
            return best_v, best_cost
        
        # Try primes m - focus on values near 93.6 (known best is 936 = 10*93.6)
        # Try a mix: some smaller primes and primes around 89-97
        primes_to_try = []
        m = 53
        while m < 150:
            if is_prime(m):
                primes_to_try.append(m)
            m += 1
        
        # Reorder: try primes near 93 first (most likely to give good results),
        # then smaller ones, then larger ones
        def priority(m):
            return abs(m - 94)
        
        primes_to_try.sort(key=priority)
        
        best_result = None
        
        for m in primes_to_try:
            if remaining() < 15:
                break
            
            # Try multiple restarts for this m
            for restart in range(30):
                if remaining() < 10:
                    break
                
                v, cost = solve_voltage(m, max_iters=3000, seed=restart * 1000 + m)
                
                if cost == 0:
                    # Construct graph
                    n = 10 * m
                    edges = []
                    for u, v_base in pet_edges:
                        e_idx = edge_index[(u, v_base)]
                        vol = int(v[e_idx])
                        for layer in range(m):
                            u_lift = u * m + layer
                            v_lift = v_base * m + (layer + vol) % m
                            if u_lift < v_lift:
                                edges.append((u_lift, v_lift))
                            else:
                                edges.append((v_lift, u_lift))
                    
                    # Verify
                    gc = check_girth(n, edges, g, deadline - 3)
                    if gc >= g:
                        if best_result is None or n < best_result[0]:
                            best_result = (n, edges)
                        # If we found a good one, we can try to find even smaller
                        if n <= 1000:
                            return best_result
                    break  # Found valid solution for this m, move to next m
        
        return best_result
    
    # ---------------- LUW fallback (for other k values) ----------------
    def luw_fallback(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
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
        index, order, edges_luw = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges_luw.append((index[vertex], index[other]))
        return len(order), edges_luw
    
    # ---------------- Main ----------------
    res = voltage_lift_petersen()
    if res is not None:
        return res
    
    # Fallback
    return luw_fallback(k, g)

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
