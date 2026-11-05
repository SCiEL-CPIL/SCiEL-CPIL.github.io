import time
import math
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

    # ---------------- Voltage Lift of Petersen Graph with Simulated Annealing ----------------
    def voltage_lift_sa():
        if k != 3 or g != 16:
            return None

        # Petersen Graph
        # Vertices 0-4: outer pentagon
        # Vertices 5-9: inner star
        base_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        n_edges = len(base_edges)
        adj = [[] for _ in range(10)]
        for u, v in base_edges:
            adj[u].append(v)
            adj[v].append(u)

        # Precompute edge indices for direction
        edge_index = {}
        for i, (u, v) in enumerate(base_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i

        # We want to find voltage assignments x in Z_p^15 such that
        # sum(x_e) != 0 mod p for all closed walks of length < g.
        
        # We will try primes p starting from 53 (10*53=530 > 510 Moore bound)
        # Moore bound for (3,16) is 510.
        # Smallest known is 936.
        # If we find a valid lift for p=53, n=530.
        
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0: return False
            for i in range(3, int(math.isqrt(n)) + 1, 2):
                if n % i == 0: return False
            return True

        # Find next prime >= n
        def next_prime(n):
            cand = n if n % 2 != 0 else n + 1
            while not is_prime(cand):
                cand += 2
            return cand

        # Start from p=53
        p = next_prime(53)
        
        # We will try to find a solution for this p.
        # If failed, try next prime, etc.
        # Max p such that 10*p < some limit?
        # The known best is 936. So p up to 93 is 930.
        # We can try up to p=100.
        
        while p <= 100 and remaining() > 10:
            if not is_prime(p):
                p = next_prime(p + 1)
                continue
                
            n_vertices = 10 * p
            
            # Simulated Annealing
            # State: x (array of 15 ints mod p)
            # Cost: Number of closed walks of length < 16 with sum 0.
            # We can't enumerate all walks easily in SA loop.
            # Instead, we check validity directly.
            # But SA needs a cost function.
            # We can use a subset of walks, or just random restarts + local search.
            
            # Better: Random restarts with local search (Hill Climbing).
            # Objective: Minimize number of short cycles in the lifted graph.
            # But checking girth is O(N).
            # We can check "short" cycles (length < 16) specifically.
            
            def count_short_cycles(x, p):
                # Build lifted graph
                adj_lift = [[] for _ in range(10 * p)]
                for u, v in base_edges:
                    e_idx = edge_index[(u, v)]
                    shift = x[e_idx] % p
                    for i in range(p):
                        u_lift = u * p + i
                        v_lift = v * p + (i + shift) % p
                        adj_lift[u_lift].append(v_lift)
                        adj_lift[v_lift].append(u_lift)
                
                count = 0
                # Check cycles of length 3 to 15
                # We can do BFS from each vertex and count back-edges that form short cycles.
                # Or just run check_girth and see if it's < 16.
                # For SA, we need a score.
                # Let's just return 1 if girth < 16, else 0.
                # This is a 0/1 function, which is hard for SA.
                # Better: Count the number of "bad" walks of length L for small L.
                
                # Let's stick to Hill Climbing with random restarts.
                # If we find a valid one, return it.
                
                return check_girth(10 * p, [ (u, v) for u in range(10*p) for v in adj_lift[u] if u < v ], g, time.time() + 1)
            
            # This is too slow for SA inner loop.
            # Instead, we use the property that we only care about sum of voltages on closed walks.
            # We can precompute the "walk vectors" for the base graph.
            # A closed walk in the base graph corresponds to a vector v in Z^15.
            # The sum of voltages is v . x.
            # We need v . x != 0 mod p for all v corresponding to walks of length < 16.
            
            # Let's generate walk vectors for lengths 3 to 15.
            # This is a one-time cost per p? No, independent of p.
            walk_vectors = set()
            for start_v in range(10):
                # DFS up to depth 15
                # Stack: (current, prev, depth, path_edges, path_dirs)
                stack = [(start_v, -1, 0, [], [])]
                while stack:
                    cur, prev, depth, p_edges, p_dirs = stack.pop()
                    if depth >= 3 and cur == start_v:
                        vec = [0] * 15
                        for e_idx, d in zip(p_edges, p_dirs):
                            vec[e_idx] += d
                        # Normalize vector? 
                        # (1, -1) and (-1, 1) are same walk reversed.
                        # Sum is same magnitude, opposite sign.
                        # We just need to ensure sum != 0.
                        walk_vectors.add(tuple(vec))
                    if depth == 15:
                        continue
                    for next_v in adj[cur]:
                        if next_v == prev:
                            continue
                        e_idx = edge_index[(cur, next_v)]
                        u, v = base_edges[e_idx]
                        d = 1 if cur == u else -1
                        stack.append((next_v, cur, depth + 1, p_edges + [e_idx], p_dirs + [d]))
            
            C = np.array(list(walk_vectors), dtype=np.int64)
            
            def is_valid(x, p):
                S = (C @ x) % p
                return not np.any(S == 0)
            
            # Search for x
            found = False
            rng = np.random.default_rng(p)
            
            # Try random restarts
            max_restarts = 5000
            for _ in range(max_restarts):
                if remaining() < 5:
                    break
                
                x = rng.integers(0, p, size=15)
                
                # Local Search: Flip one voltage at a time
                best_x = x
                best_valid = is_valid(x, p)
                
                if best_valid:
                    found = True
                    break
                
                # Simple Hill Climbing
                for _ in range(100): # iterations per restart
                    if remaining() < 5:
                        break
                    i = rng.integers(0, 15)
                    old_val = x[i]
                    new_val = rng.integers(0, p)
                    x[i] = new_val
                    
                    if is_valid(x, p):
                        found = True
                        break
                    else:
                        # Revert if it didn't help? 
                        # Since it's 0/1, we can't measure "help".
                        # So we just keep the new state.
                        pass
                
                if found:
                    break
            
            if found:
                # Construct edges
                edges_set = set()
                for u, v in base_edges:
                    e_idx = edge_index[(u, v)]
                    shift = x[e_idx] % p
                    for i in range(p):
                        u_lift = u * p + i
                        v_lift = v * p + (i + shift) % p
                        if u_lift < v_lift:
                            edges_set.add((u_lift, v_lift))
                        else:
                            edges_set.add((v_lift, u_lift))
                
                final_edges = list(edges_set)
                
                # Final verification
                gc = check_girth(10 * p, final_edges, g, deadline - 5)
                if gc == -1:
                    return None
                if gc >= g:
                    return (10 * p, final_edges)
                else:
                    # Should not happen if is_valid worked, but just in case
                    continue
            
            p = next_prime(p + 1)
            
        return None

    # Try the voltage lift
    result = voltage_lift_sa()
    if result is not None:
        return result

    # Fallback: Return a known valid graph if we have one?
    # Or just a large graph.
    # The prompt implies we should return *something* valid.
    # If SA fails, we can use the PSL(2,q) from the original code as a fallback,
    # but that was slow.
    # Let's just return a large Petersen lift with a large prime to be safe,
    # though it will score poorly.
    
    # Actually, if SA fails, we can try the original PSL code with a time limit.
    # But for this rewrite, I will focus on the SA.
    
    # If all else fails, return a simple cycle (invalid girth) or a large random graph?
    # No, must be valid.
    # Let's use the LUW construction from the original code as a fallback.
    
    def luw_construct():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
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

    return luw_construct()
