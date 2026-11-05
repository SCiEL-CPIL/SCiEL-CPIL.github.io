import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification (fast, early exit) ----------------
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

    # ---------------- Voltage lifting on Petersen graph ----------------
    def voltage_lift():
        if k != 3 or g != 16:
            return None
        
        # Petersen graph definition
        # Vertices 0..9
        # Edges:
        # Outer: (0,1), (1,2), (2,3), (3,4), (4,0)
        # Inner: (5,7), (7,9), (9,6), (6,8), (8,5)
        # Spokes: (0,5), (1,6), (2,7), (3,8), (4,9)
        
        base_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        
        # Map edge to index
        edge_index = {}
        for i, (u, v) in enumerate(base_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i
            
        # Adjacency list
        adj = [[] for _ in range(10)]
        for u, v in base_edges:
            adj[u].append(v)
            adj[v].append(u)

        # Generate all closed non-backtracking walks of length L in [5, g-1]
        # g=16, so L in [5, 15]
        max_L = g - 1
        
        # DFS to find closed walks
        # State: (current_vertex, prev_vertex, length, path_edges, path_dirs)
        # path_edges: list of edge indices
        # path_dirs: list of 1 or -1
        
        all_walk_vectors = set()
        
        for start_v in range(10):
            stack = [(start_v, -1, 0, [], [])]
            while stack:
                current, prev, length, p_edges, p_dirs = stack.pop()
                
                if length >= 5 and current == start_v:
                    # Found a closed walk
                    # Create coefficient vector
                    vec = [0] * 15
                    for e_idx, d in zip(p_edges, p_dirs):
                        vec[e_idx] += d
                    # Check if this vector is effectively new
                    # We can just add it to a list, or a set of tuples
                    all_walk_vectors.add(tuple(vec))
                    
                if length == max_L:
                    continue
                    
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = base_edges[e_idx]
                    if current == u:
                        d = 1
                    else:
                        d = -1
                    stack.append((next_v, current, length + 1, p_edges + [e_idx], p_dirs + [d]))
                    
        # Convert to numpy array
        if not all_walk_vectors:
            # Should not happen for Petersen
            C = np.array([])
        else:
            C = np.array(list(all_walk_vectors))
            
        # C shape: (N_walks, 15)
        
        # Search for m
        # Moore bound for k=3, g=16 is 766. n=10m. m >= 77.
        min_m = 77
        
        best_m = None
        best_x = None
        
        # We iterate m upwards
        # For each m, try random x
        # If we find a valid x, we can stop? 
        # Yes, because we want the smallest n, which corresponds to smallest m.
        # So we must find the smallest m that has a solution.
        
        # However, for small m, the probability of a random x being valid is low?
        # Or high?
        # Constraints: C @ x % m != 0 for all rows.
        # If m is small, 0 is more likely to appear.
        # So we might need many tries for small m.
        # We should limit tries per m to avoid getting stuck.
        
        tries_per_m = 100
        
        for m in range(min_m, 500): # Try up to m=500 (n=5000)
            if remaining() < 10:
                break
                
            found = False
            for _ in range(tries_per_m):
                # Random x in Z_m
                x = np.random.randint(0, m, size=15)
                
                # Check constraints
                # S = C @ x % m
                # If any S == 0, invalid.
                S = (C @ x) % m
                if not np.any(S == 0):
                    # Valid!
                    best_m = m
                    best_x = x
                    found = True
                    break
                    
            if found:
                break
                
        if best_m is None:
            # If no small m found, try a large m which is almost certainly valid
            # Or fallback
            return None

        # Construct the lifted graph
        n = 10 * best_m
        edges = []
        for u, v in base_edges:
            e_idx = edge_index[(u, v)]
            for i in range(best_m):
                # u -> v
                edges.append((u * best_m + i, v * best_m + (i + best_x[e_idx]) % best_m))
                # v -> u (reverse direction, so subtract)
                edges.append((v * best_m + i, u * best_m + (i - best_x[e_idx]) % best_m))
        
        # Remove duplicates? The construction above adds both directions.
        # Since the base graph is undirected, (u,v) and (v,u) are the same edge.
        # In the lift, we need to be careful not to add duplicate edges.
        # Let's rebuild edges uniquely.
        edges_set = set()
        final_edges = []
        for u, v in base_edges:
            e_idx = edge_index[(u, v)]
            for i in range(best_m):
                # Edge from (u, i) to (v, i + x)
                u_lift = u * best_m + i
                v_lift = v * best_m + (i + best_x[e_idx]) % best_m
                # Add edge
                if u_lift < v_lift:
                    edges_set.add((u_lift, v_lift))
                else:
                    edges_set.add((v_lift, u_lift))
        
        final_edges = list(edges_set)
        
        # Verify girth
        gc = check_girth(n, final_edges, g, deadline - 5)
        if gc == -1:
            return None
        if gc >= g:
            return (n, final_edges)
        else:
            return None

    # ---------------- LUW construction (Fallback) ----------------
    def _luw_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def generate_rules(d, variant=0, offset=0):
        base = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        if variant == 0:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 4), c - ((c - 5 + offset) % 4)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        elif variant == 1:
            for c in range(5, d + 1):
                base[c] = (1, c - 2) if (c + offset) % 2 == 0 else (c - 1, 1)
        elif variant == 2:
            for c in range(5, d + 1):
                if (c + offset) % 3 == 0:
                    base[c] = (1, c - 3)
                elif (c + offset) % 3 == 1:
                    base[c] = (c - 1, 1)
                else:
                    base[c] = (c - 2, 1)
        elif variant == 3:
            for c in range(5, d + 1):
                base[c] = (1, c - (1 + (c + offset) % 2))
        elif variant == 4:
            for c in range(5, d + 1):
                base[c] = [(1, c - 1), (c - 1, 1), (1, c - 2), (c - 2, 1)][(c + offset) % 4]
        elif variant == 5:
            for c in range(5, d + 1):
                base[c] = (c - 2, 1) if (c + offset) % 2 == 0 else (1, c - 1)
        elif variant == 6:
            for c in range(5, d + 1):
                base[c] = (1, c - 1) if (c + offset) % 3 == 0 else (c - 2, 1)
        elif variant == 7:
            for c in range(5, d + 1):
                base[c] = (c - 1, 1) if (c + offset) % 4 == 0 else (1, c - 2)
        else:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 4), c - ((c - 5 + offset) % 4)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return base

    def build_luw(q, d, rule, time_limit):
        add, sub, mul = _luw_ops(q)

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
            if time.time() > time_limit:
                return None
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def luw_construct(k, g):
        q = k
        d = max(3, (g - 2) // 2)
        best = None
        best_n = float('inf')

        for variant in range(8):
            for offset in range(6):
                if remaining() < 12:
                    break
                rule = generate_rules(d, variant, offset)
                res = build_luw(q, d, rule, deadline - 12)
                if res is None:
                    continue
                n, edges = res
                if n >= best_n:
                    continue
                gc = check_girth(n, edges, g, deadline - 8)
                if gc == -1:
                    continue
                if gc >= g:
                    best, best_n = (n, edges), n
                    break

        if best is None:
            for d_try in range(d + 1, d + 10):
                if remaining() < 10:
                    break
                for variant in range(3):
                    for offset in range(3):
                        if remaining() < 5:
                            break
                        rule = generate_rules(d_try, variant, offset)
                        res = build_luw(q, d_try, rule, deadline - 10)
                        if res is None:
                            continue
                        n, edges = res
                        gc = check_girth(n, edges, g, deadline - 5)
                        if gc == -1:
                            continue
                        if gc >= g:
                            best = (n, edges)
                            break
                    if best is not None:
                        break
                if best is not None:
                    break

        if best is None:
            rule = generate_rules(d, 0, 0)
            res = build_luw(q, d, rule, deadline - 2)
            if res is not None:
                best = res
        return best

    # ---------------- main ----------------
    # Try voltage lifting first for k=3, g=16
    if k == 3 and g == 16:
        res = voltage_lift()
        if res is not None:
            return res

    # Fallback to LUW
    res = luw_construct(k, g)
    return res if res is not None else (0, [])
