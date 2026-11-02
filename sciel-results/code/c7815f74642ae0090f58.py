import time
import random
import sys

def construct(k, g):
    # Voltage graph lift of Petersen graph over Z_32
    def voltage_lift_petersen():
        # Petersen graph: 10 vertices, 4-regular, girth 5
        # Vertices: 0-4 (outer pentagon), 5-9 (inner star)
        petersen_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer pentagon
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner star
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),  # spokes
        ]
        
        # Assign voltages to each edge from Z_32
        # We need to ensure no non-backtracking closed walk in Petersen
        # has total voltage 0 mod 32
        # Petersen has girth 5, so we need to check walks of length 5, 6, 7, 8, 9
        # that are non-backtracking and closed
        
        # Try different voltage assignments
        # Each edge gets a voltage in {1, 2, ..., 31}
        
        # A systematic approach: use powers of a primitive root
        # For Z_32, use voltage 3 (which has order 5 in Z_32 since 3^5 = 243 = 19 mod 32)
        # Actually let's try voltage 5: 5^1=5, 5^2=25, 5^3=125=29, 5^4=145=17, 5^5=85=21
        
        # Let's try a simple assignment and verify
        # Edge order: 0-1, 1-2, 2-3, 3-4, 4-0, 5-7, 7-9, 9-6, 6-8, 8-5, 0-5, 1-6, 2-7, 3-8, 4-9
        
        # Try voltages based on edge position
        base_voltages = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        
        # Verify: check all non-backtracking closed walks of length < 10 in Petersen
        # and ensure none have total voltage 0 mod 32
        
        def get_nonbacktracking_closed_walks(p_edges, max_len=9):
            """Find all non-backtracking closed walks up to max_len in Petersen graph"""
            adj = [[] for _ in range(10)]
            for u, v in p_edges:
                adj[u].append(v)
                adj[v].append(u)
            
            walks = []
            for start in range(10):
                # DFS with non-backtracking constraint
                stack = [(start, start, 0, [])]  # (current, prev, length, path)
                while stack:
                    curr, prev, length, path = stack.pop()
                    if length == max_len:
                        continue
                    for nxt in adj[curr]:
                        if nxt == prev and len(path) > 0:
                            continue  # non-backtracking
                        new_path = path + [nxt]
                        new_length = length + 1
                        if new_length >= 5 and nxt == start:  # only consider closed walks of length >= 5
                            # Check it's truly non-backtracking
                            is_nb = True
                            for i in range(len(new_path) - 1):
                                if new_path[i] == new_path[i + 1]:
                                    is_nb = False
                                    break
                                if i > 0 and new_path[i] == new_path[i - 1] and new_path[i + 1] != new_path[i - 2]:
                                    pass  # this is fine
                            if is_nb:
                                walks.append(new_path)
                        if new_length < max_len:
                            stack.append((nxt, curr, new_length, new_path))
            return walks
        
        # For each voltage assignment, check if any non-backtracking closed walk has voltage 0
        def check_voltage_assignment(voltages):
            """Check if voltage assignment works (no zero-voltage non-backtracking closed walks)"""
            # Build edge voltage map
            edge_vol = {}
            for i, (u, v) in enumerate(petersen_edges):
                edge_vol[(u, v)] = voltages[i]
                edge_vol[(v, u)] = (-voltages[i]) % 32
            
            # Find all non-backtracking closed walks of length 5-9
            adj = [[] for _ in range(10)]
            for u, v in petersen_edges:
                adj[u].append(v)
                adj[v].append(u)
            
            for start in range(10):
                stack = [(start, -1, 0, 0, [])]  # (curr, prev, length, total_vol, path)
                while stack:
                    curr, prev, length, vol, path = stack.pop()
                    if length == 9:
                        continue
                    for nxt in adj[curr]:
                        if nxt == prev and length > 0:
                            continue
                        # Get voltage for this edge
                        e_vol = edge_vol.get((curr, nxt), None)
                        if e_vol is None:
                            continue
                        new_vol = (vol + e_vol) % 32
                        new_length = length + 1
                        if new_length >= 5 and nxt == start:
                            if new_vol == 0:
                                return False  # Found a zero-voltage closed walk
                        if new_length < 9:
                            stack.append((nxt, curr, new_length, new_vol, path + [nxt]))
            return True
        
        # Try different voltage assignments
        best_voltages = None
        for trial in range(100):
            if trial == 0:
                voltages = base_voltages
            else:
                voltages = [random.randint(1, 31) for _ in range(15)]
            
            if check_voltage_assignment(voltages):
                best_voltages = voltages
                break
        
        if best_voltages is None:
            return None
        
        # Build the lifted graph
        n_lift = 32
        edges = []
        for i, (u, v) in enumerate(petersen_edges):
            vol = best_voltages[i]
            for coset in range(n_lift):
                u_lift = u * n_lift + coset
                v_lift = v * n_lift + ((coset + vol) % n_lift)
                edges.append((u_lift, v_lift))
        
        return 10 * n_lift, edges
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph (large but valid)
    def luw_construct(k, g, d_override=None):
        q = k
        if d_override is not None:
            d = d_override
        else:
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
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

    # Moore bound calculation (using the code's definition)
    D = (g - 2) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(D))

    deadline = time.time() + 110
    best_n = float('inf')
    best_edges = None

    def has_girth_at_least(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if (v, u) in edge_set:
                return False
            edge_set.add((u, v))
        for u in range(n):
            if len(adj[u]) != k:
                return False
        # Check girth using BFS from each vertex up to depth g//2
        half_g = g // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def random_regular_graph(n, k):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        edges = []
        valid = True
        seen = set()
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in seen:
                valid = False
                break
            seen.add(key)
            edges.append((u, v))
        if valid:
            return edges
        return None

    # Try voltage lift of Petersen graph first (target: 320 vertices)
    if k == 4 and g == 10:
        result = voltage_lift_petersen()
        if result is not None:
            n_volt, edges_volt = result
            if has_girth_at_least(n_volt, edges_volt, g):
                return n_volt, edges_volt
    
    # Try LUW with smaller d first - this is our guaranteed valid solution
    for d_try in [4, 5]:
        if time.time() > deadline:
            break
        n_luw, edges_luw = luw_construct(k, g, d_try)
        if has_girth_at_least(n_luw, edges_luw, g):
            if n_luw < best_n:
                best_n = n_luw
                best_edges = edges_luw

    if best_n != float('inf'):
        return best_n, best_edges
    
    # Fallback to standard LUW
    return luw_construct(k, g)
