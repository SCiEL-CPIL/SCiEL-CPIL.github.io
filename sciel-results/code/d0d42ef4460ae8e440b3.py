import numpy as np
import random
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # Target: k=4, g=10
    # Strategy: Voltage lift of Moebius-Kantor graph (16 vertices, 4-regular, girth 8)
    # If we find a lift with M layers such that N = 16*M < N_seed, we win.
    # Known best is 320. 16 * 20 = 320.
    # We try M in [17, 18, 19, 20, 21, 22, 23].
    
    if k == 4 and g == 10:
        # Define Moebius-Kantor Graph
        # Vertices 0..15
        # (i, j) -> i*4 + j
        adj_mk = {i: [] for i in range(16)}
        edges_mk = []
        for i in range(4):
            for j in range(4):
                u = i * 4 + j
                # Neighbors
                v1 = ((i + 1) % 4) * 4 + j
                v2 = i * 4 + ((j + 1) % 4)
                v3 = ((i + 1) % 4) * 4 + ((j + 2) % 4)
                v4 = ((i + 2) % 4) * 4 + ((j + 1) % 4)
                
                for v in [v1, v2, v3, v4]:
                    if u < v:
                        edges_mk.append((u, v))
                    adj_mk[u].append(v)
                    if v not in adj_mk[u]:
                        adj_mk[v].append(u)
        
        # Map edge to index for voltage array
        edge_index = {}
        for idx, (u, v) in enumerate(edges_mk):
            edge_index[(u, v)] = idx
            edge_index[(v, u)] = idx
        
        # Find all 8-cycles
        cycles_8 = []
        # Also find 6-cycles just in case (should be none in base, but check lift logic)
        # Actually, if base has no 6-cycles, lift has no 6-cycles?
        # Yes, simple cycle in lift projects to simple cycle in base or closed walk.
        # If base girth is 8, no simple cycles < 8.
        # So we only need to check 8-cycles.
        
        for start in range(16):
            # DFS to find cycles of length 8 starting and ending at start
            # To avoid duplicates, we only count cycles where start is the smallest vertex?
            # No, we need all cycles to constrain all of them.
            # But we can just find all unique sets of edges.
            
            # Simple DFS
            path = [start]
            visited = {start}
            
            def dfs(cur, depth, path, visited):
                if depth == 8:
                    if len(adj_mk[cur]) > 0:
                        # Check if last is connected to start
                        if start in adj_mk[cur]:
                            # Found a cycle
                            # Normalize: sort edges
                            cycle_edges = set()
                            for i in range(7):
                                u, v = path[i], path[i+1]
                                cycle_edges.add(tuple(sorted((u, v))))
                            # Add last edge
                            u, v = path[7], start
                            cycle_edges.add(tuple(sorted((u, v))))
                            # Check if we already have this cycle
                            # To avoid duplicates, we can store sorted tuple of edges
                            key = tuple(sorted(cycle_edges))
                            if key not in seen_cycles:
                                seen_cycles.add(key)
                                # Store list of edge indices
                                cycle_indices = [edge_index[e] for e in cycle_edges]
                                cycles_8.append(cycle_indices)
                    return
                
                for nb in adj_mk[cur]:
                    if nb == path[0] and depth < 7:
                        continue # Don't close early
                    if nb in visited:
                        continue
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, depth + 1, path, visited)
                    path.pop()
                    visited.remove(nb)
                    
            seen_cycles = set()
            dfs(start, 1, path, visited)
            
        # Sort cycles by length (all 8)
        # Randomize order for search
        random.shuffle(cycles_8)
        
        best_n = float('inf')
        best_edges = None
        
        # Try different M
        # 16 * 17 = 272
        # 16 * 19 = 304
        # 16 * 20 = 320
        # 16 * 23 = 368
        for M in [17, 19, 20, 23]:
            if time.time() > deadline:
                break
            
            n_vertices = 16 * M
            if n_vertices >= best_n:
                continue
                
            # Search for voltages
            # Cost: number of cycles with sum 0 mod M
            found = False
            best_voltages = None
            
            for attempt in range(200):
                if time.time() > deadline:
                    break
                    
                voltages = [random.randint(0, M - 1) for _ in range(len(edges_mk))]
                
                def compute_cost(v):
                    cost = 0
                    for cycle in cycles_8:
                        s = 0
                        for idx in cycle:
                            s += v[idx]
                        if s % M == 0:
                            cost += 1
                    return cost
                
                cost = compute_cost(voltages)
                
                # Hill Climbing
                stuck = 0
                for step in range(500):
                    if time.time() > deadline:
                        break
                        
                    if cost == 0:
                        found = True
                        best_voltages = voltages[:]
                        break
                        
                    # Pick random edge
                    edge_idx = random.randint(0, len(edges_mk) - 1)
                    old_val = voltages[edge_idx]
                    new_val = random.randint(0, M - 1)
                    while new_val == old_val:
                        new_val = random.randint(0, M - 1)
                    voltages[edge_idx] = new_val
                    
                    new_cost = compute_cost(voltages)
                    
                    if new_cost <= cost:
                        cost = new_cost
                        stuck = 0
                    else:
                        voltages[edge_idx] = old_val
                        stuck += 1
                        
                    if stuck > 100:
                        break
                
                if found:
                    break
                
            if found:
                # Build lift
                lift_edges = []
                for idx, (u, v) in enumerate(edges_mk):
                    vu = best_voltages[idx]
                    for layer in range(M):
                        u_lift = u * M + layer
                        v_lift = v * M + (layer + vu) % M
                        lift_edges.append((u_lift, v_lift))
                
                # Verify Girth
                adj_lift = {i: [] for i in range(n_vertices)}
                for u, v in lift_edges:
                    adj_lift[u].append(v)
                    adj_lift[v].append(u)
                
                min_girth = 999
                valid = True
                for start in range(n_vertices):
                    if time.time() > deadline:
                        valid = False
                        break
                    dist = {start: 0}
                    parent = {start: -1}
                    queue = [start]
                    while queue:
                        cur = queue.pop(0)
                        for nb in adj_lift[cur]:
                            if nb not in dist:
                                dist[nb] = dist[cur] + 1
                                parent[nb] = cur
                                queue.append(nb)
                            elif parent[nb] != cur and parent[cur] != nb:
                                cycle_len = dist[nb] + dist[cur] + 1
                                if cycle_len < min_girth:
                                    min_girth = cycle_len
                    if min_girth < g:
                        valid = False
                        break
                        
                if valid and min_girth >= g:
                    best_n = n_vertices
                    best_edges = lift_edges
                    # We found a valid graph. Since we try smaller M first, we can break?
                    # No, maybe a smaller M works? We iterate M in increasing order.
                    # So the first found is the smallest N.
                    return n_vertices, lift_edges
                    
        # Fallback: LUW
    # Fallback construction for general k, g
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
        mul_op = lambda x, y: table[x][y]
    else:
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
        mul_op = lambda x, y: x * y % q
    
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
                if side == 0:
                    y[c] = add_op(x[c], mul_op(y[a], x[b]))
                else:
                    y[c] = sub_op(x[c], mul_op(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
