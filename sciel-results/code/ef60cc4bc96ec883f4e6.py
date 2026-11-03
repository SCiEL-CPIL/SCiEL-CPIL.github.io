import numpy as np
import random
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k == 4 and g == 10:
        # Build PG(2,3) incidence graph
        points = []
        seen_points = set()
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    if x == 0 and y == 0 and z == 0: continue
                    if x != 0:
                        inv_x = 1 if x == 1 else 2
                        p = (1, (y * inv_x) % 3, (z * inv_x) % 3)
                    elif y != 0:
                        inv_y = 1 if y == 1 else 2
                        p = (0, 1, (z * inv_y) % 3)
                    else:
                        p = (0, 0, 1)
                    if p not in seen_points:
                        seen_points.add(p)
                        points.append(p)
        
        lines = []
        line_points = []
        seen_lines = set()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    if a == 0 and b == 0 and c == 0: continue
                    if a != 0:
                        inv_a = 1 if a == 1 else 2
                        n = (1, (b * inv_a) % 3, (c * inv_a) % 3)
                    elif b != 0:
                        inv_b = 1 if b == 1 else 2
                        n = (0, 1, (c * inv_b) % 3)
                    else:
                        n = (0, 0, 1)
                    
                    if n in seen_lines: continue
                    seen_lines.add(n)
                    lines.append(n)
                    pts = []
                    for p in points:
                        if (n[0] * p[0] + n[1] * p[1] + n[2] * p[2]) % 3 == 0:
                            pts.append(p)
                    line_points.append(pts)
        
        base_edges = []
        adj_base = {i: [] for i in range(26)}
        edge_index = {}
        for i, p in enumerate(points):
            for j, lp in enumerate(line_points):
                if p in lp:
                    u, v = i, 13 + j
                    base_edges.append((u, v))
                    adj_base[u].append(v)
                    adj_base[v].append(u)
                    edge_index[(u, v)] = len(base_edges) - 1
                    edge_index[(v, u)] = len(base_edges) - 1
        
        # Enumerate 6-cycles and 8-cycles and map to edge indices
        cycles_6_idx = []
        for p1 in range(13):
            for l1 in adj_base[p1]:
                for p2 in adj_base[l1]:
                    if p2 == p1: continue
                    for l2 in adj_base[p2]:
                        if l2 == l1: continue
                        for p3 in adj_base[l2]:
                            if p3 in (p1, p2): continue
                            for l3 in adj_base[p3]:
                                if l3 == l2: continue
                                if p1 in adj_base[l3]:
                                    # Cycle: p1-l1-p2-l2-p3-l3-p1
                                    idxs = [
                                        edge_index[(p1, l1)],
                                        edge_index[(l1, p2)],
                                        edge_index[(p2, l2)],
                                        edge_index[(l2, p3)],
                                        edge_index[(p3, l3)],
                                        edge_index[(l3, p1)]
                                    ]
                                    cycles_6_idx.append(idxs)
        
        cycles_8_idx = []
        for p1 in range(13):
            for l1 in adj_base[p1]:
                for p2 in adj_base[l1]:
                    if p2 == p1: continue
                    for l2 in adj_base[p2]:
                        if l2 == l1: continue
                        for p3 in adj_base[l2]:
                            if p3 in (p1, p2): continue
                            for l3 in adj_base[p3]:
                                if l3 == l2: continue
                                for p4 in adj_base[l3]:
                                    if p4 in (p1, p2, p3): continue
                                    for l4 in adj_base[p4]:
                                        if l4 == l3: continue
                                        if p1 in adj_base[l4]:
                                            if l4 != l1:
                                                idxs = [
                                                    edge_index[(p1, l1)],
                                                    edge_index[(l1, p2)],
                                                    edge_index[(p2, l2)],
                                                    edge_index[(l2, p3)],
                                                    edge_index[(p3, l3)],
                                                    edge_index[(l3, p4)],
                                                    edge_index[(p4, l4)],
                                                    edge_index[(l4, p1)]
                                                ]
                                                cycles_8_idx.append(idxs)
        
        # Convert to numpy arrays for fast summing
        # Each cycle is a list of edge indices
        # We want to sum voltages[edge] for each cycle
        # V is a vector of voltages
        # Sums = V[Cycles] . ones? No.
        # If Cycles is a 2D array of shape (N_cycles, L), then V[Cycles] gives (N_cycles, L)
        # Summing along axis 1 gives (N_cycles,)
        
        cycles_6_np = np.array(cycles_6_idx) if cycles_6_idx else np.empty((0, 6), dtype=int)
        cycles_8_np = np.array(cycles_8_idx) if cycles_8_idx else np.empty((0, 8), dtype=int)
        
        def compute_cost(voltages_np, m):
            # voltages_np is shape (E,)
            # cycles_6_np is shape (C6, 6)
            if len(cycles_6_np) > 0:
                sums_6 = voltages_np[cycles_6_np].sum(axis=1)
                cost_6 = np.sum(sums_6 % m == 0)
            else:
                cost_6 = 0
            if len(cycles_8_np) > 0:
                sums_8 = voltages_np[cycles_8_np].sum(axis=1)
                cost_8 = np.sum(sums_8 % m == 0)
            else:
                cost_8 = 0
            return cost_6 + cost_8

        best_cost = float('inf')
        best_voltages = None
        
        # Try m=10 first
        for m in [10, 11, 12]:
            if time.time() > deadline: break
            
            n_vertices = 26 * m
            # Moore bound check
            # 1 + 4 * (1 + 4 + 4^2 + 4^3) = 1 + 4 * (1+4+16+64) = 1 + 4*85 = 341?
            # Wait, Moore bound for girth g=10 (diameter 4? No, radius 4? g/2 = 5).
            # Bound: 1 + k * (1 + (k-1) + ... + (k-1)^(g/2 - 1))
            # g=10, g/2=5.
            # 1 + 4 * (1 + 3 + 9 + 27 + 81) = 1 + 4 * 121 = 485.
            # So 260 is well below the Moore bound?
            # Wait, the Moore bound is a LOWER bound on n.
            # If n < Moore bound, it's impossible.
            # Let's re-calculate Moore bound for (4,10).
            # n >= 1 + k * sum_{i=0}^{g/2 - 1} (k-1)^i
            # g=10, g/2=5. Sum i=0 to 4 of 3^i = 1+3+9+27+81 = 121.
            # n >= 1 + 4*121 = 485.
            # So n=260 is IMPOSSIBLE.
            # My previous assumption that 260 might work was wrong.
            # The Moore bound for (4,10) is 485.
            # So the smallest possible graph has at least 485 vertices.
            # The current LUW graph is likely around 400-500.
            # If LUW is < 485, it violates the Moore bound?
            # No, the Moore bound is for girth >= g.
            # If LUW produces a graph with girth >= 10, it MUST have n >= 485.
            # If the current program scores 1.0, and it uses LUW, then n_LUW >= 485.
            # Let's check the LUW size.
            # If n_LUW is, say, 512.
            # Can we do better than 512?
            # The best known is 320?
            # Wait, the prompt says: "The smallest known such graph has 320 vertices".
            # This contradicts the Moore bound calculation.
            # Let me re-read the Moore bound formula.
            # For a d-regular graph of girth g, n >= 1 + d * sum_{i=0}^{g/2-1} (d-1)^i.
            # For d=4, g=10.
            # 1 + 4 * (1 + 3 + 9 + 27 + 81) = 485.
            # How can there be a 320 vertex graph?
            # Maybe the girth is not 10?
            # "Small 4-regular graph of girth 10".
            # Maybe I am calculating the bound wrong.
            # Is the graph bipartite?
            # If the graph is bipartite, the bound is different?
            # No, the Moore bound applies to all graphs.
            # Let's check the bound for (3,10). 1 + 3*(1+2+4+8+16) = 1+3*31 = 94.
            # (3,10) cage is 70 vertices? No, (3,10) cage is 70?
            # The (3,10) cage is the Robertson graph? No, that's (3,10) with 70 vertices?
            # Actually, the (3,10) cage has 70 vertices.
            # Moore bound for (3,10) is 94?
            # 1 + 3*(1+2+4+8+16) = 94.
            # If the cage has 70 vertices, it violates the bound?
            # Ah, the bound is $1 + k \sum_{i=0}^{g/2 - 1} (k-1)^i$.
            # For g=10, g/2 = 5.
            # Terms: i=0,1,2,3,4.
            # Sum = 1 + 3 + 9 + 27 + 81 = 121.
            # 1 + 4*121 = 485.
            # Why is the known cage 320?
            # Maybe the known cage is not 4-regular?
            # "Small 4-regular graph of girth 10".
            # Perhaps the Moore bound is for girth > g?
            # No, girth >= g.
            # Let's check online or recall.
            # The (4,10) cage is not known?
            # "The smallest known such graph has 320 vertices".
            # This implies n=320 is possible.
            # This implies my Moore bound calculation is wrong.
            # Let's re-evaluate.
            # Maybe the graph is not simple? "simple (no loops or repeated edges)".
            # Maybe I am summing up to g/2 - 1?
            # For g=10, radius is 4?
            # A tree of radius 4 has 1 + 4 + 4*3 + 4*3^2 + 4*3^3 = 1 + 4 + 12 + 36 + 108 = 161 vertices.
            # This is the number of vertices at distance <= 4 from a root.
            # In a graph of girth 10, the ball of radius 4 is a tree.
            # So n >= 2 * (1 + 4 + 12 + 36) - 1? No.
            # The ball of radius 4 has 1 + 4 + 12 + 36 + 108 = 161 vertices.
            # Since the graph is 4-regular, the number of vertices is at least the size of the ball?
            # No, the ball is a subset.
            # If the graph is vertex-transitive, n >= 161?
            # But the Moore bound is usually tighter.
            # The standard Moore bound for (k, g) is:
            # If g is even, n >= 1 + k * sum_{i=0}^{g/2 - 1} (k-1)^i.
            # For k=4, g=10: 1 + 4 * (1 + 3 + 9 + 27 + 81) = 485.
            # If the known cage is 320, then either:
            # 1. The known cage is not 4-regular.
            # 2. The known cage has girth < 10.
            # 3. My formula is wrong.
            # Let's check (3, 8). Bound: 1 + 3*(1+2+4+8) = 1+3*15 = 46.
            # (3,8) cage is 30 vertices? No, (3,8) cage is 30?
            # The (3,8) cage is the Tutte-Coxeter graph? No, that's (3,12).
            # The (3,8) cage has 30 vertices?
            # If n=30 < 46, then the bound is wrong.
            # Ah, the bound is $1 + k \sum_{i=0}^{g/2 - 1} (k-1)^i$ is for **bipartite** graphs?
            # No, that's the standard bound.
            # Wait, for g=8, g/2=4. Sum i=0..3. 1+2+4+8=15. 1+3*15=46.
            # If the cage is 30, it violates the bound.
            # This suggests the bound is not 46.
            # Let's check the bound for (3, 8) again.
            # Maybe the sum is up to g/2 - 1?
            # For g=8, radius 3?
            # Ball of radius 3: 1 + 3 + 6 + 12 = 22.
            # If n >= 22, then 30 is valid.
            # The formula $1 + k \sum_{i=0}^{g/2 - 1} (k-1)^i$ gives 46.
            # Why?
            # Because in a non-bipartite graph, you can have odd cycles.
            # But girth is 8, so no cycles < 8.
            # The ball of radius 4 is a tree.
            # Size of ball of radius 4: 1 + 3 + 6 + 12 + 24 = 46.
            # So n >= 46.
            # If the (3,8) cage is 30, it must have girth < 8?
            # Or it's not 3-regular?
            # Actually, the (3,8) cage is the **Tutte 8-cage**? No.
            # The (3,8) cage has 30 vertices?
            # Let's assume the prompt is correct: "smallest known ... 320 vertices".
            # And "best lower bound is 243".
            # 243 is close to my Moore bound 485? No.
            # 243 is $3^5$.
            # Maybe the lower bound is derived differently.
            # Regardless, the task is to minimize n.
            # If the current program returns the LUW graph, and LUW is > 320, we can improve.
            # If LUW is < 320, we might be at the optimum or near it.
            # Given the score is 1.0, and the seed is 1.0, the current program is likely the seed.
            # The seed is the LUW construction.
            # We need to find a smaller graph.
            # If the voltage lift of PG(2,3) with m=10 gives 260, and 260 < 320, and 260 >= 243 (lower bound), then 260 is a valid candidate IF the girth is 10.
            # But I calculated the Moore bound as 485.
            # If 260 < 485, it's impossible.
            # So the voltage lift CANNOT have girth 10.
            # So the voltage lift approach will fail to find a valid graph with girth 10.
            # It will find a graph with girth < 10.
            # So we must rely on the LUW construction or a better one.
            
            # What if we use a different base graph?
            # Or a different lift?
            # If the Moore bound is indeed 485, then n must be >= 485.
            # If the known cage is 320, then the Moore bound is NOT 485.
            # Let's trust the prompt's "lower bound 243".
            # This implies n=260 is possible.
            # So my Moore bound calculation is wrong for this context.
            # Perhaps the graph is not required to be connected? "Construct a 4-regular graph". Usually implies connected.
            # If disconnected, we could take two (2,10) cages? No, degree 4.
            
            # Let's assume n=260 is possible.
            # Then the voltage lift search is the way to go.
            # The previous code failed likely due to performance or local minima.
            # I will optimize the search.

            # Reset best_cost for this m
            best_cost_m = float('inf')
            best_voltages_m = None
            
            for attempt in range(100):
                if time.time() > deadline: break
                
                voltages = np.random.randint(0, m, size=len(base_edges))
                cost = compute_cost(voltages, m)
                
                # Simulated Annealing
                T = 1.0
                for step in range(1000):
                    if time.time() > deadline: break
                    if cost == 0:
                        best_cost_m = 0
                        best_voltages_m = voltages.copy()
                        break
                    
                    # Pick random edge
                    idx = random.randint(0, len(base_edges) - 1)
                    old_val = voltages[idx]
                    new_val = random.randint(0, m - 1)
                    while new_val == old_val:
                        new_val = random.randint(0, m - 1)
                    voltages[idx] = new_val
                    new_cost = compute_cost(voltages, m)
                    
                    delta = new_cost - cost
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        cost = new_cost
                    else:
                        voltages[idx] = old_val
                    
                    T *= 0.99
                    
                    if cost == 0:
                        best_cost_m = 0
                        best_voltages_m = voltages.copy()
                        break
                
                if cost < best_cost_m:
                    best_cost_m = cost
                    best_voltages_m = voltages.copy()
                
                if best_cost_m == 0:
                    break
            
            if best_voltages_m is not None and best_cost_m == 0:
                # Build lift
                lift_edges = []
                for i, (u, v) in enumerate(base_edges):
                    vu = best_voltages_m[i]
                    for layer in range(m):
                        lift_edges.append((u * m + layer, v * m + (layer + vu) % m))
                
                # Verify girth
                adj_lift = {i: [] for i in range(n_vertices)}
                for u, v in lift_edges:
                    adj_lift[u].append(v)
                    adj_lift[v].append(u)
                
                min_girth = 999
                for start in range(n_vertices):
                    if time.time() > deadline: break
                    dist = {start: 0}
                    parent = {start: None}
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
                    if min_girth >= g:
                        break
                
                if min_girth >= g:
                    return n_vertices, lift_edges

    # Fallback LUW
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
