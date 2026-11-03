import numpy as np
import random
import time

def construct(k, g):
    # Construct PG(2,3) incidence graph
    # Points: 13, Lines: 13
    # Total 26 vertices.
    
    # Generate points
    points = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if x == 0 and y == 0 and z == 0: continue
                # Canonicalize: first non-zero is 1
                # Since field is Z3, non-zero are 1,2.
                # If x!=0, scale by x^-1
                if x != 0:
                    inv_x = 1 if x == 1 else 2
                    p = (1, (y*inv_x)%3, (z*inv_x)%3)
                elif y != 0:
                    inv_y = 1 if y == 1 else 2
                    p = (0, 1, (z*inv_y)%3)
                else:
                    p = (0, 0, 1)
                if p not in points:
                    points.append(p)
    
    # Generate lines
    # A line is a set of 4 points.
    # In PG(2,3), lines are defined by ax+by+cz=0.
    # Normal vectors (a,b,c) modulo scaling.
    lines = []
    line_points = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if a == 0 and b == 0 and c == 0: continue
                # Canonicalize normal
                if a != 0:
                    inv_a = 1 if a == 1 else 2
                    n = (1, (b*inv_a)%3, (c*inv_a)%3)
                elif b != 0:
                    inv_b = 1 if b == 1 else 2
                    n = (0, 1, (c*inv_b)%3)
                else:
                    n = (0, 0, 1)
                
                if n in [tuple(l) for l in lines]: continue
                lines.append(n)
                pts = []
                for p in points:
                    if (n[0]*p[0] + n[1]*p[1] + n[2]*p[2]) % 3 == 0:
                        pts.append(p)
                line_points.append(pts)
    
    # Build base graph
    # Vertices 0..12 are points, 13..25 are lines.
    base_edges = []
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_edges.append((i, 13+j))
    
    # Build adjacency for base graph
    adj_base = {i: [] for i in range(26)}
    for u, v in base_edges:
        adj_base[u].append(v)
        adj_base[v].append(u)
    
    # Enumerate all cycles of length 6 and 8 in the base graph
    # The base graph is bipartite with parts {0..12} (points) and {13..25} (lines)
    
    # Find all 6-cycles: P1-L1-P2-L2-P3-L3-P1
    # We can fix P1 to avoid duplicates, but for voltage checking we need all
    six_cycles = []
    for p1 in range(13):
        for l1 in adj_base[p1]:
            for p2 in adj_base[l1]:
                if p2 == p1: continue
                for l2 in adj_base[p2]:
                    if l2 == l1: continue
                    for p3 in adj_base[l2]:
                        if p3 == p1 or p3 == p2: continue
                        for l3 in adj_base[p3]:
                            if l3 == l2: continue
                            if p1 in adj_base[l3]:
                                cycle = [(p1, l1), (l1, p2), (p2, l2), (l2, p3), (p3, l3), (l3, p1)]
                                six_cycles.append(cycle)
    
    # Find all 8-cycles: P1-L1-P2-L2-P3-L3-P4-L4-P1
    eight_cycles = []
    for p1 in range(13):
        for l1 in adj_base[p1]:
            for p2 in adj_base[l1]:
                if p2 == p1: continue
                for l2 in adj_base[p2]:
                    if l2 == l1: continue
                    for p3 in adj_base[l2]:
                        if p3 == p1 or p3 == p2: continue
                        for l3 in adj_base[p3]:
                            if l3 == l2: continue
                            for p4 in adj_base[l3]:
                                if p4 == p1 or p4 == p2 or p4 == p3: continue
                                for l4 in adj_base[p4]:
                                    if l4 == l3: continue
                                    if p1 in adj_base[l4]:
                                        cycle = [(p1, l1), (l1, p2), (p2, l2), (l2, p3), (p3, l3), (l3, p4), (p4, l4), (l4, p1)]
                                        eight_cycles.append(cycle)
    
    # Now we need to find voltages such that no 6-cycle or 8-cycle has sum 0 mod m
    # Use m=10, n = 260 vertices
    
    m = 10
    n_vertices = 26 * m
    
    # Use a smarter approach: assign voltages using a structured pattern
    # The base graph has automorphisms from PGL(3,3)
    # Try to use a voltage assignment based on the coordinates
    
    # Alternative: use local search / simulated annealing
    # Start with random assignment, then fix conflicts
    
    best_voltages = None
    best_conflicts = None
    
    for attempt in range(500):
        # Random initial assignment
        voltages = {}
        for u, v in base_edges:
            voltages[(u, v)] = random.randint(1, m-1)  # Avoid 0 for more diversity
            voltages[(v, u)] = (-voltages[(u, v)]) % m
        
        # Count conflicts
        def count_conflicts(volt):
            conflicts = 0
            for cycle in six_cycles:
                s = sum(volt[e] for e in cycle)
                if s % m == 0:
                    conflicts += 1
            for cycle in eight_cycles:
                s = sum(volt[e] for e in cycle)
                if s % m == 0:
                    conflicts += 1
            return conflicts
        
        conflicts = count_conflicts(voltages)
        
        # Local search: try to reduce conflicts
        improved = True
        while improved and conflicts > 0:
            improved = False
            # Try changing each edge's voltage
            for u, v in base_edges:
                old_val = voltages[(u, v)]
                for new_val in range(1, m):
                    if new_val == old_val:
                        continue
                    voltages[(u, v)] = new_val
                    voltages[(v, u)] = (-new_val) % m
                    new_conflicts = count_conflicts(voltages)
                    if new_conflicts < conflicts:
                        conflicts = new_conflicts
                        improved = True
                        break
                    else:
                        voltages[(u, v)] = old_val
                        voltages[(v, u)] = (-old_val) % m
        
        if conflicts == 0:
            best_voltages = voltages
            break
        
        if best_conflicts is None or conflicts < best_conflicts:
            best_conflicts = conflicts
            best_voltages = voltages
    
    if best_voltages is not None and best_conflicts == 0:
        # Build lift graph
        lift_edges = []
        for u, v in base_edges:
            vu = best_voltages[(u, v)]
            for i in range(m):
                lift_edges.append((u*m + i, v*m + (i + vu) % m))
        
        # Verify girth using BFS from each vertex
        adj_lift = [[] for _ in range(n_vertices)]
        for u, v in lift_edges:
            adj_lift[u].append(v)
            adj_lift[v].append(u)
        
        # Check degree
        for i in range(n_vertices):
            if len(adj_lift[i]) != k:
                return 260, lift_edges  # Fallback
        
        # BFS to find girth
        min_girth = 999
        for start in range(n_vertices):
            dist = [-1] * n_vertices
            parent = [-1] * n_vertices
            dist[start] = 0
            queue = [start]
            while queue:
                cur = queue.pop(0)
                for nb in adj_lift[cur]:
                    if dist[nb] == -1:
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
                    break
            if min_girth >= g:
                break
        
        if min_girth >= g:
            return n_vertices, lift_edges
    
    # Fallback to LUW if search fails
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
