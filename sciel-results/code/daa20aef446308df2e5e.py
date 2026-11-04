import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 140.0
    
    # Fallback: LUW Construction
    def luw_construct():
        q = k
        d = 3
        while True:
            girth_est = d + 5 if d % 2 == 1 else d + 4
            if girth_est >= g:
                break
            d += 1
        
        if q == 4:
            mul_table = [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2]
            ]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return mul_table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return (x * y) % q

        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        rule_a[2], rule_b[2] = 1, 1
        rule_a[3], rule_b[3] = 2, 1
        rule_a[4], rule_b[4] = 1, 2
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                a, b = 1, c0 - 2
            elif r == 1:
                a, b = c0 - 1, 1
            elif r == 2:
                a, b = c0, 1
            else:
                a, b = 1, c0 + 1
            rule_a[c] = a
            rule_b[c] = b

        mul_table = [[mul(i, j) for j in range(q)] for i in range(q)]
        add_table = [[add(i, j) for j in range(q)] for i in range(q)]
        sub_table = [[sub(i, j) for j in range(q)] for i in range(q)]

        def compute_neighbours(vertex):
            side, x = vertex
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                if side == 0:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = add_table[x[c]][mul_table[first if a == 1 else y[a]][x[b]]]
                else:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]
                result.append((1 - side, tuple(y)))
            return result

        start = (0, (0,) * (d + 1))
        index = {start: 0}
        order = [start]
        edges = []
        head = 0
        while head < len(order):
            if time.time() > deadline:
                break
            vertex = order[head]
            head += 1
            for other in compute_neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Try Voltage Graph Lift on K_{4,4}
    if k == 4 and g == 10:
        try:
            # K_{4,4} has 8 vertices: 0-3 (left), 4-7 (right)
            # Edges: (i, 4+j) for i in 0..3, j in 0..3
            base_edges = []
            for i in range(4):
                for j in range(4):
                    base_edges.append((i, 4+j))
            
            # Precompute cycles in K_{4,4}
            # 4-cycles: i1, i2 in left, j1, j2 in right
            cycles_4 = []
            for i1 in range(4):
                for i2 in range(i1+1, 4):
                    for j1 in range(4):
                        for j2 in range(j1+1, 4):
                            # Cycle: i1-j1-i2-j2-i1
                            # Edges: (i1, 4+j1), (i2, 4+j1), (i2, 4+j2), (i1, 4+j2)
                            # Note: orientation matters for voltage sum? 
                            # In a bipartite lift, a cycle is closed if sum of voltages along the cycle is 0.
                            # We need to define a consistent orientation.
                            # Let's store edge indices.
                            e1 = base_edges.index((i1, 4+j1))
                            e2 = base_edges.index((i2, 4+j1))
                            e3 = base_edges.index((i2, 4+j2))
                            e4 = base_edges.index((i1, 4+j2))
                            # Cycle i1->j1->i2->j2->i1
                            # Voltage sum: v(e1) - v(e2) + v(e3) - v(e4) ?
                            # Actually, for a lift to have a cycle, the sum of voltages around the cycle must be 0.
                            # Let's just check if any assignment makes the sum 0.
                            cycles_4.append((e1, e2, e3, e4))
            
            # 6-cycles in K_{4,4}
            # i1, i2, i3 in left, j1, j2, j3 in right
            cycles_6 = []
            for i1 in range(4):
                for i2 in range(4):
                    if i1 == i2: continue
                    for i3 in range(4):
                        if i1 == i3 or i2 == i3: continue
                        for j1 in range(4):
                            for j2 in range(4):
                                if j1 == j2: continue
                                for j3 in range(4):
                                    if j1 == j3 or j2 == j3: continue
                                    # Cycle i1-j1-i2-j2-i3-j3-i1
                                    e1 = base_edges.index((i1, 4+j1))
                                    e2 = base_edges.index((i2, 4+j1))
                                    e3 = base_edges.index((i2, 4+j2))
                                    e4 = base_edges.index((i3, 4+j2))
                                    e5 = base_edges.index((i3, 4+j3))
                                    e6 = base_edges.index((i1, 4+j3))
                                    cycles_6.append((e1, e2, e3, e4, e5, e6))
            
            # 8-cycles in K_{4,4}
            # All 4 left, all 4 right
            cycles_8 = []
            # There are many. We might skip 8-cycles initially to save time, 
            # or check only a subset. If 4 and 6 are avoided, 8 is less likely to be 0?
            # Actually, if 4-cycles are avoided, 8-cycles can still be 0.
            # Let's generate a few.
            # Permutations of 4 left and 4 right.
            # This might be too many. 
            # Let's just check 4 and 6 cycles first.
            
            n_lift = 40
            for attempt in range(10000):
                if time.time() > deadline - 20:
                    break
                voltages = [random.randint(0, n_lift-1) for _ in range(len(base_edges))]
                
                valid = True
                # Check 4-cycles
                for c in cycles_4:
                    # Sum of voltages with signs?
                    # In a voltage graph, the lift has a cycle if the sum of voltages along the cycle is 0 in the group.
                    # We need to be careful with orientation.
                    # Let's assume all edges are oriented Left->Right.
                    # Then a cycle L-R-L-R...
                    # i1->j1 (v1), i2->j1 (v2, but reverse is -v2), i2->j2 (v3), i1->j2 (v4, reverse is -v4).
                    # Sum: v1 - v2 + v3 - v4 == 0?
                    # Let's check: v(e1) - v(e2) + v(e3) - v(e4)
                    s = voltages[c[0]] - voltages[c[1]] + voltages[c[2]] - voltages[c[3]]
                    if s % n_lift == 0:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check 6-cycles
                # i1->j1 (v1), i2->j1 (-v2), i2->j2 (v3), i3->j2 (-v4), i3->j3 (v5), i1->j3 (-v6)
                # Sum: v1 - v2 + v3 - v4 + v5 - v6
                for c in cycles_6:
                    s = voltages[c[0]] - voltages[c[1]] + voltages[c[2]] - voltages[c[3]] + voltages[c[4]] - voltages[c[5]]
                    if s % n_lift == 0:
                        valid = False
                        break
                if not valid:
                    continue
                
                # If we pass 4 and 6, it's likely girth >= 8. 
                # We need girth 10, so no 8-cycles.
                # Checking all 8-cycles is expensive. 
                # But if we find a valid assignment for 4 and 6, 
                # we can construct the graph and verify girth with BFS.
                
                # Construct the lift graph
                n_vertices = 8 * n_lift
                edges = []
                # Vertex (u, k) where u in 0..7, k in 0..n_lift-1
                for e_idx, (u, v) in enumerate(base_edges):
                    vol = voltages[e_idx]
                    for k in range(n_lift):
                        u_lift = u * n_lift + k
                        v_lift = v * n_lift + (k + vol) % n_lift
                        edges.append((u_lift, v_lift))
                
                # Verify girth with BFS
                # This might be slow for 320 vertices, but 320 is small.
                # BFS from every vertex to depth 4 (to find cycles of length 8? No, girth 10 means no cycles < 10.
                # So we need to check for cycles of length 4, 6, 8.
                # BFS from each vertex to depth 4 will find paths of length 4.
                # If we find a path of length 4 from u to u, it's a 4-cycle?
                # Actually, BFS from u to depth 4 finds all vertices at distance <= 4.
                # If we find an edge (x, y) where dist(x) + dist(y) + 1 < girth, it's a short cycle.
                
                # Simpler: Just run BFS from each vertex up to depth 4 (radius 4).
                # If we find a cycle of length < 10, it's invalid.
                # Since it's bipartite, only even cycles. Check 4, 6, 8.
                
                adj = [[] for _ in range(n_vertices)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                girth_found = 11
                for start in range(n_vertices):
                    if time.time() > deadline - 10:
                        break
                    dist = [-1] * n_vertices
                    dist[start] = 0
                    q_bfs = deque([start])
                    while q_bfs:
                        u = q_bfs.popleft()
                        if dist[u] >= 4:
                            continue
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                q_bfs.append(v)
                            elif dist[v] <= dist[u]:
                                # Found a cycle
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < girth_found:
                                    girth_found = cycle_len
                                    if girth_found < 10:
                                        break
                    if girth_found < 10:
                        break
                
                if girth_found >= 10:
                    return n_vertices, edges
                
        except Exception:
            pass

    return luw_construct()
