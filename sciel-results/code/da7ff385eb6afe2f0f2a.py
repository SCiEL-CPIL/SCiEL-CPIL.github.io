import numpy as np
import random
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145
    
    # Primary: LUW construction (deterministic, fast)
    q = k
    if g % 2 == 0:
        d = g // 2
    else:
        d = (g + 1) // 2
    
    if q == 4:
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul_table = [[0, 0, 0, 0],
                     [0, 1, 2, 3],
                     [0, 2, 3, 1],
                     [0, 3, 1, 2]]
        mul = lambda x, y: mul_table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]
    
    def neighbours(vertex):
        side, x = vertex
        x_list = list(x)
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x_list[c], mul(y[a], x_list[b]))
                else:
                    y[c] = sub(x_list[c], mul(x_list[a], y[b]))
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    idx = 0
    while idx < len(order):
        if time.time() > deadline - 10:
            break
        vertex = order[idx]
        idx += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n_luw = len(order)
    
    # Verify LUW graph
    adj = [[] for _ in range(n_luw)]
    for u, v in edges:
        if u < n_luw and v < n_luw:
            adj[u].append(v)
            adj[v].append(u)
    
    valid_luw = True
    for i in range(n_luw):
        if len(adj[i]) != k:
            valid_luw = False
            break
    
    luw_girth_ok = False
    if valid_luw:
        min_girth = 9999
        for s in range(n_luw):
            if time.time() > deadline - 5:
                break
            dist = [-1] * n_luw
            parent = [-1] * n_luw
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                cur = queue[qi]
                qi += 1
                for nb in adj[cur]:
                    if dist[nb] == -1:
                        dist[nb] = dist[cur] + 1
                        parent[nb] = cur
                        queue.append(nb)
                    elif parent[nb] != cur and parent[cur] != nb:
                        cycle_len = dist[nb] + dist[cur] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        if min_girth < g:
                            break
                if min_girth < g:
                    break
            if min_girth < g:
                break
        luw_girth_ok = (min_girth >= g)
    
    if valid_luw and luw_girth_ok:
        return n_luw, edges
    
    # Try voltage lift of PG(2,3) if time permits
    if time.time() < deadline - 30:
        try:
            m = 10
            # Build PG(2,3) base graph
            points = []
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        if x == 0 and y == 0 and z == 0:
                            continue
                        if x != 0:
                            inv_x = 1 if x == 1 else 2
                            p = (1, (y * inv_x) % 3, (z * inv_x) % 3)
                        elif y != 0:
                            inv_y = 1 if y == 1 else 2
                            p = (0, 1, (z * inv_y) % 3)
                        else:
                            p = (0, 0, 1)
                        if p not in points:
                            points.append(p)
            
            lines = []
            line_points = []
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        if a == 0 and b == 0 and c == 0:
                            continue
                        if a != 0:
                            inv_a = 1 if a == 1 else 2
                            n = (1, (b * inv_a) % 3, (c * inv_a) % 3)
                        elif b != 0:
                            inv_b = 1 if b == 1 else 2
                            n = (0, 1, (c * inv_b) % 3)
                        else:
                            n = (0, 0, 1)
                        if n not in lines:
                            lines.append(n)
                            pts = []
                            for p in points:
                                if (n[0] * p[0] + n[1] * p[1] + n[2] * p[2]) % 3 == 0:
                                    pts.append(p)
                            line_points.append(pts)
            
            base_edges = []
            for i, p in enumerate(points):
                for j, lp in enumerate(line_points):
                    if p in lp:
                        base_edges.append((i, 13 + j))
            
            base_adj = [[] for _ in range(26)]
            for u, v in base_edges:
                base_adj[u].append(v)
                base_adj[v].append(u)
            
            # Find all cycles of length 6 and 8 in base graph
            cycles_6 = []
            for p1 in range(13):
                for l1 in base_adj[p1]:
                    for p2 in base_adj[l1]:
                        if p2 == p1:
                            continue
                        for l2 in base_adj[p2]:
                            if l2 == l1:
                                continue
                            for p3 in base_adj[l2]:
                                if p3 == p1 or p3 == p2:
                                    continue
                                for l3 in base_adj[p3]:
                                    if l3 == l2:
                                        continue
                                    if p1 in base_adj[l3]:
                                        cycle = [p1, l1, p2, l2, p3, l3]
                                        min_idx = cycle.index(min(cycle))
                                        normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
                                        if normalized not in cycles_6:
                                            cycles_6.append(normalized)
            
            cycles_8 = []
            for p1 in range(13):
                for l1 in base_adj[p1]:
                    for p2 in base_adj[l1]:
                        if p2 == p1:
                            continue
                        for l2 in base_adj[p2]:
                            if l2 == l1:
                                continue
                            for p3 in base_adj[l2]:
                                if p3 == p1 or p3 == p2:
                                    continue
                                for l3 in base_adj[p3]:
                                    if l3 == l2 or l3 == l1:
                                        continue
                                    for p4 in base_adj[l3]:
                                        if p4 == p1 or p4 == p2 or p4 == p3:
                                            continue
                                        for l4 in base_adj[p4]:
                                            if l4 == l3 or l4 == l1 or l4 == l2:
                                                continue
                                            if p1 in base_adj[l4]:
                                                cycle = [p1, l1, p2, l2, p3, l3, p4, l4]
                                                min_idx = cycle.index(min(cycle))
                                                normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
                                                if normalized not in cycles_8:
                                                    cycles_8.append(normalized)
            
            # Voltage search
            best_voltages = None
            max_attempts = 2000
            for attempt in range(max_attempts):
                if time.time() > deadline - 15:
                    break
                voltages = {}
                for u, v in base_edges:
                    vu = random.randint(0, m - 1)
                    voltages[(u, v)] = vu
                    voltages[(v, u)] = (-vu) % m
                
                valid = True
                for cycle in cycles_6:
                    s = 0
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        s += voltages[(u, v)]
                    if s % m == 0:
                        valid = False
                        break
                if not valid:
                    continue
                for cycle in cycles_8:
                    s = 0
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        s += voltages[(u, v)]
                    if s % m == 0:
                        valid = False
                        break
                if valid:
                    best_voltages = voltages
                    break
            
            if best_voltages is not None:
                n_lift = 26 * m
                lift_edges = []
                for u, v in base_edges:
                    vu = best_voltages[(u, v)]
                    for i in range(m):
                        lift_edges.append((u * m + i, v * m + (i + vu) % m))
                
                # Verify
                adj_lift = [[] for _ in range(n_lift)]
                for u, v in lift_edges:
                    adj_lift[u].append(v)
                    adj_lift[v].append(u)
                
                deg_ok = all(len(adj_lift[i]) == k for i in range(n_lift))
                if deg_ok:
                    min_girth = 9999
                    for s in range(n_lift):
                        if time.time() > deadline - 3:
                            break
                        dist = [-1] * n_lift
                        parent = [-1] * n_lift
                        dist[s] = 0
                        queue = [s]
                        qi = 0
                        while qi < len(queue):
                            cur = queue[qi]
                            qi += 1
                            for nb in adj_lift[cur]:
                                if dist[nb] == -1:
                                    dist[nb] = dist[cur] + 1
                                    parent[nb] = cur
                                    queue.append(nb)
                                elif parent[nb] != cur and parent[cur] != nb:
                                    cycle_len = dist[nb] + dist[cur] + 1
                                    if cycle_len < min_girth:
                                        min_girth = cycle_len
                                    if min_girth < g:
                                        break
                            if min_girth < g:
                                break
                        if min_girth < g:
                            break
                    if min_girth >= g:
                        return n_lift, lift_edges
        except Exception:
            pass
    
    # Return LUW result as fallback
    return n_luw, edges
