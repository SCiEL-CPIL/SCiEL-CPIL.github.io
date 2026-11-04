import time
import random
import sys

def construct(k, g):
    deadline = time.time() + 140.0
    
    # LUW construction - primary approach
    def get_luw_graph():
        # Choose minimal d such that girth >= g
        # LUW graph CD(d,q) has girth >= d+5 if d odd, d+4 if d even
        if g <= 5:
            d = 3
        else:
            d = 3
            while True:
                girth_est = d + 5 if d % 2 == 1 else d + 4
                if girth_est >= g:
                    break
                d += 1
        
        q = k
        
        if q == 4:
            mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return mul_table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return (x * y) % q

        # Precompute rules
        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        rule_a[2], rule_b[2] = 1, 1
        rule_a[3], rule_b[3] = 2, 1
        rule_a[4], rule_b[4] = 1, 2
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            if r == 0: a, b = 1, c0 - 2
            elif r == 1: a, b = c0 - 1, 1
            elif r == 2: a, b = c0, 1
            else: a, b = 1, c0 + 1
            rule_a[c] = a
            rule_b[c] = b

        def compute_neighbours(vertex):
            side, x = vertex
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                if side == 0:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = sub(x[c], mul(x[a], y[b]))
                result.append((1 - side, tuple(y)))
            return result

        start = (0, (0,) * (d + 1))
        index = {start: 0}
        order = [start]
        edges = []
        
        for idx in range(len(order)):
            if time.time() > deadline:
                break
            vertex = order[idx]
            for other in compute_neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    # Try voltage graph approach for k=4, g=10 (quick attempt)
    if k == 4 and g == 10:
        n_base = 8
        group_size = 40
        n_total = n_base * group_size
        
        # Base graph: Complement of Q3
        base_edges = []
        for i in range(8):
            for j in range(8):
                if i == j: continue
                if (i ^ j) in [1, 2, 4]: continue
                if i < j:
                    base_edges.append((i, j))
        
        # Quick random voltage search with limited trials
        best_n = n_total + 1
        best_edges = None
        
        max_trials = 2000
        for trial in range(max_trials):
            if time.time() > deadline - 20:
                break
            
            voltages = {}
            for u, v in base_edges:
                voltages[(u, v)] = random.randint(0, group_size - 1)
                voltages[(v, u)] = (group_size - voltages[(u, v)]) % group_size
            
            adj = [[] for _ in range(n_total)]
            for u, v in base_edges:
                vu = voltages[(u, v)]
                for g1 in range(group_size):
                    g2 = (g1 + vu) % group_size
                    u_id = u * group_size + g1
                    v_id = v * group_size + g2
                    adj[u_id].append(v_id)
                    adj[v_id].append(u_id)
            
            # Quick girth check using BFS from a few vertices
            girth_ok = True
            check_vertices = [0, 1, 2, 3, 4]  # Sample vertices
            for start in check_vertices:
                dist = [-1] * n_total
                parent = [-1] * n_total
                queue = [start]
                dist[start] = 0
                parent[start] = -2
                
                while queue:
                    curr = queue.pop(0)
                    if dist[curr] >= g:
                        break
                    for nbr in adj[curr]:
                        if dist[nbr] == -1:
                            dist[nbr] = dist[curr] + 1
                            parent[nbr] = curr
                            queue.append(nbr)
                        elif nbr != parent[curr]:
                            cycle_len = dist[curr] + dist[nbr] + 1
                            if cycle_len < g:
                                girth_ok = False
                                break
                if not girth_ok:
                    break
            
            if girth_ok:
                # Full verification
                full_girth = g
                for start in range(n_total):
                    dist = [-1] * n_total
                    parent = [-1] * n_total
                    queue = [start]
                    dist[start] = 0
                    parent[start] = -2
                    
                    while queue:
                        curr = queue.pop(0)
                        if dist[curr] >= full_girth:
                            break
                        for nbr in adj[curr]:
                            if dist[nbr] == -1:
                                dist[nbr] = dist[curr] + 1
                                parent[nbr] = curr
                                queue.append(nbr)
                            elif nbr != parent[curr]:
                                cycle_len = dist[curr] + dist[nbr] + 1
                                if cycle_len < full_girth:
                                    full_girth = cycle_len
                                    if full_girth < g:
                                        break
                    if full_girth < g:
                        break
                
                if full_girth >= g:
                    best_n = n_total
                    edges = []
                    for i in range(n_total):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    best_edges = edges
                    break
        
        if best_edges is not None:
            return best_n, best_edges
    
    # Fallback to LUW
    return get_luw_graph()
