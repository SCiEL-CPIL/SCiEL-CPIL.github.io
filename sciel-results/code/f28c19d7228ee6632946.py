"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    deadline = time.time() + 130.0  # safety margin before 150s kill
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned.
    q = k
    # For girth g, we need d such that d + 5 >= g for odd d
    # Try smallest d first, verify girth, increase if needed
    best_n = None
    best_edges = None
    
    for d_candidate in range(max(3, g - 5), g + 10):
        if d_candidate % 2 == 0:
            continue  # only odd d gives guaranteed girth d+5
        if time.time() > deadline - 20:
            break
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add_fn = lambda x, y: x ^ y
            sub_fn = lambda x, y: x ^ y
            mul_fn = lambda x, y: table[x][y]
        else:
            add_fn = lambda x, y: (x + y) % q
            sub_fn = lambda x, y: (x - y) % q
            mul_fn = lambda x, y: x * y % q
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d_candidate + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        # Precompute rule as list for faster access
        rule_list = [None] * (d_candidate + 1)
        for c in range(2, d_candidate + 1):
            rule_list[c] = rule[c]
        
        def neighbours(vertex, q=q, d=d_candidate, rule_list=rule_list, add_fn=add_fn, sub_fn=sub_fn, mul_fn=mul_fn):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule_list[c]
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b])) if side == 0 else sub_fn(x[c], mul_fn(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d_candidate)
        index, order, edges = {start: 0}, [start], []
        
        for vertex in order:
            if time.time() > deadline - 10:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        if time.time() > deadline - 5:
            break
        
        n = len(order)
        # Verify all vertices have degree k
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(dg != k for dg in deg):
            continue
        
        # Verify girth >= g using BFS from each vertex up to depth g//2
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        girth_ok = True
        half_g = g // 2
        for s in range(n):
            if time.time() > deadline - 3:
                break
            # BFS from s, tracking parent to detect cycles
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue) and girth_ok:
                u = queue[qi]
                qi += 1
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and dist[u] + dist[v] + 1 < g:
                        girth_ok = False
                        break
            if not girth_ok:
                break
        
        if girth_ok:
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges
            # For LUW, smaller d gives smaller graph but may have smaller girth
            # If this d works, larger d won't be smaller, so we can stop
            # Actually larger d = more vertices, so smallest valid d is best
            break
    
    if best_n is not None:
        return best_n, best_edges
    else:
        # Fallback: use the original d calculation
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add_fn = lambda x, y: x ^ y
            sub_fn = lambda x, y: x ^ y
            mul_fn = lambda x, y: table[x][y]
        else:
            add_fn = lambda x, y: (x + y) % q
            sub_fn = lambda x, y: (x - y) % q
            mul_fn = lambda x, y: x * y % q
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        rule_list = [None] * (d + 1)
        for c in range(2, d + 1):
            rule_list[c] = rule[c]
        
        def neighbours(vertex, q=q, d=d, rule_list=rule_list, add_fn=add_fn, sub_fn=sub_fn, mul_fn=mul_fn):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule_list[c]
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b])) if side == 0 else sub_fn(x[c], mul_fn(x[a], y[b]))
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
# EVOLVE-BLOCK-END
