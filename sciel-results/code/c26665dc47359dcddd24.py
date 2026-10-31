"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import deque
import time

def construct(k, g):
    # Random lift of Petersen graph with local search to achieve girth g
    base_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),
        (5,7), (7,9), (9,6), (6,8), (8,5),
        (0,5), (1,6), (2,7), (3,8), (4,9)
    ]
    edge_keys = [(min(u,v), max(u,v)) for u, v in base_edges]
    
    def build_lift(m, twists):
        n = 10 * m
        adj = [[] for _ in range(n)]
        for u, v in base_edges:
            key = (min(u,v), max(u,v))
            t = twists.get(key, 0)
            for i in range(m):
                u_lift = u * m + i
                v_lift = v * m + (i + t) % m
                adj[u_lift].append(v_lift)
                adj[v_lift].append(u_lift)
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return n, adj, edges
    
    def find_short_cycle(n, adj, max_len):
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            queue = deque([(start, -1)])
            dist[start] = 0
            while queue:
                u, par = queue.popleft()
                for v in adj[u]:
                    if v == par:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append((v, u))
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if 3 <= cycle_len < max_len:
                            cycle = []
                            path_u = []
                            cur = u
                            while cur != -1:
                                path_u.append(cur)
                                cur = parent[cur]
                            path_v = []
                            cur = v
                            while cur != -1:
                                path_v.append(cur)
                                cur = parent[cur]
                            set_u = set(path_u)
                            lca = None
                            for node in path_v:
                                if node in set_u:
                                    lca = node
                                    break
                            if lca is None:
                                continue
                            cycle_u = []
                            cur = u
                            while cur != lca:
                                cycle_u.append(cur)
                                cur = parent[cur]
                            cycle_u.append(lca)
                            cycle_v = []
                            cur = v
                            while cur != lca:
                                cycle_v.append(cur)
                                cur = parent[cur]
                            cycle = cycle_u + list(reversed(cycle_v[1:]))
                            cycle_edges = []
                            for i in range(len(cycle)):
                                a, b = cycle[i], cycle[(i+1) % len(cycle)]
                                if a < b:
                                    cycle_edges.append((a, b))
                                else:
                                    cycle_edges.append((b, a))
                            return cycle, cycle_edges
        return None
    
    def try_2switch(adj, edges, e1, e2):
        u, v = e1
        x, y = e2
        if u == x or u == y or v == x or v == y:
            return False, edges
        if x in adj[u] or y in adj[u] or x in adj[v] or y in adj[v]:
            return False, edges
        if u in adj[x] or u in adj[y] or v in adj[x] or v in adj[y]:
            return False, edges
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        adj[u].append(x)
        adj[x].append(u)
        adj[v].append(y)
        adj[y].append(v)
        edge_set = set()
        for e in edges:
            a, b = e
            if a < b:
                edge_set.add((a,b))
            else:
                edge_set.add((b,a))
        edge_set.remove(tuple(sorted(e1)))
        edge_set.remove(tuple(sorted(e2)))
        edge_set.add(tuple(sorted((u,x))))
        edge_set.add(tuple(sorted((v,y))))
        new_edges = [list(e) for e in edge_set]
        return True, new_edges
    
    def check_girth(n, adj, target_g):
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            queue = deque([(start, -1)])
            dist[start] = 0
            while queue:
                u, par = queue.popleft()
                for v in adj[u]:
                    if v == par:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append((v, u))
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_g:
                            return False
        return True
    
    best_result = None
    best_n = float('inf')
    start_time = time.time()
    
    for m in range(60, 130, 1):
        if time.time() - start_time > 100:
            break
        n = 10 * m
        if n >= best_n:
            continue
        
        for attempt in range(15):
            if time.time() - start_time > 105:
                break
            random.seed(99999 + m * 1000 + attempt)
            twists = {key: random.randint(0, m-1) for key in edge_keys}
            
            n_lift, adj, edges = build_lift(m, twists)
            
            if check_girth(n_lift, adj, g):
                if n_lift < best_n:
                    best_n = n_lift
                    best_result = (n_lift, edges)
                break
            
            for it in range(300):
                if time.time() - start_time > 108:
                    break
                result = find_short_cycle(n_lift, adj, g)
                if result is None:
                    if check_girth(n_lift, adj, g):
                        if n_lift < best_n:
                            best_n = n_lift
                            best_result = (n_lift, edges)
                    break
                
                cycle, cycle_edges = result
                switched = False
                if len(cycle_edges) >= 4:
                    for i in range(len(cycle_edges)):
                        for j in range(i+2, min(i+3, len(cycle_edges))):
                            if j == i + len(cycle_edges) - 1:
                                continue
                            e1, e2 = cycle_edges[i], cycle_edges[j]
                            success, new_edges = try_2switch(adj, edges, e1, e2)
                            if success:
                                edges = new_edges
                                switched = True
                                break
                        if switched:
                            break
                if not switched:
                    break
            
            if check_girth(n_lift, adj, g):
                if n_lift < best_n:
                    best_n = n_lift
                    best_result = (n_lift, edges)
                break
    
    if best_result is not None:
        return best_result
    
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
# EVOLVE-BLOCK-END
