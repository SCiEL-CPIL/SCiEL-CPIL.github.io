import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145
    
    n = 20
    colors = 20
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def check_c4(c, u, v, colour_class, adj):
        for x in adj[c][u]:
            if x == v:
                continue
            for y in adj[c][x]:
                if y == u or y == v:
                    continue
                if (min(y, v), max(y, v)) in colour_class[c]:
                    return True
        return False
    
    def verify(edges):
        adj = {}
        for u, v, c in edges:
            if u not in adj:
                adj[u] = {}
            if v not in adj:
                adj[v] = {}
            adj[u][v] = c
            adj[v][u] = c
        
        for a in range(n):
            for b in range(a+1, n):
                if b not in adj.get(a, {}):
                    continue
                c_ab = adj[a][b]
                for d in range(b+1, n):
                    if d not in adj.get(b, {}):
                        continue
                    c_bd = adj[b][d]
                    for c in range(a+1, n):
                        if c == b:
                            continue
                        if c not in adj.get(d, {}):
                            continue
                        if c not in adj.get(a, {}):
                            continue
                        c_dc = adj[d][c]
                        c_ca = adj[a][c]
                        if len({c_ab, c_bd, c_dc, c_ca}) < 4:
                            return False
        return True
    
    best_edges = []
    best_count = 0
    
    rng = random.Random(12345)
    
    for trial in range(200):
        if time.time() > DEADLINE - 5:
            break
        
        colour_class = [set() for _ in range(colors)]
        adj = [dict() for _ in range(colors)]
        for c in range(colors):
            for u in range(n):
                adj[c][u] = set()
        
        assigned = {}
        edge_order = list(all_edges)
        if trial > 0:
            rng.shuffle(edge_order)
        
        for (u, v) in edge_order:
            if time.time() > DEADLINE - 2:
                break
            for c in range(colors):
                if not check_c4(c, u, v, colour_class, adj):
                    colour_class[c].add((u, v))
                    adj[c][u].add(v)
                    adj[c][v].add(u)
                    assigned[(u, v)] = c
                    break
        
        count = len(assigned)
        if count > best_count:
            candidate = [[u, v, c] for (u, v), c in assigned.items()]
            if verify(candidate):
                best_count = count
                best_edges = candidate
    
    return best_edges
