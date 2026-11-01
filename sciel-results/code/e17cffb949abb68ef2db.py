def construct(k, g):
    import time
    from collections import deque
    import numpy as np
    import random
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    def time_left():
        return deadline - time.time()
    
    # Known optimal cages
    known_cages = {}
    
    # (3,3): K4
    known_cages[(3, 3)] = (4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])
    # (3,4): K_{3,3}
    known_cages[(3, 4)] = (6, [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)])
    # (3,5): Petersen
    petersen_edges = [(0,1),(1,2),(2,3),(3,4),(4,0),
                      (5,7),(7,9),(9,6),(6,8),(8,5),
                      (0,5),(1,6),(2,7),(3,8),(4,9)]
    known_cages[(3, 5)] = (10, petersen_edges)
    # (4,3): K5
    known_cages[(4, 3)] = (5, [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)])
    # (4,4): K_{4,4}
    known_cages[(4, 4)] = (8, [(i,j) for i in range(4) for j in range(4,8)])
    # (5,3): K6
    known_cages[(5, 3)] = (6, [(i,j) for i in range(6) for j in range(i+1,6)])
    # (5,4): K_{5,5}
    known_cages[(5, 4)] = (10, [(i,j) for i in range(5) for j in range(5,10)])
    
    # (3,6): Heawood Graph (Levi graph of Fano plane)
    # 7 points (0-6), 7 lines (7-13)
    # Fano plane lines: 012, 034, 056, 135, 146, 236, 245
    heawood_lines = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    heawood_edges = []
    for i, line in enumerate(heawood_lines):
        line_node = 7 + i
        for point in line:
            heawood_edges.append((point, line_node))
    known_cages[(3, 6)] = (14, heawood_edges)
    
    # (3,7): Tutte-Coxeter Graph (30 vertices, 3-regular, girth 8? No, girth 8 is McGee. Tutte-Coxeter is girth 8?
    # Actually, Tutte-Coxeter is (3,7) cage? No, (3,7) is 30 vertices.
    # The (3,7) cage is the Tutte-Coxeter graph.
    # I will skip hardcoding this for now and rely on LUW if not found.
    
    if (k, g) in known_cages:
        return known_cages[(k, g)]
    
    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    def girth_check_fast(n, adj, g):
        """Faster girth check: only explore up to depth g-1."""
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= g - 1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < g - 1:
                            queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True
    
    def build_luw(d, q):
        """Build Lazebnik-Ustimenko-Woldar graph CD(d,q)."""
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        
        rule = {}
        rule[2] = (1,1)
        rule[3] = (2,1)
        rule[4] = (1,2)
        for c in range(5, d+1):
            r = (c-5)%4
            c0 = c - (c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1-side), tuple(y)
        
        start = (0, (None,) + (0,)*d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges
    
    # Try LUW constructions
    best_n = None
    best_edges = None
    
    q = k
    d_min = max(3, (g + 1) // 2)
    
    for d in range(d_min, d_min + 3):
        if time_left() < 20:
            break
        try:
            n_luw, edges_luw = build_luw(d, q)
            
            if n_luw > 50000:
                continue
            
            adj_luw = build_adj(n_luw, edges_luw)
            
            ok = True
            for i in range(n_luw):
                if len(adj_luw[i]) != k:
                    ok = False
                    break
            if not ok:
                continue
            
            if girth_check_fast(n_luw, adj_luw, g):
                if best_n is None or n_luw < best_n:
                    best_n = n_luw
                    best_edges = edges_luw
                break
                
        except Exception:
            continue
    
    if best_n is None:
        # Fallback: simple circulant
        n = k + g
        jumps = list(range(1, (k+1)//2 + 1))
        if k % 2 == 0:
            jumps = list(range(1, k//2 + 1))
        
        edges = []
        seen = set()
        for i in range(n):
            for j in jumps:
                u = i
                v = (i + j) % n
                if u < v:
                    edge = (u, v)
                else:
                    edge = (v, u)
                if edge not in seen:
                    seen.add(edge)
                    edges.append(edge)
        best_n = n
        best_edges = edges
    
    return best_n, best_edges
