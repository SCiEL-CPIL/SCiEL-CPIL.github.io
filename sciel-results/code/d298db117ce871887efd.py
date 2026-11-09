import numpy as np
import time
import random

def construct():
    n = 20
    deadline = time.time() + 110
    best_edges = 0
    best_adj = None
    
    random.seed(42)
    np.random.seed(42)
    
    # Precompute 4-cycles: each 4-cycle is a set of 4 edges (tuples (u,v) with u < v)
    all_4cycles = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # 4-cycles with vertices a, b, c, d
                    # (a,b), (b,c), (c,d), (d,a)
                    edges1 = [(a,b), (b,c), (c,d), (a,d)]
                    all_4cycles.append(edges1)
                    # (a,b), (b,d), (d,c), (c,a)
                    edges2 = [(a,b), (b,d), (c,d), (a,c)]
                    all_4cycles.append(edges2)
                    # (a,c), (c,b), (b,d), (d,a)
                    edges3 = [(a,c), (b,c), (b,d), (a,d)]
                    all_4cycles.append(edges3)
    
    # Build edge-to-cycles mapping
    edge_to_cycles = {}
    for i, cyc in enumerate(all_4cycles):
        for e in cyc:
            if e not in edge_to_cycles:
                edge_to_cycles[e] = []
            edge_to_cycles[e].append(i)
    
    def count_violations(adj):
        """Count the number of 4-cycles that are NOT rainbow."""
        vcount = 0
        for cyc in all_4cycles:
            colors = []
            valid = True
            for e in cyc:
                u, v = e
                if u > v:
                    u, v = v, u
                c = adj[u, v]
                if c == -1:
                    valid = False
                    break
                colors.append(c)
            if not valid:
                continue
            if len(set(colors)) < 4:
                vcount += 1
        return vcount
    
    def can_add_or_recolor(adj, u, v, new_c, old_c=-1):
        """Check if setting adj[u,v] = new_c maintains the rainbow 4-cycle property.
        If old_c >= 0, we're recoloring; otherwise adding a new edge."""
        if u > v:
            u, v = v, u
        if u == v:
            return False
        
        # Temporarily set the edge
        old_val = adj[u, v]
        adj[u, v] = new_c
        adj[v, u] = new_c
        
        # Check all 4-cycles that include this edge
        cycles = edge_to_cycles.get((u, v), [])
        for ci in cycles:
            cyc = all_4cycles[ci]
            colors = []
            valid = True
            for e in cyc:
                a, b = e
                if a > b:
                    a, b = b, a
                c = adj[a, b]
                if c == -1:
                    valid = False
                    break
                colors.append(c)
            if not valid:
                continue
            if len(set(colors)) < 4:
                adj[u, v] = old_val
                adj[v, u] = old_val
                return False
        
        adj[u, v] = old_val
        adj[v, u] = old_val
        return True
    
    for restart in range(200):
        if time.time() > deadline - 2:
            break
        
        # Build an initial random graph: start with random sparse graph, then add edges
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)
        
        # Try to add edges in random order
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline - 2:
                break
            if adj[u, v] != -1:
                continue
            # Try a random color
            for attempt in range(5):
                c = random.randint(0, n-1)
                if can_add_or_recolor(adj, u, v, c):
                    adj[u, v] = c
                    adj[v, u] = c
                    break
        
        # Local search: try to add remaining edges and recolor existing ones
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try adding missing edges
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == -1]
            random.shuffle(edges)
            for u, v in edges:
                if time.time() > deadline - 2:
                    break
                for c in range(n):
                    if can_add_or_recolor(adj, u, v, c):
                        adj[u, v] = c
                        adj[v, u] = c
                        improved = True
                        break
        
        cnt = int(np.sum(adj != -1) // 2)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj.copy()
            if best_edges >= 180:
                break
    
    res = []
    if best_adj is None:
        for v in range(1, n):
            res.append([0, v, 0])
        return res
    
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] != -1:
                res.append([i, j, int(best_adj[i, j])])
    return res
