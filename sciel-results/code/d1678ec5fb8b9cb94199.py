import numpy as np
import time
import random

def count_edges(adj):
    return np.sum(adj != -1) // 2

def construct():
    n = 20
    deadline = time.time() + 115
    random.seed(42)
    np.random.seed(42)
    
    # Precompute all 4-cycles: list of (u, v, x, y) where u-v-x-y-u is a cycle
    # We need adj[u,v], adj[v,x], adj[x,y], adj[y,u] all present and 4 distinct colors
    cycles = []
    for u in range(n):
        for v in range(u+1, n):
            for x in range(n):
                if x == u or x == v:
                    continue
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    # To avoid duplicates, we can just enumerate all and dedupe later,
                    # or use a canonical form. For now, just check all.
                    # A 4-cycle u-v-x-y means edges (u,v), (v,x), (x,y), (y,u)
                    cycles.append((u, v, x, y))
    
    # Actually, let's be more careful. A 4-cycle is a set of 4 edges.
    # The edges are (u,v), (v,x), (x,y), (y,u).
    # We need to make sure we don't count the same cycle multiple times or in reverse.
    # But for validity checking, it's fine to check all ordered tuples that form a cycle.
    # The condition is: all 4 edges exist AND all 4 colors are distinct.
    
    # Precompute for each edge (u,v), which cycles it participates in.
    # This allows fast checking: when adding edge (u,v) with color c,
    # we only need to check cycles that include (u,v).
    
    # Let's build: edge_to_cycles[(u,v)] = list of (x, y) such that (u,v,x,y) is a 4-cycle
    # But we also need to handle the reverse direction and different starting points.
    # 
    # Actually, let's think differently. 
    # When we add edge (u,v) with color c, we need to check all 4-cycles that include edge (u,v).
    # A 4-cycle containing edge (u,v) has the form: u-v-x-y-u where x,y are other vertices.
    # So the other three edges are (v,x), (x,y), (y,u).
    # 
    # For each pair (x,y) with x != u,v and y != u,v,x, we need:
    #   adj[v,x] != -1, adj[x,y] != -1, adj[y,u] != -1
    #   and {c, adj[v,x], adj[x,y], adj[y,u]} has 4 distinct elements.
    
    # So for edge (u,v), the relevant "witnesses" are pairs (x,y) where:
    #   x is a neighbor of v (x != u), y is a neighbor of x (y != u,v,x), and y is a neighbor of u.
    # 
    # Let's precompute for each vertex u, its neighbors.
    
    best_count = 0
    best_adj = None
    
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Initialize with a random spanning tree
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)
        
        vertices = list(range(n))
        random.shuffle(vertices)
        for i in range(1, n):
            u = vertices[i]
            v = vertices[random.randint(0, i-1)]
            c = random.randint(0, n-1)
            adj[u, v] = c
            adj[v, u] = c
        
        # Greedy edge addition
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            
            colors = list(range(n))
            random.shuffle(colors)
            
            for c in colors:
                # Check validity: for all x (neighbor of v, x != u), y (neighbor of x, y != u,v,x),
                # if y is neighbor of u, then {c, adj[v,x], adj[x,y], adj[y,u]} must have 4 distinct colors
                valid = True
                for x in range(n):
                    if x == u or x == v:
                        continue
                    if adj[v, x] == -1:
                        continue
                    for y in range(n):
                        if y == u or y == v or y == x:
                            continue
                        if adj[x, y] == -1:
                            continue
                        if adj[y, u] == -1:
                            continue
                        cs = {c, adj[v,x], adj[x,y], adj[y,u]}
                        if len(cs) < 4:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    adj[u, v] = c
                    adj[v, u] = c
                    break
        
        # Local search: iteratively try to add missing edges and recolor
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding missing edges
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == -1]
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline:
                    break
                if adj[u, v] != -1:
                    continue
                
                colors = list(range(n))
                random.shuffle(colors)
                
                for c in colors:
                    valid = True
                    for x in range(n):
                        if x == u or x == v:
                            continue
                        if adj[v, x] == -1:
                            continue
                        for y in range(n):
                            if y == u or y == v or y == x:
                                continue
                            if adj[x, y] == -1:
                                continue
                            if adj[y, u] == -1:
                                continue
                            cs = {c, adj[v,x], adj[x,y], adj[y,u]}
                            if len(cs) < 4:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        adj[u, v] = c
                        adj[v, u] = c
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try recoloring an existing edge
            existing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] != -1]
            random.shuffle(existing)
            
            for u, v in existing[:15]:
                if time.time() > deadline:
                    break
                
                old_c = adj[u, v]
                colors = list(range(n))
                random.shuffle(colors)
                
                for new_c in colors:
                    if new_c == old_c:
                        continue
                    
                    # Check if recoloring is valid
                    valid = True
                    for x in range(n):
                        if x == u or x == v:
                            continue
                        if adj[v, x] == -1:
                            continue
                        for y in range(n):
                            if y == u or y == v or y == x:
                                continue
                            if adj[x, y] == -1:
                                continue
                            if adj[y, u] == -1:
                                continue
                            cs = {new_c, adj[v,x], adj[x,y], adj[y,u]}
                            if len(cs) < 4:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        adj[u, v] = new_c
                        adj[v, u] = new_c
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj.copy()
        
        if best_count > 150:
            break
    
    # Convert to result
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
