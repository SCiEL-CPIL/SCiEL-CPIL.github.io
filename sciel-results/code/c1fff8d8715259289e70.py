import numpy as np
import time
import random

def count_edges(adj):
    return np.sum(adj != -1) // 2

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    random.seed(42)
    np.random.seed(42)
    
    # Precompute 4-cycle witnesses for each edge (u,v)
    # witness[(u,v)] = list of (x, y) such that u-v-x-y-u is a 4-cycle
    witness = {}
    for u in range(n):
        for v in range(u+1, n):
            wl = []
            for x in range(n):
                if x == u or x == v:
                    continue
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    wl.append((x, y))
            witness[(u, v)] = wl
            witness[(v, u)] = wl
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_forbidden(u, v, adj):
        """
        Returns set of colors forbidden for edge (u,v).
        If the 3 other edges in any 4-cycle have duplicate colors, return all colors.
        """
        forbidden = set()
        for (x, y) in witness[(u, v)]:
            if adj[v, x] == -1:
                continue
            if adj[x, y] == -1:
                continue
            if adj[y, u] == -1:
                continue
            
            c1 = adj[v, x]
            c2 = adj[x, y]
            c3 = adj[y, u]
            
            if c1 == c2 or c1 == c3 or c2 == c3:
                return set(range(n))
            
            forbidden.add(c1)
            forbidden.add(c2)
            forbidden.add(c3)
        
        return forbidden
    
    def is_recolor_valid(u, v, new_c, adj):
        """Check if recoloring edge (u,v) to new_c keeps all 4-cycles rainbow."""
        for (x, y) in witness[(u, v)]:
            if adj[v, x] == -1:
                continue
            if adj[x, y] == -1:
                continue
            if adj[y, u] == -1:
                continue
            
            c1 = adj[v, x]
            c2 = adj[x, y]
            c3 = adj[y, u]
            
            if new_c == c1 or new_c == c2 or new_c == c3:
                return False
        return True
    
    def is_valid(adj):
        """Check if the entire coloring is valid (all 4-cycles rainbow)."""
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v] == -1:
                    continue
                for (x, y) in witness[(u, v)]:
                    if adj[v, x] == -1:
                        continue
                    if adj[x, y] == -1:
                        continue
                    if adj[y, u] == -1:
                        continue
                    
                    c1 = adj[u, v]
                    c2 = adj[v, x]
                    c3 = adj[x, y]
                    c4 = adj[y, u]
                    
                    if len({c1, c2, c3, c4}) < 4:
                        return False
        return True
    
    best_count = 0
    best_adj = None
    
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Initialize with random spanning tree
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
        
        # Greedy edge addition with forbidden-color check
        edges = all_edges[:]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            
            forb = get_forbidden(u, v, adj)
            if len(forb) < n:
                available = [c for c in range(n) if c not in forb]
                c = random.choice(available)
                adj[u, v] = c
                adj[v, u] = c
        
        # Local search: iteratively add missing edges and recolor
        for iteration in range(800):
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
                
                forb = get_forbidden(u, v, adj)
                if len(forb) < n:
                    available = [c for c in range(n) if c not in forb]
                    c = random.choice(available)
                    adj[u, v] = c
                    adj[v, u] = c
                    improved = True
            
            if improved:
                continue
            
            # Try recoloring existing edges to unblock additions
            existing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] != -1]
            random.shuffle(existing)
            
            for u, v in existing[:30]:
                if time.time() > deadline:
                    break
                
                old_c = adj[u, v]
                colors = list(range(n))
                random.shuffle(colors)
                
                for new_c in colors:
                    if new_c == old_c:
                        continue
                    
                    if is_recolor_valid(u, v, new_c, adj):
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
        
        if best_count > 160:
            break
    
    # Final polish: try harder recoloring on best solution
    if best_adj is not None and time.time() < deadline - 5:
        adj = best_adj.copy()
        for polish_iter in range(500):
            if time.time() > deadline - 2:
                break
            
            improved = False
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == -1]
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline - 2:
                    break
                if adj[u, v] != -1:
                    continue
                forb = get_forbidden(u, v, adj)
                if len(forb) < n:
                    available = [c for c in range(n) if c not in forb]
                    c = random.choice(available)
                    adj[u, v] = c
                    adj[v, u] = c
                    improved = True
            
            if not improved:
                existing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] != -1]
                random.shuffle(existing)
                for u, v in existing[:50]:
                    if time.time() > deadline - 2:
                        break
                    old_c = adj[u, v]
                    colors = list(range(n))
                    random.shuffle(colors)
                    for new_c in colors:
                        if new_c == old_c:
                            continue
                        if is_recolor_valid(u, v, new_c, adj):
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
            best_adj = adj
    
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
