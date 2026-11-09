import numpy as np
import time
import random

def construct():
    n = 20
    deadline = time.time() + 115
    best_edges = 0
    best_adj = None
    
    random.seed(42)
    np.random.seed(42)
    
    # Precompute witnesses: for edge (u,v), list of (x,y) such that v-x-y-u is a 4-cycle
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
            witness[(u,v)] = wl
            witness[(v,u)] = wl

    def get_forbidden_and_blocked(adj, u, v):
        """Return (forbidden_set, blocked_flag) for adding edge (u,v)."""
        if u > v:
            u, v = v, u
        wl = witness[(u,v)]
        forbidden = set()
        for x, y in wl:
            c_vx = adj[v, x]
            if c_vx == -1:
                continue
            c_xy = adj[x, y]
            if c_xy == -1:
                continue
            c_yu = adj[y, u]
            if c_yu == -1:
                continue
            
            if c_vx == c_xy or c_vx == c_yu or c_xy == c_yu:
                return forbidden, True
            
            forbidden.add(c_vx)
            forbidden.add(c_xy)
            forbidden.add(c_yu)
        
        return forbidden, False

    def find_valid_color(adj, u, v, prefer_rare=True):
        """Find a valid color for edge (u,v)."""
        forbidden, blocked = get_forbidden_and_blocked(adj, u, v)
        if blocked:
            return -1
        available = [c for c in range(n) if c not in forbidden]
        if not available:
            return -1
        if prefer_rare:
            color_counts = np.zeros(n, dtype=int)
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j] != -1:
                        color_counts[adj[i, j]] += 1
            best_c = min(available, key=lambda c: color_counts[c])
            return best_c
        return random.choice(available)

    def can_recolor_edge_fast(adj, u, v, new_c):
        """Check if recoloring edge (u,v) to new_c keeps all 4-cycles valid."""
        if u > v:
            u, v = v, u
        old_c = adj[u, v]
        if old_c == new_c:
            return True
        
        adj[u, v] = new_c
        adj[v, u] = new_c
        
        valid = True
        wl = witness[(u,v)]
        for x, y in wl:
            c_vx = adj[v, x]
            if c_vx == -1:
                continue
            c_xy = adj[x, y]
            if c_xy == -1:
                continue
            c_yu = adj[y, u]
            if c_yu == -1:
                continue
            
            if new_c == c_vx or new_c == c_xy or new_c == c_yu:
                valid = False
                break
        
        adj[u, v] = old_c
        adj[v, u] = old_c
        
        return valid

    def try_add_with_recolor(adj, u, v, max_recolor_attempts=3):
        """Try to add edge (u,v) by recoloring blocking edges. Returns True if added."""
        if adj[u, v] != -1:
            return True
        
        c = find_valid_color(adj, u, v)
        if c != -1:
            adj[u, v] = c
            adj[v, u] = c
            return True
        
        # Identify blocking cycles and try to recolor edges within them
        if u > v:
            u, v = v, u
        wl = witness[(u,v)]
        
        # Collect all blocking configurations
        blocking_cycles = []
        for x, y in wl:
            c_vx = adj[v, x]
            if c_vx == -1:
                continue
            c_xy = adj[x, y]
            if c_xy == -1:
                continue
            c_yu = adj[y, u]
            if c_yu == -1:
                continue
            
            # This cycle blocks if any two of (c_vx, c_xy, c_yu) are equal
            # OR if the set {c_vx, c_xy, c_yu} covers all available colors
            if c_vx == c_xy or c_vx == c_yu or c_xy == c_yu:
                blocking_cycles.append((x, y, c_vx, c_xy, c_yu))
            else:
                # Check if this cycle alone blocks all colors
                blocked_colors = {c_vx, c_xy, c_yu}
                if len(blocked_colors) >= n:
                    blocking_cycles.append((x, y, c_vx, c_xy, c_yu))
        
        if not blocking_cycles:
            return False
        
        random.shuffle(blocking_cycles)
        
        for x, y, c_vx, c_xy, c_yu in blocking_cycles[:max_recolor_attempts * 2]:
            # Try recoloring (v, x)
            for new_c in range(n):
                if new_c == c_vx:
                    continue
                if can_recolor_edge_fast(adj, v, x, new_c):
                    adj[v, x] = new_c
                    adj[x, v] = new_c
                    c2 = find_valid_color(adj, u, v)
                    if c2 != -1:
                        adj[u, v] = c2
                        adj[v, u] = c2
                        return True
                    else:
                        adj[v, x] = c_vx
                        adj[x, v] = c_vx
            
            # Try recoloring (x, y)
            for new_c in range(n):
                if new_c == c_xy:
                    continue
                if can_recolor_edge_fast(adj, x, y, new_c):
                    adj[x, y] = new_c
                    adj[y, x] = new_c
                    c2 = find_valid_color(adj, u, v)
                    if c2 != -1:
                        adj[u, v] = c2
                        adj[v, u] = c2
                        return True
                    else:
                        adj[x, y] = c_xy
                        adj[y, x] = c_xy
            
            # Try recoloring (y, u)
            for new_c in range(n):
                if new_c == c_yu:
                    continue
                if can_recolor_edge_fast(adj, y, u, new_c):
                    adj[y, u] = new_c
                    adj[u, y] = new_c
                    c2 = find_valid_color(adj, u, v)
                    if c2 != -1:
                        adj[u, v] = c2
                        adj[v, u] = c2
                        return True
                    else:
                        adj[y, u] = c_yu
                        adj[u, y] = c_yu
        
        return False

    def local_search(adj, max_rounds=80):
        """Iteratively try to add missing edges via recoloring. Returns edge count."""
        edges_missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == -1]
        random.shuffle(edges_missing)
        
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            progress = False
            random.shuffle(edges_missing)
            for u, v in edges_missing[:]:
                if adj[u, v] != -1:
                    edges_missing.remove((u, v))
                    continue
                if try_add_with_recolor(adj, u, v):
                    edges_missing.remove((u, v))
                    progress = True
            if not progress:
                break
        
        return np.sum(adj != -1) // 2

    for restart in range(3000):
        if time.time() > deadline:
            break
        
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)
        
        # Build initial spanning tree with random colors
        vertices = list(range(n))
        random.shuffle(vertices)
        for i in range(1, n):
            u = vertices[i]
            v = vertices[random.randint(0, i-1)]
            c = random.randint(0, n-1)
            adj[u, v] = c
            adj[v, u] = c
        
        # Greedy add edges in random order
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            c = find_valid_color(adj, u, v)
            if c != -1:
                adj[u, v] = c
                adj[v, u] = c
        
        # Local search to add more edges
        cnt = local_search(adj)
        
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj.copy()
        
        if best_edges > 175:
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
