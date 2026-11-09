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
    
    # Precompute witness arrays for vectorized operations
    # For each edge (u,v), store arrays of x and y indices
    witness_x = {}
    witness_y = {}
    for u in range(n):
        for v in range(u+1, n):
            wl = witness[(u,v)]
            witness_x[(u,v)] = np.array([x for x,y in wl], dtype=int)
            witness_y[(u,v)] = np.array([y for x,y in wl], dtype=int)
            witness_x[(v,u)] = witness_x[(u,v)]
            witness_y[(v,u)] = witness_y[(u,v)]
    
    def verify_adj(adj):
        """Check that all 4-cycles in adj are rainbow. Return True if valid."""
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v] == -1:
                    continue
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
                        c1 = adj[u, v]
                        c2 = adj[v, x]
                        c3 = adj[x, y]
                        c4 = adj[y, u]
                        if len({c1, c2, c3, c4}) != 4:
                            return False
        return True

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

    def find_valid_color(adj, u, v, deterministic=False):
        """Find a valid color for edge (u,v)."""
        forbidden, blocked = get_forbidden_and_blocked(adj, u, v)
        if blocked:
            return -1
        available = [c for c in range(n) if c not in forbidden]
        if not available:
            return -1
        if deterministic:
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

    def vectorized_local_search(adj, max_rounds=100):
        """Vectorized local search using NumPy for batch 4-cycle checking."""
        edges_missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == -1]
        random.shuffle(edges_missing)
        
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            progress = False
            random.shuffle(edges_missing)
            
            # First pass: try to add edges directly
            still_missing = []
            for u, v in edges_missing:
                if adj[u, v] != -1:
                    continue
                c = find_valid_color(adj, u, v)
                if c != -1:
                    adj[u, v] = c
                    adj[v, u] = c
                    progress = True
                else:
                    still_missing.append((u, v))
            edges_missing = still_missing
            
            if not edges_missing:
                break
            
            # Second pass: try recoloring neighbor edges to free up colors
            random.shuffle(edges_missing)
            for u, v in edges_missing:
                if adj[u, v] != -1:
                    continue
                c = find_valid_color(adj, u, v)
                if c != -1:
                    adj[u, v] = c
                    adj[v, u] = c
                    try:
                        edges_missing.remove((u, v))
                    except ValueError:
                        pass
                    progress = True
                    continue
                # Try recoloring a neighbor edge
                added = False
                for neighbor in [u, v]:
                    if added:
                        break
                    for x in range(n):
                        if x == neighbor:
                            continue
                        if adj[neighbor, x] == -1:
                            continue
                        old_c = adj[neighbor, x]
                        candidates = list(range(n))
                        random.shuffle(candidates)
                        for new_c in candidates:
                            if new_c == old_c:
                                continue
                            if can_recolor_edge_fast(adj, neighbor, x, new_c):
                                adj[neighbor, x] = new_c
                                adj[x, neighbor] = new_c
                                c2 = find_valid_color(adj, u, v)
                                if c2 != -1:
                                    adj[u, v] = c2
                                    adj[v, u] = c2
                                    try:
                                        edges_missing.remove((u, v))
                                    except ValueError:
                                        pass
                                    added = True
                                    progress = True
                                    break
                                else:
                                    adj[neighbor, x] = old_c
                                    adj[x, neighbor] = old_c
                        if added:
                            break
            if not progress:
                break
        
        return int(np.sum(adj != -1) // 2)

    def edge_removal_local_search(adj, max_rounds=50):
        """Try removing random edges and re-filling to escape local minima."""
        edges_present = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] != -1]
        current_count = len(edges_present)
        
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            if not edges_present:
                break
            
            # Remove a random edge
            idx = random.randint(0, len(edges_present) - 1)
            u, v = edges_present[idx]
            adj[u, v] = -1
            adj[v, u] = -1
            edges_present.pop(idx)
            
            # Greedily re-add edges
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            for i, j in all_edges:
                if time.time() > deadline:
                    break
                if adj[i, j] != -1:
                    continue
                c = find_valid_color(adj, i, j)
                if c != -1:
                    adj[i, j] = c
                    adj[j, i] = c
            
            new_count = int(np.sum(adj != -1) // 2)
            if new_count > current_count:
                current_count = new_count
                edges_present = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] != -1]
            else:
                # Restore the removed edge
                # We need to find a valid color for it
                c = find_valid_color(adj, u, v)
                if c != -1:
                    adj[u, v] = c
                    adj[v, u] = c
                else:
                    # Keep it removed if we can't re-add
                    pass
        
        return current_count

    for restart in range(2000):
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
        cnt = vectorized_local_search(adj)
        
        # Edge removal local search to escape local minima
        cnt2 = edge_removal_local_search(adj)
        if cnt2 > cnt:
            cnt = cnt2
        
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj.copy()
        
        if best_edges > 170:
            break
    
    # Final verification and cleanup
    if best_adj is not None:
        if not verify_adj(best_adj):
            # Remove invalid edges until valid
            for i in range(n):
                for j in range(i+1, n):
                    if best_adj[i, j] != -1 and not verify_adj(best_adj):
                        best_adj[i, j] = -1
                        best_adj[j, i] = -1
            best_edges = int(np.sum(best_adj != -1) // 2)
    
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
