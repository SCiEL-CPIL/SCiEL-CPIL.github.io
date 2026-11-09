import numpy as np
import time
import random

def count_edges(adj):
    return np.sum(adj != -1) // 2

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    random.seed(42)
    np.random.seed(42)
    
    # Precompute 4-cycle witnesses for each edge (u,v)
    # witness[(u,v)] = list of (x, y) such that u-v-x-y-u is a 4-cycle
    # Also precompute as numpy arrays for vectorized operations
    witness = {}
    witness_arrays = {}
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
            # Store as numpy array for vectorized access
            if wl:
                arr = np.array(wl, dtype=np.int64)
                witness_arrays[(u, v)] = arr
                witness_arrays[(v, u)] = arr
            else:
                witness_arrays[(u, v)] = np.empty((0, 2), dtype=np.int64)
                witness_arrays[(v, u)] = np.empty((0, 2), dtype=np.int64)
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_forbidden(u, v, adj):
        """
        Returns set of colors forbidden for edge (u,v).
        Vectorized using numpy where possible.
        """
        arr = witness_arrays[(u, v)]
        if len(arr) == 0:
            return set()
        
        x = arr[:, 0]
        y = arr[:, 1]
        
        # Get colors of the three other edges in each 4-cycle
        c1 = adj[v, x]  # edge v-x
        c2 = adj[x, y]  # edge x-y
        c3 = adj[y, u]  # edge y-u
        
        # Check which cycles are complete (all three edges exist)
        complete = (c1 != -1) & (c2 != -1) & (c3 != -1)
        
        if not np.any(complete):
            return set()
        
        c1 = c1[complete]
        c2 = c2[complete]
        c3 = c3[complete]
        
        # Check for duplicate colors in existing cycles (shouldn't happen in valid state)
        if np.any(c1 == c2) or np.any(c1 == c3) or np.any(c2 == c3):
            return set(range(n))
        
        forbidden = set()
        forbidden.update(c1.tolist())
        forbidden.update(c2.tolist())
        forbidden.update(c3.tolist())
        
        return forbidden
    
    def is_recolor_valid(u, v, new_c, adj):
        """Check if recoloring edge (u,v) to new_c keeps all 4-cycles rainbow. Vectorized."""
        arr = witness_arrays[(u, v)]
        if len(arr) == 0:
            return True
        
        x = arr[:, 0]
        y = arr[:, 1]
        
        c1 = adj[v, x]
        c2 = adj[x, y]
        c3 = adj[y, u]
        
        complete = (c1 != -1) & (c2 != -1) & (c3 != -1)
        
        if not np.any(complete):
            return True
        
        c1 = c1[complete]
        c2 = c2[complete]
        c3 = c3[complete]
        
        # Check if new_c conflicts with any existing color
        if np.any(c1 == new_c) or np.any(c2 == new_c) or np.any(c3 == new_c):
            return False
        
        return True
    
    def find_blocking_edge(u, v, adj):
        """
        For a missing edge (u,v), find a specific edge involved in a conflicting 4-cycle.
        """
        forb = get_forbidden(u, v, adj)
        if len(forb) < n:
            return None
            
        arr = witness_arrays[(u, v)]
        if len(arr) == 0:
            return None
        
        x = arr[:, 0]
        y = arr[:, 1]
        
        c1 = adj[v, x]
        c2 = adj[x, y]
        c3 = adj[y, u]
        
        complete = (c1 != -1) & (c2 != -1) & (c3 != -1)
        
        if not np.any(complete):
            return None
        
        x_c = x[complete]
        y_c = y[complete]
        c1_c = c1[complete]
        c2_c = c2[complete]
        c3_c = c3[complete]
        
        # Collect potential blockers
        potential_blockers = []
        for i in range(len(x_c)):
            if c1_c[i] == c2_c[i] or c1_c[i] == c3_c[i] or c2_c[i] == c3_c[i]:
                return (v, int(x_c[i]))
            potential_blockers.append((v, int(x_c[i])))
            potential_blockers.append((int(x_c[i]), int(y_c[i])))
            potential_blockers.append((int(y_c[i]), u))
            if len(potential_blockers) > 20:
                break
        
        if potential_blockers:
            return random.choice(potential_blockers)
            
        return None
    
    def vectorized_color_check(adj):
        """
        Vectorized check of all 4-cycle constraints.
        Returns the number of violations (0 means valid).
        """
        violations = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v] == -1:
                    continue
                arr = witness_arrays[(u, v)]
                if len(arr) == 0:
                    continue
                x = arr[:, 0]
                y = arr[:, 1]
                c1 = adj[v, x]
                c2 = adj[x, y]
                c3 = adj[y, u]
                complete = (c1 != -1) & (c2 != -1) & (c3 != -1)
                if not np.any(complete):
                    continue
                c0 = np.full(np.sum(complete), adj[u, v])
                c1_c = c1[complete]
                c2_c = c2[complete]
                c3_c = c3[complete]
                # Check for duplicates
                if np.any(c0 == c1_c) or np.any(c0 == c2_c) or np.any(c0 == c3_c) or \
                   np.any(c1_c == c2_c) or np.any(c1_c == c3_c) or np.any(c2_c == c3_c):
                    violations += 1
        return violations
    
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
        
        # Local search with vectorized checks
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try adding missing edges
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
            
            # 2. Targeted Recoloring
            if missing:
                u, v = random.choice(missing)
                blocker = find_blocking_edge(u, v, adj)
                if blocker:
                    a, b = blocker
                    old_c = adj[a, b]
                    colors = list(range(n))
                    random.shuffle(colors)
                    for new_c in colors:
                        if new_c == old_c:
                            continue
                        if is_recolor_valid(a, b, new_c, adj):
                            adj[a, b] = new_c
                            adj[b, a] = new_c
                            improved = True
                            # Immediately try to add the edge that was blocked
                            if adj[u, v] == -1:
                                forb = get_forbidden(u, v, adj)
                                if len(forb) < n:
                                    available = [c for c in range(n) if c not in forb]
                                    if available:
                                        c_add = random.choice(available)
                                        adj[u, v] = c_add
                                        adj[v, u] = c_add
                            break
            
            # 3. Random Recoloring (Escape local optima) - vectorized batch
            if not improved:
                existing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] != -1]
                random.shuffle(existing)
                
                # Try recoloring a batch of edges simultaneously
                batch_size = min(30, len(existing))
                for u, v in existing[:batch_size]:
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
                # Stuck in this restart
                break
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj.copy()
        
        if best_count > 160:
            break
    
    # Final polish with more aggressive vectorized recoloring
    if best_adj is not None and time.time() < deadline - 5:
        adj = best_adj.copy()
        for polish_iter in range(2000):
            if time.time() > deadline - 2:
                break
            
            improved = False
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == -1]
            random.shuffle(missing)
            
            # Try adding
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
                # Targeted recoloring on best solution
                if missing:
                    u, v = random.choice(missing)
                    blocker = find_blocking_edge(u, v, adj)
                    if blocker:
                        a, b = blocker
                        old_c = adj[a, b]
                        colors = list(range(n))
                        random.shuffle(colors)
                        for new_c in colors:
                            if new_c == old_c:
                                continue
                            if is_recolor_valid(a, b, new_c, adj):
                                adj[a, b] = new_c
                                adj[b, a] = new_c
                                improved = True
                                # Try adding again
                                if adj[u, v] == -1:
                                    forb = get_forbidden(u, v, adj)
                                    if len(forb) < n:
                                        available = [c for c in range(n) if c not in forb]
                                        if available:
                                            c_add = random.choice(available)
                                            adj[u, v] = c_add
                                            adj[v, u] = c_add
                                break
            
            if not improved:
                # More aggressive random recoloring
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
