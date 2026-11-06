import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    n = 20
    num_colors = 20
    
    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m = len(all_edges)
    
    best_count = 0
    best_edges = []
    
    def check_validity_fast(edges):
        """Verify all 4-cycles are rainbow using adjacency matrix."""
        adj = np.full((n, n), -1, dtype=np.int8)
        for u, v, c in edges:
            adj[u][v] = c
            adj[v][u] = c
        
        for u in range(n):
            for v in range(n):
                if u == v or adj[u][v] == -1:
                    continue
                for w in range(n):
                    if w == u or w == v or adj[v][w] == -1:
                        continue
                    for x in range(n):
                        if x == u or x == v or x == w:
                            continue
                        if adj[w][x] == -1 or adj[x][u] == -1:
                            continue
                        c1 = adj[u][v]
                        c2 = adj[v][w]
                        c3 = adj[w][x]
                        c4 = adj[x][u]
                        if c1 == c2 or c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4 or c3 == c4:
                            return False
        return True
    
    def greedy_build(seed, init_edges=None, time_limit=None):
        """Greedy edge addition with optional initial edges and time limit."""
        rng = random.Random(seed)
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        if init_edges:
            for u, v, c in init_edges:
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
            existing = set((min(u, v), max(u, v)) for u, v, c in edges)
            remaining = [e for e in all_edges if e not in existing]
        else:
            remaining = all_edges[:]
        
        rng.shuffle(remaining)
        
        for (u, v) in remaining:
            if time_limit and time.time() > time_limit:
                break
            
            forbidden = set()
            feasible = True
            
            # Check all 4-cycles that would be created by adding (u,v)
            for a in neighbors[u]:
                if a == v:
                    continue
                ca = adj[u][a]
                for b in neighbors[v]:
                    if b == u or b == a:
                        continue
                    if adj[a][b] != -1:
                        cb = adj[a][b]
                        cv = adj[b][v]
                        # 4-cycle: a-u-v-b
                        if ca == cb or cb == cv or ca == cv:
                            feasible = False
                            break
                        forbidden.add(ca)
                        forbidden.add(cb)
                        forbidden.add(cv)
                if not feasible:
                    break
            
            if feasible:
                if forbidden:
                    candidates = [c for c in range(num_colors) if c not in forbidden]
                    if candidates:
                        c = rng.choice(candidates)
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
                else:
                    c = rng.randrange(num_colors)
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v, c])
        
        return edges
    
    def bipartite_seed(seed):
        """Create a bipartite graph seed to avoid C4-free local optima."""
        rng = random.Random(seed)
        group_a = list(range(10))
        group_b = list(range(10, 20))
        
        edges = []
        # Add cross edges with distinct colors, moderate density
        for a in group_a:
            for b in group_b:
                if rng.random() < 0.35:
                    c = rng.randrange(num_colors)
                    edges.append([a, b, c])
        
        return edges
    
    def try_add_edge(adj, neighbors, u, v, rng):
        """Try to add edge (u,v) with a valid color. Returns True if added."""
        forbidden = set()
        feasible = True
        
        for a in neighbors[u]:
            if a == v:
                continue
            ca = adj[u][a]
            for b in neighbors[v]:
                if b == u or b == a:
                    continue
                if adj[a][b] != -1:
                    cb = adj[a][b]
                    cv = adj[b][v]
                    if ca == cb or cb == cv or ca == cv:
                        return False
                    forbidden.add(ca)
                    forbidden.add(cb)
                    forbidden.add(cv)
        
        if forbidden:
            candidates = [c for c in range(num_colors) if c not in forbidden]
            if not candidates:
                return False
            c = rng.choice(candidates)
        else:
            c = rng.randrange(num_colors)
        
        adj[u][v] = c
        adj[v][u] = c
        neighbors[u].add(v)
        neighbors[v].add(u)
        return True
    
    def recolor_search(edges, seed, time_limit):
        """Try to improve by re-coloring existing edges to unlock new additions."""
        rng = random.Random(seed)
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        for u, v, c in edges:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try re-coloring each edge
            for idx in range(len(edges)):
                if time.time() > time_limit:
                    break
                u, v, old_c = edges[idx]
                
                for new_c in range(num_colors):
                    if new_c == old_c:
                        continue
                    
                    # Check if re-coloring is valid
                    valid = True
                    for a in neighbors[u]:
                        if a == v:
                            continue
                        ca = adj[u][a]
                        for b in neighbors[v]:
                            if b == u or b == a:
                                continue
                            if adj[a][b] != -1:
                                cb = adj[a][b]
                                cv = adj[b][v]
                                if ca == new_c or new_c == cv or cv == cb or ca == cb:
                                    valid = False
                                    break
                        if not valid:
                            break
                    
                    if not valid:
                        continue
                    
                    # Re-color is valid; check if it allows adding a new edge
                    added = False
                    for (wu, wv) in all_edges:
                        if adj[wu][wv] != -1:
                            continue
                        
                        forbidden_add = set()
                        feasible_add = True
                        for a in neighbors[wu]:
                            if a == wv:
                                continue
                            ca = adj[wu][a]
                            for b in neighbors[wv]:
                                if b == wu or b == a:
                                    continue
                                if adj[a][b] != -1:
                                    cb = adj[a][b]
                                    cv = adj[b][wv]
                                    if ca == cb or cb == cv or ca == cv:
                                        feasible_add = False
                                        break
                                    forbidden_add.add(ca)
                                    forbidden_add.add(cb)
                                    forbidden_add.add(cv)
                            if not feasible_add:
                                break
                        
                        if not feasible_add:
                            continue
                        
                        candidates = [c for c in range(num_colors) if c not in forbidden_add]
                        if candidates:
                            nc = rng.choice(candidates)
                            # Apply re-coloring
                            adj[u][v] = new_c
                            adj[v][u] = new_c
                            # Apply new edge
                            adj[wu][wv] = nc
                            adj[wv][wu] = nc
                            neighbors[wu].add(wv)
                            neighbors[wv].add(wu)
                            edges[idx] = [u, v, new_c]
                            edges.append([wu, wv, nc])
                            improved = True
                            added = True
                            break
                    
                    if added:
                        break
                
                if added:
                    break
            
            # After re-coloring, try to add more edges greedily
            if improved:
                existing = set((min(u, v), max(u, v)) for u, v, c in edges)
                remaining = [e for e in all_edges if e not in existing]
                rng.shuffle(remaining)
                for (u, v) in remaining:
                    if time.time() > time_limit:
                        break
                    if try_add_edge(adj, neighbors, u, v, rng):
                        edges.append([u, v, adj[u][v]])
        
        return edges
    
    # Phase 1: Many greedy restarts with different seeds, including bipartite seeds
    num_restarts = 3000
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        if i % 5 == 0:
            # Bipartite seed every 5th restart
            init = bipartite_seed(i)
            edges = greedy_build(i, init, deadline - 30)
        else:
            edges = greedy_build(i, None, deadline - 30)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    # Phase 2: Local search - remove k edges, refill
    current_best = best_edges[:]
    
    for restart_round in range(80):
        if time.time() > deadline - 20:
            break
        
        if len(current_best) > 5:
            k = random.randint(2, 12)
            to_remove = random.sample(current_best, min(k, len(current_best)))
            rem_set = set((min(u, v), max(u, v)) for u, v, c in to_remove)
            base = [e for e in current_best if (min(e[0], e[1]), max(e[0], e[1])) not in rem_set]
            
            seed = random.randint(0, 10**9)
            new_edges = greedy_build(seed, base, deadline - 20)
            
            if len(new_edges) > best_count:
                best_count = len(new_edges)
                best_edges = new_edges[:]
                current_best = new_edges[:]
    
    # Phase 3: Re-coloring search
    if time.time() < deadline - 10:
        recolored = recolor_search(current_best[:], 42, deadline - 5)
        if len(recolored) > best_count:
            best_count = len(recolored)
            best_edges = recolored[:]
            current_best = recolored[:]
    
    # Final verification
    if best_edges and not check_validity_fast(best_edges):
        # Find a valid fallback
        best_edges = []
        for i in range(100):
            candidate = greedy_build(1000 + i)
            if check_validity_fast(candidate):
                best_edges = candidate
                break
        if not best_edges:
            best_edges = [[0, v, v % 20] for v in range(1, 20)]
    
    if not best_edges:
        best_edges = [[0, v, v % 20] for v in range(1, 20)]
    
    return best_edges
