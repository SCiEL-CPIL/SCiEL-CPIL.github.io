import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Increased deadline to 135s (15s safety margin)
    n = 20
    num_colors = 20
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    m = len(all_edges)
    
    best_count = 0
    best_edges = []
    
    def greedy_build(seed, init_edges=None, order=None):
        rng = random.Random(seed)
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        color_counts = [0] * num_colors
        
        if init_edges:
            for u, v, c in init_edges:
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
                color_counts[c] += 1
            existing = set((min(u,v), max(u,v)) for u, v, c in edges)
            remaining = [e for e in all_edges if e not in existing]
        else:
            remaining = all_edges[:]
            
        if order:
            # Filter remaining to match order
            order_set = set(order)
            remaining = [e for e in remaining if e in order_set]
            # Add any edges not in order (should be none if order is all_edges)
            # But if order is a subset, we might miss some. 
            # Let's assume order is a permutation of all_edges or a subset.
            # If subset, we append the rest shuffled.
            remaining_set = set(remaining)
            rest = [e for e in all_edges if e not in remaining_set and e not in existing]
            rng.shuffle(rest)
            remaining = remaining + rest
        else:
            rng.shuffle(remaining)
        
        for (u, v) in remaining:
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
                        # Choose least used color among candidates
                        c = min(candidates, key=lambda x: color_counts[x])
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
                        color_counts[c] += 1
                else:
                    # Choose least used color overall
                    c = min(range(num_colors), key=lambda x: color_counts[x])
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v, c])
                    color_counts[c] += 1
        
        return edges
    
    def check_validity(edges):
        """Verify all 4-cycles are rainbow."""
        adj = [[-1] * n for _ in range(n)]
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
                        if len({c1, c2, c3, c4}) != 4:
                            return False
        return True
    
    def local_search_improve(edges, seed, time_limit):
        """Local search: remove k edges, refill greedily."""
        rng = random.Random(seed)
        current_edges = edges[:]
        current_count = len(current_edges)
        
        while time.time() < time_limit:
            # Randomly remove k edges
            k = rng.randint(1, 5)
            if current_count <= k:
                break
            
            to_remove = rng.sample(current_edges, k)
            rem_set = set((min(u,v), max(u,v)) for u, v, c in to_remove)
            base = [e for e in current_edges if (min(e[0], e[1]), max(e[0], e[1])) not in rem_set]
            
            # Refill greedily
            new_edges = greedy_build(rng.randint(0, 10**9), base)
            
            if len(new_edges) > current_count:
                current_edges = new_edges
                current_count = len(new_edges)
                
            # Occasionally try a full restart with current best as seed?
            # No, just keep improving locally.
            
        return current_edges
    
    # Phase 1: Greedy random restarts with Bipartite Bias
    # We will run several types of restarts:
    # 1. Standard random
    # 2. Bipartite-First (Cross edges prioritized)
    
    num_restarts = 400
    for i in range(num_restarts):
        if time.time() > deadline - 20: # Reserve time for local search
            break
        
        if i % 2 == 0:
            # Standard random
            edges = greedy_build(i)
        else:
            # Bipartite-First
            rng = random.Random(i)
            verts = list(range(n))
            rng.shuffle(verts)
            V1 = set(verts[:10])
            V2 = set(verts[10:])
            
            cross_edges = [(u, v) for u in V1 for v in V2 if u < v]
            intra_edges = [(u, v) for u in range(n) for v in range(u+1, n) if not (u in V1 and v in V2)]
            
            rng.shuffle(cross_edges)
            rng.shuffle(intra_edges)
            order = cross_edges + intra_edges
            
            edges = greedy_build(i, order=order)
            
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    # Phase 2: Local search on the best solution found
    if time.time() < deadline - 5:
        ls_limit = min(deadline - 2, time.time() + 30)
        improved = local_search_improve(best_edges[:], 42, ls_limit)
        if len(improved) > best_count:
            best_count = len(improved)
            best_edges = improved[:]
    
    # Final verification - ensure the result is valid
    if best_edges and not check_validity(best_edges):
        # Find a valid fallback
        best_edges = []
        for i in range(50):
            candidate = greedy_build(1000 + i)
            if check_validity(candidate):
                best_edges = candidate
                break
        if not best_edges:
            best_edges = [[0, v, v % 20] for v in range(1, 20)]
    
    if not best_edges:
        best_edges = [[0, v, v % 20] for v in range(1, 20)]
    
    return best_edges
