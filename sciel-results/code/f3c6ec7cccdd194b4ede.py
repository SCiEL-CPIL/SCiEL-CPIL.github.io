import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    target_edges = 401
    
    a_size = 9
    b_size = 31
    
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    needed_in_B = target_edges - (a_size * b_size)  # 122
    
    best_score = float('inf')
    best_edges = None
    
    # Precompute all B-B pairs
    all_B_pairs = list(itertools.combinations(B, 2))
    
    def compute_score(edges, adj_full):
        score = 0
        for u, v in edges:
            common = adj_full[u] & adj_full[v]
            if len(common) > score:
                score = len(common)
        return score
    
    def build_adj_full(edges):
        adj_full = {v: set() for v in range(n)}
        for u, v in edges:
            adj_full[u].add(v)
            adj_full[v].add(u)
        return adj_full
    
    def greedy_B_edges(pairs, max_deg, target_count):
        """Greedy: pick edge that minimizes max resulting degree and avoids triangles."""
        B_edges = []
        degree = {v: 0 for v in B}
        adj = {v: set() for v in B}
        
        for u, v in pairs:
            if len(B_edges) == target_count:
                break
            common = adj[u] & adj[v]
            if not common:
                if degree[u] < max_deg and degree[v] < max_deg:
                    B_edges.append((u, v))
                    degree[u] += 1
                    degree[v] += 1
                    adj[u].add(v)
                    adj[v].add(u)
        
        return B_edges, degree, adj
    
    def local_search(edges, adj_full, current_score, max_iters=200):
        """Try to reduce max book by swapping B-B edges."""
        current_edges = set(edges)
        current_adj = {v: set(adj_full[v]) for v in range(n)}
        cur_score = current_score
        b_edge_set = set((u, v) for u, v in current_edges if u >= a_size and v >= a_size)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Find edge with max book
            max_book = 0
            max_edge = None
            for u, v in current_edges:
                common = current_adj[u] & current_adj[v]
                if len(common) > max_book:
                    max_book = len(common)
                    max_edge = (u, v)
            
            if max_edge is None or max_book <= 6:
                break
            
            u, v = max_edge
            
            # If it's a B-B edge, try swapping it
            if u >= a_size and v >= a_size:
                # Remove (u,v)
                test_b_edges = b_edge_set - {(u, v)}
                
                # Try adding another B-B edge
                for (x, y) in all_B_pairs:
                    if (x, y) in test_b_edges or (y, x) in test_b_edges:
                        continue
                    if (x, y) == (u, v) or (x, y) == (v, u):
                        continue
                    
                    # Check no triangle created
                    # x and y must not share a common neighbor in B (excluding the removed edge)
                    common_xy = set()
                    for nb in current_adj[x]:
                        if nb >= a_size and nb in current_adj[y]:
                            common_xy.add(nb)
                    # Also check if removing (u,v) helps
                    # If u==x or u==y or v==x or v==y, the removed edge might break a triangle
                    if u == x or u == y or v == x or v == y:
                        # Re-check after removal
                        common_xy = set()
                        for nb in current_adj[x]:
                            if nb >= a_size and nb in current_adj[y] and nb not in (u, v):
                                common_xy.add(nb)
                        # Also need to remove u/v from consideration if they were common neighbors
                        if u in common_xy and (u, v) not in current_edges:  # already removed
                            common_xy.discard(u)
                        if v in common_xy:
                            common_xy.discard(v)
                    
                    if common_xy:
                        continue
                    
                    # Check degree balance
                    deg_x = len([nb for nb in current_adj[x] if nb >= a_size])
                    deg_y = len([nb for nb in current_adj[y] if nb >= a_size])
                    if deg_x >= 10 or deg_y >= 10:
                        continue
                    
                    # Compute new score incrementally
                    # Remove (u,v), add (x,y)
                    # Only edges incident to u, v, x, y have changed book sizes
                    affected = set()
                    for nb in current_adj[u]:
                        affected.add((min(u, nb), max(u, nb)))
                    for nb in current_adj[v]:
                        affected.add((min(v, nb), max(v, nb)))
                    for nb in current_adj[x]:
                        affected.add((min(x, nb), max(x, nb)))
                    for nb in current_adj[y]:
                        affected.add((min(y, nb), max(y, nb)))
                    affected.add((u, v))
                    affected.add((x, y))
                    
                    new_score = 0
                    for eu, ev in current_edges:
                        if (min(eu, ev), max(eu, ev)) == (u, v):
                            continue  # removed
                        if (min(eu, ev), max(eu, ev)) == (x, y):
                            continue  # will be added
                        # For unaffected edges, book size unchanged
                        if (min(eu, ev), max(eu, ev)) not in affected:
                            common = current_adj[eu] & current_adj[ev]
                            if len(common) > new_score:
                                new_score = len(common)
                    
                    # Check added edge (x,y)
                    common_xy_new = current_adj[x] & current_adj[y]
                    if u in common_xy_new:
                        common_xy_new.discard(u)
                    if v in common_xy_new:
                        common_xy_new.discard(v)
                    if len(common_xy_new) > new_score:
                        new_score = len(common_xy_new)
                    
                    if new_score < cur_score:
                        # Accept
                        current_edges.discard((u, v))
                        current_edges.add((x, y))
                        current_adj[u].discard(v)
                        current_adj[v].discard(u)
                        current_adj[x].add(y)
                        current_adj[y].add(x)
                        b_edge_set = set((min(a, b), max(a, b)) for a, b in current_edges if a >= a_size and b >= a_size)
                        cur_score = new_score
                        break
                else:
                    break  # no improvement found for this edge
            
            # If it's an A-B edge, try to reduce deg_B of the B vertex
            else:
                b_vertex = v if u < a_size else u
                # Find B-B edges incident to b_vertex
                incident = [(bu, bv) for (bu, bv) in b_edge_set if bu == b_vertex or bv == b_vertex]
                
                improved = False
                for (bu, bv) in incident:
                    other = bv if bu == b_vertex else bu
                    # Try removing (b_vertex, other)
                    test_b_edges = b_edge_set - {(b_vertex, other)}
                    
                    # Try adding a B-B edge not incident to b_vertex
                    for (x, y) in all_B_pairs:
                        if x == b_vertex or y == b_vertex:
                            continue
                        if (x, y) in test_b_edges or (y, x) in test_b_edges:
                            continue
                        
                        # Check no triangle
                        common_xy = current_adj[x] & current_adj[y]
                        common_xy = {nb for nb in common_xy if nb >= a_size}
                        if common_xy:
                            continue
                        
                        deg_x = len([nb for nb in current_adj[x] if nb >= a_size])
                        deg_y = len([nb for nb in current_adj[y] if nb >= a_size])
                        if deg_x >= 10 or deg_y >= 10:
                            continue
                        
                        # Quick score check
                        new_score = 0
                        for eu, ev in current_edges:
                            if (min(eu, ev), max(eu, ev)) == (min(b_vertex, other), max(b_vertex, other)):
                                continue
                            if (min(eu, ev), max(eu, ev)) == (x, y):
                                continue
                            common = current_adj[eu] & current_adj[ev]
                            if len(common) > new_score:
                                new_score = len(common)
                        
                        common_xy_new = current_adj[x] & current_adj[y]
                        common_xy_new = {nb for nb in common_xy_new if nb >= a_size}
                        if len(common_xy_new) > new_score:
                            new_score = len(common_xy_new)
                        
                        if new_score < cur_score:
                            current_edges.discard((b_vertex, other))
                            current_edges.add((x, y))
                            current_adj[b_vertex].discard(other)
                            current_adj[other].discard(b_vertex)
                            current_adj[x].add(y)
                            current_adj[y].add(x)
                            b_edge_set = set((min(a, b), max(a, b)) for a, b in current_edges if a >= a_size and b >= a_size)
                            cur_score = new_score
                            improved = True
                            break
                    
                    if improved:
                        break
                
                if not improved:
                    break
        
        return list(current_edges), cur_score
    
    # Try multiple random seeds
    for seed in range(100):
        if time.time() > deadline - 8:
            break
        
        random.seed(seed)
        
        # Shuffle pairs for diversity
        pairs = all_B_pairs[:]
        random.shuffle(pairs)
        
        # Try different max degree limits
        found = False
        for max_deg in [8, 9, 10]:
            B_edges, degree, adj = greedy_B_edges(pairs, max_deg, needed_in_B)
            
            if len(B_edges) < needed_in_B:
                continue
            
            # Build full edge set
            edges = set()
            for a in A:
                for b in B:
                    edges.add((a, b))
            for u, v in B_edges:
                edges.add((u, v))
            
            if len(edges) != target_edges:
                continue
            
            adj_full = build_adj_full(edges)
            score = compute_score(edges, adj_full)
            
            if score < best_score:
                best_score = score
                best_edges = set(edges)
                if score <= 6:
                    break
            
            # Local search refinement
            if time.time() < deadline - 10:
                refined_edges, refined_score = local_search(edges, adj_full, score, max_iters=100)
                if refined_score < best_score:
                    best_score = refined_score
                    best_edges = set(refined_edges)
                    if score <= 6:
                        break
            
            found = True
            if best_score <= 6:
                break
        
        if best_score <= 6:
            break
    
    # If no valid solution found, fall back
    if best_edges is None:
        edges = set()
        for a in A:
            for b in B:
                edges.add((a, b))
        
        random.shuffle(all_B_pairs)
        for i, (u, v) in enumerate(all_B_pairs):
            if i < needed_in_B:
                edges.add((u, v))
        
        best_edges = edges
    
    result = [[u, v] for u, v in best_edges]
    return result
