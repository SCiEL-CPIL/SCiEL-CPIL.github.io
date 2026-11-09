import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 40
    target_edges = 401
    
    a_size = 9
    b_size = 31
    
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    needed_in_B = target_edges - (a_size * b_size)  # 122
    
    best_score = float('inf')
    best_edges = None
    
    # Try multiple random seeds for diversity
    for seed in range(50):
        if time.time() > deadline - 10:
            break
            
        random.seed(seed)
        
        # Build B edges with triangle-free constraint and degree limit
        B_edges = []
        degree = {v: 0 for v in B}
        adj = {v: set() for v in B}
        
        # Generate all possible pairs, shuffled
        pairs = list(itertools.combinations(B, 2))
        random.shuffle(pairs)
        
        # Try max degree 8 first
        for u, v in pairs:
            if len(B_edges) == needed_in_B:
                break
            common = adj[u] & adj[v]
            if not common:
                if degree[u] < 8 and degree[v] < 8:
                    B_edges.append((u, v))
                    degree[u] += 1
                    degree[v] += 1
                    adj[u].add(v)
                    adj[v].add(u)
        
        # If we couldn't reach 122 with max degree 8, try with max degree 9
        if len(B_edges) < needed_in_B:
            B_edges = []
            degree = {v: 0 for v in B}
            adj = {v: set() for v in B}
            
            for u, v in pairs:
                if len(B_edges) == needed_in_B:
                    break
                common = adj[u] & adj[v]
                if not common:
                    if degree[u] < 9 and degree[v] < 9:
                        B_edges.append((u, v))
                        degree[u] += 1
                        degree[v] += 1
                        adj[u].add(v)
                        adj[v].add(u)
        
        # If still not enough, relax further
        if len(B_edges) < needed_in_B:
            B_edges = []
            degree = {v: 0 for v in B}
            adj = {v: set() for v in B}
            
            for u, v in pairs:
                if len(B_edges) == needed_in_B:
                    break
                common = adj[u] & adj[v]
                if not common:
                    if degree[u] < 10 and degree[v] < 10:
                        B_edges.append((u, v))
                        degree[u] += 1
                        degree[v] += 1
                        adj[u].add(v)
                        adj[v].add(u)
        
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
        
        # Compute current score (max book size)
        # For each edge, count triangles containing it
        adj_full = {v: set() for v in range(n)}
        for u, v in edges:
            adj_full[u].add(v)
            adj_full[v].add(u)
        
        score = 0
        for u, v in edges:
            common = adj_full[u] & adj_full[v]
            book_size = len(common)
            if book_size > score:
                score = book_size
        
        if score < best_score:
            best_score = score
            best_edges = set(edges)
            
            # If we've achieved the theoretical lower bound, stop
            if score <= 7:
                break
        
        # Local search: try to improve by swapping edges
        if time.time() > deadline - 15:
            break
        
        # Try removing one B edge and adding another to reduce max book
        current_B_edges = set(B_edges)
        current_degree = dict(degree)
        current_adj = {v: set(adj[v]) for v in B}
        
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            
            # Find the edge with maximum book size
            max_book_edge = None
            max_book = 0
            for u, v in edges:
                common = adj_full[u] & adj_full[v]
                if len(common) > max_book:
                    max_book = len(common)
                    max_book_edge = (u, v)
            
            if max_book_edge is None:
                break
            
            u, v = max_book_edge
            
            # Try to remove this edge and add a different one
            # Only consider B-B edges for removal (A-B edges are fixed by structure)
            if u < a_size or v < a_size:
                # This is an A-B edge, can't remove it
                # Try to reduce book by modifying B edges instead
                # For A-B edge (a,b), book size = deg_B(b)
                # So we need to reduce deg_B(b)
                b_vertex = v if u < a_size else u
                a_vertex = u if u < a_size else v
                
                # Try to find a B edge incident to b_vertex that we can remove
                # and replace with a non-incident edge
                removed = False
                for (bu, bv) in list(current_B_edges):
                    if bu != b_vertex and bv != b_vertex:
                        continue
                    
                    # Try removing this edge
                    new_B_edges = current_B_edges - {(bu, bv)}
                    
                    # Try adding a different B edge
                    # Find candidates that don't create triangles
                    candidates = []
                    for (cu, cv) in itertools.combinations(B, 2):
                        if (cu, cv) in new_B_edges or (cv, cu) in new_B_edges:
                            continue
                        # Check if adding (cu,cv) creates triangle
                        common_c = (current_adj[cu] & current_adj[cv])
                        # But we removed (bu,bv), so adj changed
                        # This is complex; skip for now
                        pass
                    
                    # Simpler: just try to swap with a random edge
                    # For now, break out of local search for A-B edges
                    break
                
                continue
            
            # It's a B-B edge, try to remove and replace
            # Remove (u,v) from B edges
            test_B_edges = current_B_edges - {(u, v)}
            
            # Try adding a different B edge that reduces the max book
            # We want to add an edge (x,y) where x,y are in B
            # and the new edge doesn't create triangles
            added = False
            for (x, y) in itertools.combinations(B, 2):
                if (x, y) in test_B_edges or (y, x) in test_B_edges:
                    continue
                if (x, y) == (u, v) or (x, y) == (v, u):
                    continue
                
                # Check if (x,y) can be added without creating triangles
                # Common neighbors of x and y in the graph with (u,v) removed
                adj_test = {v: set(adj_full[v]) for v in B}
                adj_test[u].discard(v)
                adj_test[v].discard(u)
                
                common_xy = adj_test[x] & adj_test[y]
                if common_xy:
                    continue
                
                # Check degree constraints (keep max degree reasonable)
                deg_x = len(adj_test[x])
                deg_y = len(adj_test[y])
                if deg_x >= 10 or deg_y >= 10:
                    continue
                
                # Add (x,y) and compute new score
                test_edges = set()
                for a in A:
                    for b in B:
                        test_edges.add((a, b))
                for eu, ev in test_B_edges:
                    test_edges.add((eu, ev))
                test_edges.add((x, y))
                
                if len(test_edges) != target_edges:
                    continue
                
                # Compute new score
                adj_test_full = {v: set() for v in range(n)}
                for eu, ev in test_edges:
                    adj_test_full[eu].add(ev)
                    adj_test_full[ev].add(eu)
                
                new_score = 0
                for eu, ev in test_edges:
                    common_e = adj_test_full[eu] & adj_test_full[ev]
                    if len(common_e) > new_score:
                        new_score = len(common_e)
                
                if new_score < max_book:
                    # Accept this swap
                    current_B_edges = test_B_edges | {(x, y)}
                    edges = test_edges
                    adj_full = adj_test_full
                    score = new_score
                    improved = True
                    added = True
                    break
            
            if not added:
                break
        
        if score < best_score:
            best_score = score
            best_edges = set(edges)
            if score <= 7:
                break
    
    # If no valid solution found, fall back to a simple construction
    if best_edges is None:
        edges = set()
        for a in A:
            for b in B:
                edges.add((a, b))
        
        # Add 122 edges within B
        B_pairs = list(itertools.combinations(B, 2))
        random.shuffle(B_pairs)
        for i, (u, v) in enumerate(B_pairs):
            if i < needed_in_B:
                edges.add((u, v))
        
        best_edges = edges
    
    result = [[u, v] for u, v in best_edges]
    return result
