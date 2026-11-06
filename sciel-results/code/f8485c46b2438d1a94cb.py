import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 40
    m = 401
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_edges_set = set(all_edges)
    
    # Helper: compute book sizes for all edges in a given adjacency matrix
    def compute_books(adj):
        # C = A^2, C[u,v] = number of common neighbors of u and v
        C = adj @ adj
        books = {}
        for u, v in all_edges:
            if adj[u, v]:
                books[(u, v)] = C[u, v]
        return books, C
    
    # Helper: get max book and sum of squares
    def get_stats(books):
        if not books:
            return 0, 0
        max_book = max(books.values())
        sum_sq = sum(b*b for b in books.values())
        return max_book, sum_sq
    
    # Initial construction: start with a balanced bipartite graph K20,20
    # This has 400 edges and 0 triangles, so max book = 0 for all edges.
    # We need 1 more edge. Adding any edge within a part creates triangles.
    # To minimize the max book, we should add an edge that creates few triangles.
    # In K20,20, adding edge (u,v) within part A creates 20 triangles (each w in B forms triangle u-v-w).
    # So max book = 20.
    
    # Better strategy: start with a random graph and use local search.
    # But let's try a structured start first.
    
    # Alternative: Use a "blow-up" of a small graph.
    # Or, use spectral methods to find a good partition.
    
    # Let's use a random initialization with a fixed seed for reproducibility,
    # then use a sophisticated local search.
    
    rng = np.random.default_rng(12345)
    
    # Start with a random graph
    initial_indices = rng.choice(len(all_edges), size=m, replace=False)
    edges = set(all_edges[i] for i in initial_indices)
    
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    books, C = compute_books(adj)
    current_max_book, current_sum_sq = get_stats(books)
    
    best_max_book = current_max_book
    best_sum_sq = current_sum_sq
    best_adj = adj.copy()
    best_edges = set(edges)
    
    no_improvement_count = 0
    stagnation_threshold = 50
    
    while time.time() < deadline:
        # Identify edges with high book sizes (hot edges)
        hot_edges = [e for e, b in books.items() if b >= current_max_book - 1]
        
        if not hot_edges:
            break
        
        # Identify non-edges with low potential book sizes (cold non-edges)
        # For a non-edge (u,v), the book size it would have is C[u,v]
        # We want to add edges with small C[u,v]
        
        non_edges = list(all_edges_set - edges)
        if not non_edges:
            break
        
        # Sort non-edges by their potential book size
        non_edges.sort(key=lambda e: C[e[0], e[1]])
        
        # Take the coldest non-edges
        K_cold = min(30, len(non_edges))
        cold_non_edges = non_edges[:K_cold]
        
        # Take the hottest edges
        K_hot = min(15, len(hot_edges))
        random.shuffle(hot_edges)
        hot_samples = hot_edges[:K_hot]
        
        best_swap = None
        best_new_max = current_max_book
        best_new_sum = current_sum_sq
        
        for e_rm in hot_samples:
            u, v = e_rm
            if e_rm not in edges:
                continue
            
            for e_add in cold_non_edges:
                a, b = e_add
                if e_add in edges:
                    continue
                
                # Quick check: if C[a,b] >= best_new_max, skip
                if C[a, b] >= best_new_max:
                    continue
                
                # Simulate swap
                adj[u, v] = 0
                adj[v, u] = 0
                adj[a, b] = 1
                adj[b, a] = 1
                
                C_new = adj @ adj
                
                # Compute new stats
                new_max = 0
                new_sum = 0
                for e in edges:
                    if e == e_rm:
                        continue
                    eu, ev = e
                    b_val = C_new[eu, ev]
                    if b_val > new_max:
                        new_max = b_val
                    new_sum += b_val * b_val
                
                # Add the new edge
                b_ab = C_new[a, b]
                if b_ab > new_max:
                    new_max = b_ab
                new_sum += b_ab * b_ab
                
                # Revert
                adj[u, v] = 1
                adj[v, u] = 1
                adj[a, b] = 0
                adj[b, a] = 0
                
                if new_max < best_new_max or (new_max == best_new_max and new_sum < best_new_sum):
                    best_new_max = new_max
                    best_new_sum = new_sum
                    best_swap = (e_rm, e_add)
        
        if best_swap:
            e_rm, e_add = best_swap
            u, v = e_rm
            a, b = e_add
            
            # Apply swap
            adj[u, v] = 0
            adj[v, u] = 0
            adj[a, b] = 1
            adj[b, a] = 1
            
            edges.remove(e_rm)
            edges.add(e_add)
            
            books, C = compute_books(adj)
            current_max_book, current_sum_sq = get_stats(books)
            
            if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
                best_max_book = current_max_book
                best_sum_sq = current_sum_sq
                best_adj = adj.copy()
                best_edges = set(edges)
            
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
            # If stuck, do a random perturbation
            if no_improvement_count > stagnation_threshold:
                if edges and non_edges:
                    e1 = random.choice(list(edges))
                    e2 = random.choice(non_edges)
                    
                    u1, v1 = e1
                    u2, v2 = e2
                    
                    adj[u1, v1] = 0
                    adj[v1, u1] = 0
                    adj[u2, v2] = 1
                    adj[v2, u2] = 1
                    
                    edges.remove(e1)
                    edges.add(e2)
                    
                    books, C = compute_books(adj)
                    current_max_book, current_sum_sq = get_stats(books)
                    
                    if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
                        best_max_book = current_max_book
                        best_sum_sq = current_sum_sq
                        best_adj = adj.copy()
                        best_edges = set(edges)
                    
                    no_improvement_count = 0
    
    # Return best found
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                result.append([i, j])
    
    # Sanity check
    if len(result) != 401:
        # Fallback: K20,20 + 1 edge
        result = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    return result
