import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 40
    m = 401
    
    # 1. DETERMINISTIC SPECTRAL INITIALIZATION
    # We want to start with a graph that is "as bipartite as possible" to minimize triangles.
    # A good proxy is to use the Fiedler vector (2nd eigenvector of Laplacian) of a random 
    # or structured graph to define a partition, then build edges based on that partition 
    # with some noise.
    # However, a simpler and very effective deterministic start for "min max book" is 
    # to take a balanced bipartite graph (K20,20) and add edges within parts in a 
    # "spread out" manner to avoid clustering.
    
    # Let's use a slightly more sophisticated init:
    # Start with K20,20 (400 edges). We need 1 more edge.
    # If we just add (0,1), the book size of (0,1) is 20 (connected to all 20 in B).
    # This is bad (max book 20).
    
    # Better Init:
    # Random graph G(40, 401) usually has max book ~ 15-20.
    # We want to start lower.
    # Let's try a "quasi-random" construction using a modular grid or similar.
    # Or, use the Fiedler vector of a complete graph? No, that's trivial.
    
    # Let's stick to a robust random start but with a specific seed, 
    # then rely on the vectorized search to do the heavy lifting.
    # The key improvement is in the SEARCH, not just the init.
    
    rng = np.random.default_rng(42)
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Initial random selection
    initial_indices = rng.choice(len(all_edges), size=m, replace=False)
    edges = set(all_edges[i] for i in initial_indices)
    
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
        
    # Helper to compute books and stats
    def compute_stats(adj_mat):
        # Book size of edge (u,v) is number of common neighbors = (A^2)[u,v]
        C = adj_mat @ adj_mat
        # We only care about edges that exist
        # Create a mask of edges
        # Efficient way: extract values where adj is 1
        # But we need to map back to edge indices for swapping.
        
        # Let's maintain a list of edge indices into a flattened upper triangle
        # or just a list of (u,v) pairs.
        pass

    # Data Structures for Vectorized Search
    # We will maintain:
    # 1. adj: 40x40 int8 matrix
    # 2. edge_list: list of (u,v) tuples currently in graph
    # 3. non_edge_list: list of (u,v) tuples NOT in graph
    
    edge_list = list(edges)
    all_possible_set = set(all_edges)
    non_edge_list = list(all_possible_set - edges)
    
    # Precompute C = adj @ adj
    C = adj @ adj
    
    # Current stats
    def get_current_stats():
        max_book = 0
        sum_sq = 0
        for u, v in edge_list:
            b = C[u, v]
            if b > max_book:
                max_book = b
            sum_sq += b*b
        return max_book, sum_sq

    current_max_book, current_sum_sq = get_current_stats()
    
    best_max_book = current_max_book
    best_sum_sq = current_sum_sq
    best_adj = adj.copy()
    best_edges = set(edge_list)
    
    no_improvement_count = 0
    
    while time.time() < deadline:
        # 1. Identify Hot Edges (max book) and Cold Non-Edges (low potential book)
        
        # To make this vectorized, we work with arrays.
        # Convert edge_list to array of indices? 
        # Let's stick to lists for simplicity but batch the evaluation.
        
        # Identify hot edges: those with book == current_max_book
        hot_edges = [e for e in edge_list if C[e[0], e[1]] == current_max_book]
        
        if not hot_edges:
            break
            
        # Identify cold non-edges: those with C[u,v] < current_max_book
        # Ideally we want the lowest C[u,v]
        # We can sort non_edge_list by C[u,v]
        # But C changes after every swap.
        
        # Optimization: 
        # Instead of trying all (hot, cold) pairs, we sample.
        # But we can vectorize the evaluation of a batch of swaps.
        
        # Let's pick a batch of candidate swaps.
        # Sample K_hot hot edges and K_cold cold non-edges.
        
        K_hot = min(20, len(hot_edges))
        K_cold = min(50, len(non_edge_list))
        
        # Shuffle to avoid bias
        random.shuffle(hot_edges)
        hot_samples = hot_edges[:K_hot]
        
        random.shuffle(non_edge_list)
        # Filter for "cold" ones if possible, otherwise just take random
        # C[u,v] for non-edges is the number of triangles it would create.
        # We want these to be small.
        
        cold_samples = []
        for e in non_edge_list:
            if C[e[0], e[1]] < current_max_book:
                cold_samples.append(e)
            if len(cold_samples) >= K_cold:
                break
        
        if len(cold_samples) < K_cold:
            # Fill with random non-edges
            for e in non_edge_list:
                if e not in cold_samples:
                    cold_samples.append(e)
                if len(cold_samples) >= K_cold:
                    break
                    
        cold_samples = cold_samples[:K_cold]
        
        # Now we have K_hot * K_cold candidate swaps.
        # We want to find the one that minimizes (new_max, new_sum_sq).
        
        # Vectorized Evaluation Strategy:
        # For a swap (remove (u,v), add (a,b)):
        # 1. Remove (u,v) from graph.
        # 2. Add (a,b) to graph.
        # 3. Update C matrix? 
        #    C_new = C + delta
        #    Delta from removing (u,v):
        #    - C[u,v] decreases by deg(u)+deg(v)-2 (no, C[u,v] is common neighbors).
        #    - Actually, C is A^2. 
        #    - Change in A: A[u,v]=0, A[v,u]=0.
        #    - Change in A^2: 
        #      (A^2)[i,j] = sum_k A[i,k]A[k,j]
        #      Removing A[u,v] affects (A^2)[u,*] and (A^2)[*,v] and (A^2)[v,*] and (A^2)[*,u].
        #      Specifically, for any i,j:
        #      (A^2_new)[i,j] = (A^2)[i,j] - A[i,u]A[u,j] - A[i,v]A[v,j] (approx, need to be careful with u/v terms)
        
        # This is getting complex to vectorize perfectly for arbitrary swaps without rebuilding.
        # Given n=40, rebuilding C (40x40 matmul) is very fast (~microseconds).
        # The bottleneck in the previous code was the Python loop over candidates.
        
        # New Strategy:
        # 1. Generate all K_hot * K_cold candidate swaps.
        # 2. For each candidate, we need to evaluate the new max book.
        # 3. We can do this in a loop, but we optimize the inner check.
        
        best_swap = None
        best_new_max = current_max_book
        best_new_sum = current_sum_sq
        
        # To speed up, we can pre-filter candidates that clearly won't work.
        # If we add (a,b), the new edge has book size C[a,b].
        # If C[a,b] >= current_max_book, it might not reduce the max (unless we remove the unique max edge and C[a,b] < current_max_book).
        
        for e_rm in hot_samples:
            u, v = e_rm
            # Remove (u,v)
            # We need to know the new max book.
            # The max book can only decrease if we remove an edge that was part of the max.
            # It can increase if we add an edge with high book.
            
            for e_add in cold_samples:
                a, b = e_add
                
                # Quick check: 
                # New edge (a,b) has book size C[a,b].
                # If C[a,b] > best_new_max, this swap is bad (unless best_new_max is already very high and we are desperate, but we want to minimize).
                # Actually, if C[a,b] > current_max_book, it's definitely bad.
                # If C[a,b] == current_max_book, it might be neutral or bad.
                # If C[a,b] < current_max_book, it's potentially good.
                
                if C[a, b] >= best_new_max:
                    continue
                
                # Simulate swap
                # We modify adj temporarily
                adj[u, v] = 0
                adj[v, u] = 0
                adj[a, b] = 1
                adj[b, a] = 1
                
                # Recalculate C
                C_new = adj @ adj
                
                # Calculate new max and sum_sq
                # We only need to check edges that were affected?
                # No, max could be elsewhere.
                # But n=40 is small.
                new_max = 0
                new_sum = 0
                for e in edge_list:
                    # Note: edge_list still contains e_rm and does not contain e_add.
                    # We need to handle this.
                    pass
                
                # This inner loop is O(E) ~ 400.
                # 20 * 50 = 1000 swaps.
                # 1000 * 400 = 400,000 ops per iteration.
                # Plus matmul 40x40.
                # This should be fast enough in Python if we optimize the loop.
                
                # Let's optimize the stats calculation.
                # We can maintain a sorted list of books? No, too complex.
                # Just iterate.
                
                # To make this faster, we can pre-compute the list of edges that are NOT (u,v) and NOT (a,b).
                # Actually, let's just iterate over all edges in edge_list, skipping e_rm, and adding e_add.
                
                # We need to be careful: edge_list is the CURRENT list.
                # e_rm is in edge_list. e_add is NOT in edge_list.
                
                # Let's just do a full scan.
                for e in edge_list:
                    if e == e_rm:
                        continue
                    bu, bv = e
                    b_val = C_new[bu, bv]
                    if b_val > new_max:
                        new_max = b_val
                    new_sum += b_val * b_val
                
                # Add the new edge (a,b)
                b_ab = C_new[a, b]
                if b_ab > new_max:
                    new_max = b_ab
                new_sum += b_ab * b_ab
                
                # Revert adj
                adj[u, v] = 1
                adj[v, u] = 1
                adj[a, b] = 0
                adj[b, a] = 0
                
                # Compare
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
            
            edge_list.remove(e_rm)
            edge_list.append(e_add)
            
            non_edge_list.append(e_rm)
            non_edge_list.remove(e_add)
            
            # Update C
            C = adj @ adj
            
            current_max_book = best_new_max
            current_sum_sq = best_new_sum
            
            # Update best
            if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
                best_max_book = current_max_book
                best_sum_sq = current_sum_sq
                best_adj = adj.copy()
                best_edges = set(edge_list)
                
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            # If stuck, do a random perturbation
            if no_improvement_count > 10:
                # Random swap
                if edge_list and non_edge_list:
                    e1 = random.choice(edge_list)
                    e2 = random.choice(non_edge_list)
                    
                    u1, v1 = e1
                    u2, v2 = e2
                    
                    adj[u1, v1] = 0
                    adj[v1, u1] = 0
                    adj[u2, v2] = 1
                    adj[v2, u2] = 1
                    
                    edge_list.remove(e1)
                    edge_list.append(e2)
                    
                    non_edge_list.append(e1)
                    non_edge_list.remove(e2)
                    
                    C = adj @ adj
                    current_max_book, current_sum_sq = get_current_stats()
                    
                    if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
                        best_max_book = current_max_book
                        best_sum_sq = current_sum_sq
                        best_adj = adj.copy()
                        best_edges = set(edge_list)
                        
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
