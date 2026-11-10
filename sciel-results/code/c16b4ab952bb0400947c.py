import numpy as np
import random
import time

def construct():
    n = 31
    m = 91
    
    # Deadline guard
    start_time = time.time()
    time_limit = 110  # 10 seconds margin from 120s limit? 
    # The prompt says 120s limit. 150s kill. 
    # Let's use 100s to be safe.
    
    def count_c4(A):
        # A is 0-1 matrix
        # A^2[i,j] is number of common neighbors
        # C4 count = 0.5 * sum_{i<j} C(A^2[i,j], 2)
        A2 = np.dot(A, A)
        # C(k, 2) = k(k-1)/2
        # We need sum over i < j
        # A2 is symmetric.
        # Triangular sum
        idx = np.triu_indices(n, 1)
        vals = A2[idx]
        # vals are integers
        c4s = np.sum(vals * (vals - 1) / 2)
        return c4s / 2.0
        
    # Initialize random graph
    # We need exactly m edges
    # Use a random permutation of all possible edges and take first m
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    edges = all_edges[:m]
    
    A = np.zeros((n, n), dtype=int)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
        
    current_c4 = count_c4(A)
    
    # Local search
    # We will try to improve by swapping edges
    # To speed up, we can precompute something?
    # For now, just iterate.
    
    # List of current edges
    edge_set = set(edges)
    
    # List of non-edges
    non_edges = [e for e in all_edges if e not in edge_set]
    
    improved = True
    while improved and time.time() - start_time < 100:
        improved = False
        # Try all swaps? That's 91 * 479 ~ 43k swaps.
        # Each swap: update A, count C4, revert A.
        # Counting C4 is ~ 31x31 matmul + sum. 
        # 43k * (matmul + sum) might be slow in pure Python loop if not vectorized?
        # But we can just do it. 43k iterations is small.
        
        best_swap = None
        best_c4 = current_c4
        
        # Optimization: Only try swaps that might reduce C4.
        # Removing an edge (u,v) reduces C4s containing (u,v).
        # Adding an edge (x,y) adds C4s containing (x,y).
        
        # We can approximate the change.
        # But exact calculation is safer.
        
        # To make it faster, we can sample swaps or use a heuristic.
        # Let's try a greedy approach: 
        # For each edge e, find the best non-edge f to swap with.
        
        for u, v in edges:
            # Remove u-v
            A[u, v] = 0
            A[v, u] = 0
            base_c4 = count_c4(A)
            
            # Try adding each non-edge
            # We can filter non-edges that are "far" from u,v?
            # Just iterate.
            for x, y in non_edges:
                if (x, y) == (u, v) or (x, y) == (v, u):
                    continue
                # Add x-y
                A[x, y] = 1
                A[y, x] = 1
                
                new_c4 = count_c4(A)
                
                # Revert x-y
                A[x, y] = 0
                A[y, x] = 0
                
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_swap = ((u, v), (x, y))
            
            # Revert u-v
            A[u, v] = 1
            A[v, u] = 1
            
            if best_swap and best_swap[0] == (u, v):
                # Apply the best swap found for this edge? 
                # No, we should find the global best first, then apply.
                # Or apply immediately if it's an improvement?
                # Hill climbing: apply first improvement or best improvement?
                # Best improvement is slower but better.
                pass

        # If we found a best swap globally?
        # The loop above checks all edges and all non-edges for each edge.
        # It finds the best swap for each edge, but we need the global best.
        # Let's restructure:
        # Find global best swap.
        
        # This nested loop is 91 * 479 = 43k iterations.
        # Each iteration does 2 count_c4 calls? 
        # No, I removed u-v, then added x-y, counted, removed x-y.
        # So for each (u,v) and (x,y), we call count_c4 once.
        # Total 43k calls to count_c4.
        # count_c4 does 31x31 matmul.
        # 43k * (31^3 * 2) ops ~ 43k * 60k ops ~ 2.5 billion ops.
        # This might be too slow in Python/NumPy if not optimized.
        # NumPy matmul is C-optimized. 31x31 matmul is very fast.
        # Let's estimate. 31x31 matmul takes ~ 1-2 microseconds?
        # 43k * 2us = 86ms.
        # Plus Python overhead. Maybe 1-2 seconds per iteration.
        # So we can do many iterations.
        
        # Let's refine the loop to find the single best swap in one pass.
        
        best_swap = None
        best_c4 = current_c4
        
        # We can parallelize? No, standard library.
        
        # Let's just do it.
        
        for u, v in edges:
            A[u, v] = 0
            A[v, u] = 0
            
            for x, y in non_edges:
                if (x, y) == (u, v) or (x, y) == (v, u):
                    continue
                
                A[x, y] = 1
                A[y, x] = 1
                
                new_c4 = count_c4(A)
                
                A[x, y] = 0
                A[y, x] = 0
                
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_swap = ((u, v), (x, y))
            
            A[u, v] = 1
            A[v, u] = 1
            
            # Early exit if we found a swap that reduces C4 to 0? Unlikely.
            # If we found a significant improvement, we could break?
            # No, we want the best.
            
        if best_swap:
            e_out, e_in = best_swap
            # Apply swap
            A[e_out[0], e_out[1]] = 0
            A[e_out[1], e_out[0]] = 0
            A[e_in[0], e_in[1]] = 1
            A[e_in[1], e_in[0]] = 1
            
            # Update edge lists
            edge_set.remove(e_out)
            edge_set.add(e_in)
            edges.remove(e_out)
            edges.append(e_in)
            non_edges.remove(e_in)
            non_edges.append(e_out)
            
            current_c4 = best_c4
            improved = True
        else:
            improved = False
            
    # Return the edges
    return [[u, v] for u, v in edges]
