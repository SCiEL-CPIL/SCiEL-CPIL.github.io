"""Minimize the largest book in a 40-vertex, 401-edge graph.

Combines:
  - Fast book-size evaluation via matrix multiplication (A @ A.T)
  - Simulated Annealing with maintained edge/non-edge lists for efficient move generation
  - Multiple restarts to escape local minima
"""
import time
import math
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n = 40
    m = 401
    total_pairs = n * (n - 1) // 2
    num_non_edges = total_pairs - m

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def get_max_book(A):
        """Compute the maximum book size using matrix multiplication."""
        # A is n x n 0/1 matrix
        # A @ A.T gives number of length-2 paths between i and j
        # For an edge (i,j), the book size is (A @ A.T)[i,j]
        AA = A @ A.T
        # We only care about edges where A[i,j] == 1
        # We can mask non-edges
        # Since A is symmetric and 0/1, AA is symmetric.
        # max book = max(AA[i,j] for all edges (i,j))
        # To avoid Python loop, we can use numpy
        # Create a mask of edges
        # But A is already the mask.
        # We want max of AA where A is 1.
        # Set non-edges to -inf or 0?
        # If we set non-edges to 0, we might pick a non-edge with 0 triangles, which is fine as long as we only consider edges.
        # Actually, book size is defined for edges.
        # We can do:
        # AA_masked = np.where(A == 1, AA, -1)
        # return int(AA_masked.max())
        
        # Optimization: 
        # The diagonal of AA is the degree. We don't care about diagonal.
        # Let's just compute max over the upper triangle where A is 1.
        # For N=40, a simple loop over edges might be comparable to numpy overhead if we don't vectorize well.
        # But vectorizing is better.
        
        # Efficient vectorized max:
        # Flatten A and AA?
        # Let's try:
        # edges_mask = A > 0
        # vals = AA[edges_mask]
        # return int(vals.max()) if len(vals) > 0 else 0
        
        # This creates a boolean array of size 1600, and a data array.
        # This is very fast.
        
        mask = A > 0
        if not np.any(mask):
            return 0
        vals = AA[mask]
        return int(vals.max())

    def build_lists(A):
        """Return (edge_list, non_edge_list) of (i,j) with i<j."""
        edge_list, non_edge_list = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j]:
                    edge_list.append((i, j))
                else:
                    non_edge_list.append((i, j))
        return edge_list, non_edge_list

    def random_graph():
        """Generate a random graph with m edges."""
        A = np.zeros((n, n), dtype=np.int32)
        # Shuffle all edges and pick first m
        shuffled_edges = all_edges.copy()
        random.shuffle(shuffled_edges)
        for (i, j) in shuffled_edges[:m]:
            A[i, j] = 1
            A[j, i] = 1
        return A

    def sa_run(A, T0, T1, steps, time_limit):
        """Run Simulated Annealing."""
        edge_list, non_edge_list = build_lists(A)
        # Index maps for O(1) swap in lists
        edge_idx = {e: k for k, e in enumerate(edge_list)}
        non_edge_idx = {e: k for k, e in enumerate(non_edge_list)}
        
        current_max = get_max_book(A)
        best_A = A.copy()
        best_max = current_max
        
        T = T0
        # Cooling rate to reach T1 in 'steps'
        # T * (cooling_rate)^steps = T1
        # cooling_rate = (T1/T0)^(1/steps)
        if T1 > 0:
            cooling_rate = (T1 / T0) ** (1.0 / steps)
        else:
            cooling_rate = 0.9995 # Fallback
        
        for step in range(steps):
            if time.time() > time_limit:
                break
                
            # Pick random edge to remove
            k = random.randint(0, len(edge_list) - 1)
            u, v = edge_list[k]
            
            # Pick random non-edge to add
            l = random.randint(0, len(non_edge_list) - 1)
            a, b = non_edge_list[l]
            
            # Ensure we don't swap an edge with itself (impossible by lists)
            # But check if (u,v) is same as (a,b) ? No, one is edge, one is non-edge.
            
            # Perform swap
            A[u, v] = 0; A[v, u] = 0
            A[a, b] = 1; A[b, a] = 1
            
            new_max = get_max_book(A)
            delta = new_max - current_max
            
            # Metropolis criterion
            if delta <= 0 or (T > 1e-5 and random.random() < math.exp(-delta / T)):
                # Accept
                current_max = new_max
                
                # Update lists
                # Replace (u,v) in edge_list with (a,b)
                edge_list[k] = (a, b)
                edge_idx[(a, b)] = k
                del edge_idx[(u, v)]
                
                # Replace (a,b) in non_edge_list with (u,v)
                non_edge_list[l] = (u, v)
                non_edge_idx[(u, v)] = l
                del non_edge_idx[(a, b)]
                
                if current_max < best_max:
                    best_max = current_max
                    best_A = A.copy()
            else:
                # Reject
                A[u, v] = 1; A[v, u] = 1
                A[a, b] = 0; A[b, a] = 0
                
            T *= cooling_rate
            
        return best_A, best_max

    # Main search logic
    best_A_overall = None
    best_score_overall = float('inf')
    
    # Run multiple restarts
    # Budget time for each restart
    # Total time ~ 115s.
    # Let's do 5 restarts of 20s each? Or dynamic.
    
    # First, try a long SA run
    A_init = random_graph()
    A_best, score_best = sa_run(A_init, T0=2.0, T1=0.01, steps=50000, time_limit=start_time + 60)
    
    if score_best < best_score_overall:
        best_score_overall = score_best
        best_A_overall = A_best
    
    # Second restart with different parameters
    A_init2 = random_graph()
    A_best2, score_best2 = sa_run(A_init2, T0=1.0, T1=0.001, steps=50000, time_limit=start_time + 100)
    
    if score_best2 < best_score_overall:
        best_score_overall = score_best2
        best_A_overall = A_best2
        
    # Final local search on the best result to squeeze out more
    if best_A_overall is not None:
        A_ls = best_A_overall.copy()
        edge_list, non_edge_list = build_lists(A_ls)
        edge_idx = {e: k for k, e in enumerate(edge_list)}
        non_edge_idx = {e: k for k, e in enumerate(non_edge_list)}
        
        current_max = get_max_book(A_ls)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try to improve by swapping any edge with any non-edge
            # To be faster, only try swaps involving edges with high book sizes?
            # Or just random swaps until no improvement found in a batch.
            
            # Randomized local search
            for _ in range(1000):
                if time.time() > deadline - 2:
                    break
                    
                k = random.randint(0, len(edge_list) - 1)
                u, v = edge_list[k]
                l = random.randint(0, len(non_edge_list) - 1)
                a, b = non_edge_list[l]
                
                A_ls[u, v] = 0; A_ls[v, u] = 0
                A_ls[a, b] = 1; A_ls[b, a] = 1
                
                new_max = get_max_book(A_ls)
                
                if new_max < current_max:
                    current_max = new_max
                    # Update lists
                    edge_list[k] = (a, b)
                    edge_idx[(a, b)] = k
                    del edge_idx[(u, v)]
                    
                    non_edge_list[l] = (u, v)
                    non_edge_idx[(u, v)] = l
                    del non_edge_idx[(a, b)]
                    
                    if current_max < best_score_overall:
                        best_score_overall = current_max
                        best_A_overall = A_ls.copy()
                    improved = True
                else:
                    A_ls[u, v] = 1; A_ls[v, u] = 1
                    A_ls[a, b] = 0; A_ls[b, a] = 0

    # Convert best_A_overall to list of edges
    if best_A_overall is None:
        # Fallback: return random graph
        A_final = random_graph()
    else:
        A_final = best_A_overall
        
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if A_final[i, j]:
                result.append([i, j])
                
    # Verify count
    if len(result) != m:
        # Fallback: rebuild from scratch if something went wrong
        A_fallback = random_graph()
        result = []
        for i in range(n):
            for j in range(i + 1, n):
                if A_fallback[i, j]:
                    result.append([i, j])

    return result
