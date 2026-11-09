"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n = 40
    m = 401
    vertices = list(range(n))
    
    # Helper to compute max book size for a given edge list
    # A book on edge (u,v) is the number of w such that (u,w) and (v,w) are edges.
    # This is equivalent to the number of triangles containing (u,v).
    
    def get_max_book(edges):
        # Build adjacency matrix for fast lookup
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        
        max_book = 0
        # For each edge, count common neighbors
        # We can vectorize this: for each edge (u,v), common neighbors = sum(adj[u] & adj[v])
        # But iterating 401 edges and doing a 40-element dot product is fast enough.
        for u, v in edges:
            # Common neighbors of u and v
            # adj[u] is a row vector of 0s and 1s.
            # We need to count how many w != u, v have adj[u][w]=1 and adj[v][w]=1.
            # Since adj[u][u]=0 and adj[v][v]=0, and adj[u][v]=1, adj[v][u]=1,
            # we can just sum adj[u] * adj[v].
            # Wait, adj[u][v] is 1, so the product at index v is 1*1=1? No, adj[u][v] is 1, adj[v][v] is 0.
            # adj[u] * adj[v] element-wise.
            # At index w: adj[u][w] * adj[v][w].
            # Sum over all w.
            # This counts triangles containing edge (u,v).
            common = np.dot(adj[u], adj[v])
            if common > max_book:
                max_book = common
        return max_book

    # Initial random graph with 401 edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    edges = all_pairs[:m]
    
    current_max_book = get_max_book(edges)
    best_edges = edges[:]
    best_max_book = current_max_book
    
    # Local Search: Simulated Annealing or Hill Climbing
    # Move: Swap an edge (u,v) with a non-edge (x,y)
    # Or: Remove an edge and add another (keep m constant)
    
    # To make search efficient, we can focus on edges that are part of the max book.
    # However, a generic random swap is often effective if we have enough time.
    
    # Let's try a simple hill-climbing with random restarts if stuck.
    # Given 140s, we can do many steps.
    
    # Precompute edge set for O(1) lookup
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    def edge_key(u, v):
        return (min(u, v), max(u, v))
    
    # Optimization: Instead of recalculating max book from scratch (O(m*n)),
    # we can update it incrementally or use a faster approximation.
    # But for n=40, m=401, O(m*n) = 16040 ops per evaluation.
    # 10,000 evaluations = 160M ops. NumPy dot products are fast, but Python loop over edges is slow.
    
    # Let's vectorize the max book calculation.
    def get_max_book_fast(edges_list):
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges_list:
            adj[u, v] = 1
            adj[v, u] = 1
        
        # Compute triangle count for each edge
        # Matrix M where M[u,v] = number of triangles containing (u,v)
        # M = (A^3)_{uv} / 2? No.
        # Number of triangles containing (u,v) is (A * A * A)_{uv} / 2 ?
        # Let A be adjacency matrix.
        # (A^3)_{uv} = sum_k sum_l A_{uk} A_{kl} A_{lv}.
        # This counts walks of length 3 from u to v.
        # A walk u-w-v-w-u? No.
        # Triangles containing (u,v) correspond to w such that u-w, w-v, v-u are edges.
        # This is exactly the number of common neighbors of u and v.
        # Common neighbors = sum_w A_{uw} A_{vw}.
        # This is (A^2)_{uv}.
        # So the book size for edge (u,v) is (A^2)_{uv}.
        
        A2 = np.dot(adj, adj)
        
        # We only care about entries where adj[u,v] == 1.
        # Create a mask of edges
        mask = (adj == 1)
        # Set non-edges to -inf so they don't get picked
        scores = np.where(mask, A2, -1)
        
        return int(np.max(scores))

    # Update best_edges with fast calculation
    current_max_book = get_max_book_fast(edges)
    best_edges = edges[:]
    best_max_book = current_max_book
    
    # Local Search Loop
    # We will perform random edge swaps.
    # Pick a random edge (u,v) to remove.
    # Pick a random non-edge (x,y) to add.
    # Evaluate new max book.
    
    # To speed up, we can precompute the list of non-edges.
    # But the set of non-edges changes.
    
    # Let's just do random swaps.
    steps = 0
    temp = 1.0  # Not really SA, just hill climbing with occasional random moves
    
    while time.time() < deadline:
        steps += 1
        # Random swap
        # Pick random edge index
        idx = random.randint(0, m - 1)
        u, v = edges[idx]
        
        # Pick random non-edge
        # Generate random pair
        while True:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x == y:
                continue
            key = edge_key(x, y)
            if key not in edge_set:
                break
        
        # Perform swap
        # Remove (u,v), add (x,y)
        edge_set.remove(edge_key(u, v))
        edge_set.add(key)
        
        # Update edges list
        edges[idx] = [x, y]
        
        # Evaluate
        new_max_book = get_max_book_fast(edges)
        
        if new_max_book < current_max_book:
            current_max_book = new_max_book
            if current_max_book < best_max_book:
                best_max_book = current_max_book
                best_edges = edges[:]
                # If we hit 6, we can probably stop or keep trying for 5?
                # Theorem says >= 40/6 = 6.66, so 7 is lower bound?
                # Wait, 40/6 = 6.66. So max book must be at least 7?
                # "score >= 40/6". Since score is integer, score >= 7.
                # So 7 is the theoretical minimum.
                if best_max_book == 7:
                    # Can we do better? No, theorem says >= 7.
                    # So 7 is optimal.
                    break
        else:
            # Revert swap
            edge_set.remove(key)
            edge_set.add(edge_key(u, v))
            edges[idx] = [u, v]
            
            # Optional: Random walk to escape local minima
            # If stuck for a while, do a random move regardless
            # For now, simple hill climbing.
            
    # Return best_edges found
    # Ensure exactly 401 distinct edges
    # The swap logic maintains this.
    return best_edges
# EVOLVE-BLOCK-END
