"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph with 40 vertices and 401 edges minimizing the largest book
    (max over edges of number of triangles containing that edge).
    
    Approach: Spectral optimization via local search.
    1. Initialize with a quasi-random regular-like graph (balanced bipartite + one intra edge)
    2. Use local edge swaps to flatten the triangle distribution (A^2 degrees)
    3. Guide swaps using eigenvalue gap to maintain spectral properties
    """
    n = 40
    target_edges = 401
    
    # Step 1: Initialize with a balanced bipartite graph K20,20 + one intra-part edge
    # This gives 400 + 1 = 401 edges
    edges_set = set()
    for a in range(20):
        for b in range(20, 40):
            edges_set.add((a, b))
    # Add one edge within part A (vertices 0-19)
    edges_set.add((0, 1))
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.int32)
    for u, v in edges_set:
        adj[u, v] = 1
        adj[v, u] = 1
    
    # Helper: compute book size for each edge
    def compute_books(adj):
        """For each edge (u,v), book size = number of common neighbors = (A^2)[u][v]"""
        A2 = adj @ adj
        books = {}
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    books[(u, v)] = A2[u, v]
        return books
    
    # Helper: compute max book
    def max_book(adj):
        books = compute_books(adj)
        return max(books.values()) if books else 0
    
    # Initial max book
    current_max = max_book(adj)
    
    # Step 2: Local search - try edge swaps to reduce max book
    # An edge swap: remove edge (u,v) and add edge (w,x) where (w,x) is not an edge
    # We want swaps that reduce the maximum book size
    
    best_adj = adj.copy()
    best_max = current_max
    
    # Precompute neighbor sets for efficiency
    def get_neighbors(adj, u):
        return set(np.where(adj[u] == 1)[0])
    
    # Local search iterations
    import random
    random.seed(42)
    
    max_iterations = 5000
    for iteration in range(max_iterations):
        books = compute_books(adj)
        current_max = max(books.values()) if books else 0
        
        if current_max <= 6:
            break
        
        # Find edges with high book sizes
        high_book_edges = [(e, b) for e, b in books.items() if b >= current_max]
        
        if not high_book_edges:
            break
        
        # Try to reduce the book size of a high-book edge
        # Strategy: find an edge (u,v) with high book, try to remove it and add another edge
        # that doesn't create large books
        
        improved = False
        random.shuffle(high_book_edges)
        
        for (u, v), book_size in high_book_edges[:5]:  # Try top 5 high-book edges
            if book_size < current_max:
                continue
            
            # Try removing (u,v) and adding a different edge
            # The new edge should be between vertices that have few common neighbors
            # to keep book sizes low
            
            # Find candidates for new edge
            neighbors_u = get_neighbors(adj, u)
            neighbors_v = get_neighbors(adj, v)
            
            # Vertices not connected to u or v, or connected to only one
            # We want to add an edge (w,x) where w and x have few common neighbors
            # after removing (u,v)
            
            # Temporarily remove (u,v)
            adj[u, v] = 0
            adj[v, u] = 0
            
            A2_temp = adj @ adj
            
            # Find best candidate edge to add
            best_gain = 0
            best_new_edge = None
            best_new_max = current_max + 100
            
            # Sample candidate edges to keep computation tractable
            candidates = []
            for w in range(n):
                for x in range(w+1, n):
                    if w == x:
                        continue
                    if adj[w, x]:
                        continue
                    # Check if adding this edge would create a large book
                    # New book sizes: for edge (w,x), book = A2_temp[w][x]
                    # For edges incident to w or x, book sizes change
                    new_book_wx = A2_temp[w, x]
                    
                    # Estimate max book after adding (w,x)
                    # Edges incident to w: their book sizes increase by number of common neighbors with x
                    # This is complex, so use a heuristic
                    candidates.append((w, x, new_book_wx))
            
            # Sort by new book size (ascending) and try a few
            candidates.sort(key=lambda c: c[2])
            
            for w, x, new_book in candidates[:20]:
                # Add edge (w,x) temporarily
                adj[w, x] = 1
                adj[x, w] = 1
                
                A2_check = adj @ adj
                max_b = 0
                for a in range(n):
                    for b in range(a+1, n):
                        if adj[a, b]:
                            if A2_check[a, b] > max_b:
                                max_b = A2_check[a, b]
                
                # Remove (w,x)
                adj[w, x] = 0
                adj[x, w] = 0
                
                if max_b < best_new_max:
                    best_new_max = max_b
                    best_new_edge = (w, x)
                    best_gain = current_max - max_b
            
            # Restore (u,v)
            adj[u, v] = 1
            adj[v, u] = 1
            
            # Apply best swap if it improves
            if best_new_edge is not None and best_new_max < current_max:
                adj[u, v] = 0
                adj[v, u] = 0
                w, x = best_new_edge
                adj[w, x] = 1
                adj[x, w] = 1
                improved = True
                current_max = best_new_max
                break
        
        if not improved:
            # Try random perturbation to escape local minimum
            if iteration % 100 == 99:
                # Random edge swap
                edge_list = [(u, v) for u in range(n) for v in range(u+1, n) if adj[u, v]]
                if edge_list:
                    u, v = random.choice(edge_list)
                    adj[u, v] = 0
                    adj[v, u] = 0
                    
                    # Add random non-edge
                    non_edges = [(a, b) for a in range(n) for b in range(a+1, n) if not adj[a, b]]
                    if non_edges:
                        w, x = random.choice(non_edges)
                        adj[w, x] = 1
                        adj[x, w] = 1
            
            # Check if we've improved overall best
            cur_max = max_book(adj)
            if cur_max < best_max:
                best_max = cur_max
                best_adj = adj.copy()
    
    # Use the best found
    adj = best_adj
    
    # Extract edges from adjacency matrix
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                result.append([u, v])
    
    # Ensure exactly 401 edges
    if len(result) != target_edges:
        # Fallback to initial construction
        result = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    return result
# EVOLVE-BLOCK-END
