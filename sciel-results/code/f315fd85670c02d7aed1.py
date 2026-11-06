import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n = 40
    target_edges = 401

    # We represent the graph as an adjacency matrix for fast triangle counting.
    # A[u, v] = 1 if edge exists, 0 otherwise.
    # Book size of edge (u, v) is (A @ A)[u, v].

    def get_books(A):
        # Returns the matrix of book sizes for all pairs.
        # We only care about edges, but computing full matrix is fast (40x40).
        return A @ A

    def max_book(A):
        books = get_books(A)
        # Mask non-edges to 0
        masked = np.where(A > 0, books, 0)
        return int(masked.max())

    # --- Phase 1: Fast Random Initialization ---
    # Start with a balanced random graph near the threshold
    # Use a simple random graph with edge probability tuned for ~401 edges
    
    p = target_edges / (n * (n-1) / 2)
    
    A = np.zeros((n, n), dtype=np.int8)
    added_edges = []
    
    # Generate random edges until we have enough
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges[:target_edges]:
        A[u, v] = 1
        A[v, u] = 1
        added_edges.append((u, v))
    
    current_max = max_book(A)
    best_max = current_max
    best_A = A.copy()
    
    # Precompute degree and common neighbors for incremental updates
    # B[u,v] = number of common neighbors of u and v (for u != v)
    # When we add edge (x,y):
    #   For any pair (u,v): B[u,v] increases by 1 if (u==x and v==y) or (u==y and v==x)
    #   More precisely, adding edge (x,y) affects B[u,v] for all (u,v) where 
    #   the new edge creates a new path u-x-y-v or u-y-x-v
    # This is complex. Let's just use full recomputation but optimize the search.
    
    # --- Phase 2: Efficient Local Search ---
    # Key insight: Instead of trying 20 random swaps per iteration,
    # we can be smarter about which edges to swap.
    
    # Strategy:
    # 1. Find the max-book edge (u,v)
    # 2. To reduce its book, we need to remove one of its common neighbors' connections
    #    or add an edge that doesn't create new triangles through (u,v)
    # 3. Better: Try swapping (u,v) with a non-edge (x,y) where x,y have few common neighbors
    #    and also x,y don't share many neighbors with u or v
    
    # Precompute list of all edges and non-edges for faster access
    edge_set = set(added_edges)
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edge_set]
    
    # Main search loop
    while time.time() < deadline:
        # Recompute books
        books = A @ A
        masked_books = np.where(A > 0, books, 0)
        max_val = int(masked_books.max())
        
        if max_val < best_max:
            best_max = max_val
            best_A = A.copy()
            
        # The theoretical minimum is ceil(40/6) = 7
        if best_max <= 7:
            break

        # Find all edges with book size == max_val
        idxs = np.argwhere(masked_books == max_val)
        if len(idxs) == 0:
            break
            
        # Pick a random max-book edge to remove
        u, v = idxs[random.randint(0, len(idxs)-1)]
        
        # Try to find a good non-edge to add
        # Heuristic: prefer non-edges (x,y) where:
        # 1. (x,y) has few common neighbors (low book if added)
        # 2. x and y don't share many neighbors with u or v (to avoid increasing other books)
        
        # Sample candidate non-edges
        improved = False
        # Try up to 50 random non-edges, but prioritize ones with low common neighbors
        for _ in range(50):
            if not non_edges:
                break
            idx = random.randint(0, len(non_edges)-1)
            x, y = non_edges[idx]
            
            # Quick check: how many common neighbors would (x,y) have?
            common_xy = int(np.sum(A[x] & A[y]))
            
            # Check if adding (x,y) would create triangles through u or v
            # Triangle through u: need x-u and y-u edges, or x-u and u-y edges
            # Actually, triangle (x,y,u) exists if A[x,u] and A[y,u]
            # Triangle (x,y,v) exists if A[x,v] and A[y,v]
            
            tri_through_u = int(A[x, u]) + int(A[y, u])  # 0, 1, or 2
            tri_through_v = int(A[x, v]) + int(A[y, v])
            
            # If tri_through_u >= 2, then (x,y,u) is a triangle, increasing book of edges (x,u) and (y,u)
            # We want to minimize the impact
            
            # Tentative move
            A[u, v] = 0
            A[v, u] = 0
            A[x, y] = 1
            A[y, x] = 1
            
            new_books = A @ A
            new_masked = np.where(A > 0, new_books, 0)
            new_max = int(new_masked.max())
            
            if new_max < max_val:
                # Accept! Update non_edges
                non_edges.remove((x, y))
                non_edges.append((min(u,v), max(u,v)))
                edge_set.add((x, y))
                edge_set.remove((u, v))
                improved = True
                break
            else:
                # Revert
                A[u, v] = 1
                A[v, u] = 1
                A[x, y] = 0
                A[y, x] = 0
        
        if not improved:
            # If we couldn't improve by removing a max-edge,
            # try a random swap that doesn't increase max
            found_move = False
            for _ in range(20):
                # Pick random edge to remove
                ru, rv = random.choice(list(edge_set))
                # Pick random non-edge to add
                if not non_edges:
                    break
                rx, ry = non_edges[random.randint(0, len(non_edges)-1)]
                
                A[ru, rv] = 0
                A[rv, ru] = 0
                A[rx, ry] = 1
                A[ry, rx] = 1
                
                nb = A @ A
                nm = np.where(A > 0, nb, 0)
                nmax = int(nm.max())
                
                if nmax <= max_val:
                    # Accept
                    edge_set.add((rx, ry))
                    edge_set.remove((ru, rv))
                    non_edges.remove((rx, ry))
                    non_edges.append((min(ru,rv), max(ru,rv)))
                    found_move = True
                    break
                else:
                    # Revert
                    A[ru, rv] = 1
                    A[rv, ru] = 1
                    A[rx, ry] = 0
                    A[ry, rx] = 0
            
            if not found_move:
                # Stuck for this iteration, continue to next
                pass

    # Final check and extraction
    final_A = best_A
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if final_A[i, j] == 1:
                edges.append([i, j])
                
    # Ensure exactly 401 edges
    if len(edges) != target_edges:
        # Fallback: regenerate if something went wrong
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        edges = all_edges[:target_edges]
        
    return edges

