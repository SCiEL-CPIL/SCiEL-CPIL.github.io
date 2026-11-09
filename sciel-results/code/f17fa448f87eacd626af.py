import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 40
    target_edges = 401
    
    # 1. Construct initial graph: Blow-up of C5
    # Partition 40 vertices into 5 sets of 8: V0..V4
    # Connect Vi to V(i+1)%5 and V(i-1)%5 completely.
    # This gives 5 * 8 * 8 = 320 edges.
    # Degrees are 16.
    # No triangles in C5, so initial max book is 0.
    
    adj = np.zeros((n, n), dtype=bool)
    partition_size = 8
    
    for i in range(5):
        start_i = i * partition_size
        end_i = start_i + partition_size
        start_j1 = ((i + 1) % 5) * partition_size
        end_j1 = start_j1 + partition_size
        start_j2 = ((i - 1) % 5) * partition_size
        end_j2 = start_j2 + partition_size
        
        # Connect Vi to V(i+1)
        adj[start_i:end_i, start_j1:end_j1] = True
        adj[start_j1:end_j1, start_i:end_i] = True
        
        # Connect Vi to V(i-1)
        adj[start_i:end_i, start_j2:end_j2] = True
        adj[start_j2:end_j2, start_i:end_i] = True
        
    # Current edges: 320. Target: 401. Need 81 more.
    # We will add edges within partitions to increase density while keeping book size low.
    # Adding an edge (u,v) within Vi creates triangles with any common neighbor in V(i+1) or V(i-1).
    # Since u and v are in Vi, their common neighbors are all of V(i+1) and V(i-1) (size 16).
    # So any intra-partition edge has book size 16.
    # This is too high. We want max book <= 7.
    
    # Alternative Strategy: Start with a random graph with 401 edges and optimize.
    # The C5 blow-up has too much structure that forces high book sizes if we just add edges.
    # Let's use a randomized greedy construction or start from a random graph and anneal, 
    # but with a faster evaluation and stricter acceptance criteria.
    
    # Actually, the prompt suggests a "deterministic spectral heuristic" or "different design".
    # Let's try a different deterministic construction: 
    # Start with a random graph with 401 edges.
    # Then perform a local search: 
    # 1. Find edge with max book.
    # 2. Try to swap it with a non-edge that reduces the max book or keeps it same while reducing sum of squares of book sizes.
    
    # Let's implement a fast local search starting from a random graph.
    # To ensure speed, we will use numpy for book size calculation.
    
    # Generate a random graph with 401 edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    np.random.seed(42)
    selected_indices = np.random.choice(len(all_edges), target_edges, replace=False)
    for idx in selected_indices:
        u, v = all_edges[idx]
        adj[u, v] = True
        adj[v, u] = True
    
    # Helper to compute book sizes for all edges
    def compute_book_sizes(adj_mat):
        # Book size of edge (u,v) is number of common neighbors.
        # We can compute the triangle count matrix T = (A^2 * A) / 2 ? No.
        # Number of triangles containing edge (u,v) is (A[u] & A[v]).sum()
        # Let's compute a matrix B where B[u,v] is book size of (u,v).
        # B = (A @ A) - A? No.
        # (A @ A)[u,v] is number of common neighbors.
        # So B = A @ A.
        # For non-edges, B[u,v] is still number of common neighbors (length 2 path).
        # We only care about B[u,v] for edges (u,v).
        
        A = adj_mat.astype(np.int8)
        common_neighbors = A @ A
        # common_neighbors[u,v] is the book size of edge (u,v) if it exists.
        return common_neighbors
    
    # Initial score
    common_neighbors = compute_book_sizes(adj)
    current_max_book = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                if common_neighbors[u, v] > current_max_book:
                    current_max_book = int(common_neighbors[u, v])
    
    best_adj = adj.copy()
    best_score = current_max_book
    
    # Local Search
    # We will iterate through edges with high book size and try to swap them.
    # To be deterministic and fast, we will do multiple passes.
    
    # Precompute list of edges
    edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
    non_edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
    
    # For efficiency, we don't want to recompute full matrix every time if possible,
    # but with n=40, A@A is very fast (40x40 matmul).
    # Let's do a simple hill climbing.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Recompute book sizes to be safe
        common_neighbors = compute_book_sizes(adj)
        
        # Find edges with book size > best_score - 1? 
        # Actually, we want to minimize the MAX.
        # So we look for the edge with the current maximum book size.
        max_b = 0
        max_edge = None
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    b = int(common_neighbors[u, v])
                    if b > max_b:
                        max_b = b
                        max_edge = (u, v)
        
        if max_b <= 6: # Theoretical minimum is likely 7, but let's aim for low
            break
            
        # Try to swap max_edge with a non-edge
        u, v = max_edge
        
        # We want to find a non-edge (x,y) such that adding it and removing (u,v)
        # results in a lower max book size.
        
        # Heuristic: Remove (u,v). This reduces book sizes of edges in triangles containing (u,v).
        # Add (x,y). This increases book sizes of edges in triangles containing (x,y).
        
        # To find a good swap, we can check a subset of non-edges.
        # Let's pick non-edges that are "far" from u and v in some sense, or just random.
        # Since we want deterministic, let's iterate through non-edges in a fixed order.
        
        # To speed up, we can pre-filter non-edges.
        # A good swap should not create new high-book edges.
        
        # Let's just try the first 100 non-edges in a shuffled deterministic order?
        # No, let's try to find a swap that reduces the max book size of the removed edge's neighborhood.
        
        # Simple approach: Try all non-edges? 780 - 401 = 379 non-edges.
        # For each, we simulate the swap.
        # Simulating swap requires updating book sizes for affected edges.
        # Affected edges are those sharing a vertex with (u,v) or (x,y).
        # That's at most 2*(n-1) + 2*(n-1) ~ 160 edges.
        # Checking max book over all 401 edges is O(E) = 401.
        # Total per swap: ~ 500 ops.
        # 379 swaps * 500 ops = ~190k ops per iteration.
        # This is very fast.
        
        found_swap = False
        # Sort non-edges by some heuristic? 
        # Let's just iterate.
        for x, y in non_edge_list:
            # Skip if (x,y) is (u,v)
            if (x == u and y == v) or (x == v and y == u):
                continue
                
            # Simulate: Remove (u,v), Add (x,y)
            # We need to check if the new max book size is < current max_b
            
            # To do this efficiently without full recompute:
            # 1. Identify edges whose book size changes.
            #    - Edges (u,w) where w is neighbor of v: lose 1 if (u,v) removed? No.
            #    Book size of (u,w) is count of common neighbors.
            #    If we remove (u,v), (u,v) is no longer an edge, but does it affect common neighbors of (u,w)?
            #    Common neighbors of (u,w) are vertices z such that (u,z) and (w,z) exist.
            #    Removing (u,v) only affects common neighbors if v is a common neighbor of (u,w).
            #    i.e., if (u,v) and (w,v) exist.
            
            # This incremental update is complex. Given n=40, let's just do the full recompute for the candidate swap.
            # It's O(n^3) = 64000 ops per candidate? No, A@A is O(n^3) but with small n it's fast.
            # Actually, Python loop for max book is O(E) = 401.
            # A@A is done by numpy, very fast.
            
            # Apply
            adj[u, v] = False
            adj[v, u] = False
            adj[x, y] = True
            adj[y, x] = True
            
            # Recompute
            common_neighbors_new = compute_book_sizes(adj)
            
            # Find new max
            new_max_b = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j]:
                        b = int(common_neighbors_new[i, j])
                        if b > new_max_b:
                            new_max_b = b
                            if new_max_b >= max_b: # Early exit if not better
                                break
                if new_max_b >= max_b:
                    break
            
            # Revert
            adj[u, v] = True
            adj[v, u] = True
            adj[x, y] = False
            adj[y, x] = False
            
            if new_max_b < max_b:
                # Accept swap
                adj[u, v] = False
                adj[v, u] = False
                adj[x, y] = True
                adj[y, x] = True
                
                # Update lists
                edge_list.remove((u, v))
                non_edge_list.remove((x, y))
                edge_list.append((x, y))
                non_edge_list.append((u, v))
                
                max_b = new_max_b
                improved = True
                found_swap = True
                
                if max_b < best_score:
                    best_score = max_b
                    best_adj = adj.copy()
                
                break # Restart search with new graph state
        
        if not found_swap:
            # No single swap improves the max book size.
            # Try a 2-swap or accept a swap that keeps max same but reduces sum of squares?
            # For now, stop.
            break

    # Return best found
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                result.append([u, v])
    
    return result
