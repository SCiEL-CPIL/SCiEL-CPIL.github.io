import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 40
    target_edges = 401
    
    # 1. Structured Initialization: Turán graph T(40, 2) is K20,20 (400 edges).
    # Add one random edge within a partition to reach 401 edges.
    # This starts with a very low triangle count (only 1 triangle initially if the new edge connects to nodes in the other partition? 
    # No, K20,20 is bipartite, so 0 triangles. Adding (0,1) creates triangles with all 20 nodes in partition B. 
    # Wait, book size of (0,1) would be 20. That's bad.
    # Better initialization: Start with a random graph but bias towards low clustering?
    # Actually, the Turan graph T(40, 3) has 400 edges? No.
    # T(40, 2) = K20,20 has 400 edges. Score is 0.
    # If we add 1 edge, say (0,1) in part A. Triangles are (0,1,w) for w in B. There are 20 such w.
    # So the book size of (0,1) is 20. The max book is 20.
    # We want to minimize the MAX book.
    # A random graph G(40, 401) has average degree 20.05.
    # Expected triangles ~ n * (p)^3 * C(n,3)? No.
    # Let's stick to a randomized start but maybe better than pure random?
    # Pure random is actually decent for finding low max-book structures because it spreads out.
    # However, the "heavy-edge first" search is the key differentiator.
    
    # Let's use a random start as it's robust, but optimize the search.
    vertices = list(range(n))
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    current_edges = set(all_pairs[:target_edges])
    
    # Use a numpy matrix for fast adjacency checks and book size calculation
    # adj[i, j] = 1 if edge exists
    adj = np.zeros((n, n), dtype=np.int8)
    
    def update_adj(edge, val):
        u, v = edge
        adj[u, v] = val
        adj[v, u] = val
        
    for e in current_edges:
        update_adj(e, 1)
        
    def get_books():
        # Returns a dict of edge -> book_size, or just the max and the list of heavy edges
        # Book size of edge (u,v) is number of common neighbors.
        # In matrix terms: sum(adj[u, :] * adj[v, :])
        # This is O(n) per edge. 401 * 40 = 16040 ops.
        # With numpy: 
        # We can compute the triangle count matrix T where T[u,v] = common neighbors.
        # T = adj @ adj - diag(adj) ... wait, adj@adj gives common neighbors for all pairs.
        # Let's compute A^2.
        A = adj.astype(np.int32)
        A2 = A @ A
        # The diagonal of A2 is the degree.
        # The off-diagonal A2[u,v] is the number of common neighbors of u and v.
        # This is exactly the book size for edge (u,v) if edge exists.
        # We only care about edges that exist.
        
        max_book = 0
        heavy_edges = []
        # Iterate over existing edges
        # To avoid iterating all 780 pairs, we can use the edge list
        # But we need to map back to indices.
        # Since n=40, iterating all pairs is 780, which is fast.
        # Let's just find the max among existing edges.
        
        # Mask for existing edges
        # We can just check the upper triangle
        
        # For efficiency in the loop, let's precompute the list of edges as indices
        # But we need to update this list on swaps.
        
        # Let's just do a vectorized max on the masked A2?
        # We need to ignore non-edges.
        # Create a mask of edges
        edge_mask = np.triu(adj, 1) # 1 where edge exists
        
        # Get book sizes for all potential edges
        books = np.triu(A2, 1)
        
        # For non-edges, book size is irrelevant, but we want max over edges.
        # Set non-edge books to 0
        books_masked = np.where(edge_mask == 1, books, 0)
        
        max_idx = np.argmax(books_masked)
        max_book = books_masked.flat[max_idx]
        
        # Identify heavy edges for targeted search
        # Edges with book size > max_book - 2 are "heavy"
        # We want to reduce the max, so we should focus on edges with the highest book sizes.
        
        # Find indices of edges with book size >= max_book - 1 (or just top few)
        # To keep it simple, let's find all edges with book size == max_book
        # And maybe some with max_book - 1
        
        # Convert flat index to (u, v)
        # This is done in the search loop to save time if we don't need the list immediately?
        # No, we need to pick a heavy edge to remove.
        
        # Let's collect a list of candidate edges to remove:
        # Those with book size >= max_book - 1
        threshold = max_book - 1
        # Get coordinates
        u_coords, v_coords = np.where(books_masked >= threshold)
        
        # Filter to only actual edges (though mask handled 0s, if max_book is 0, threshold is -1, all 0s pass? 
        # If max_book is 0, we are done.
        if max_book == 0:
            return 0, []
            
        candidates = [(u_coords[i], v_coords[i]) for i in range(len(u_coords)) if edge_mask[u_coords[i], v_coords[i]] == 1]
        
        return max_book, candidates

    current_max_book, _ = get_books()
    best_max_book = current_max_book
    best_edges = set(current_edges)
    
    # Search Strategy:
    # 1. Identify heavy edges (high book size).
    # 2. Remove a heavy edge.
    # 3. Add a random light edge (or random edge) to restore count.
    # 4. Accept if max book decreases, or with probability exp(-delta/T) if it stays same/increases slightly (to escape local minima).
    
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    # Pre-compute all possible non-edges for faster sampling?
    # Or just sample random pairs.
    
    while time.time() < deadline:
        current_max_book, heavy_candidates = get_books()
        
        if current_max_book == 0:
            break
            
        # If we have heavy candidates, pick one to remove
        if heavy_candidates:
            old_u, old_v = random.choice(heavy_candidates)
            old_edge = (old_u, old_v) if old_u < old_v else (old_v, old_u)
        else:
            # Fallback to random edge if no heavy ones found (shouldn't happen if max > 0)
            old_edge = random.choice(list(current_edges))
            
        # Remove old edge
        current_edges.remove(old_edge)
        update_adj(old_edge, 0)
        
        # Find a new edge to add
        # We want to add an edge that doesn't increase the max book too much.
        # Ideally, an edge with low book size potential.
        # But we don't know the book size until we add it.
        # Heuristic: Add an edge between nodes with low common neighbors?
        # Or just random.
        
        # Let's try a few random candidates and pick the one with lowest resulting max book?
        # That's expensive (multiple evaluations).
        # Let's just pick one random non-edge.
        
        new_edge = None
        attempts = 0
        while attempts < 20:
            u, v = random.sample(vertices, 2)
            candidate = (u, v) if u < v else (v, u)
            if candidate not in current_edges:
                new_edge = candidate
                break
            attempts += 1
            
        if new_edge is None:
            # Couldn't find a non-edge? Unlikely.
            # Revert
            current_edges.add(old_edge)
            update_adj(old_edge, 1)
            continue
            
        # Add new edge
        current_edges.add(new_edge)
        update_adj(new_edge, 1)
        
        new_max_book, _ = get_books()
        
        delta = new_max_book - current_max_book
        
        # Acceptance criteria
        if delta < 0:
            # Improvement
            current_max_book = new_max_book
            if new_max_book < best_max_book:
                best_max_book = new_max_book
                best_edges = set(current_edges)
        elif delta == 0:
            # Neutral move, accept to explore
            current_max_book = new_max_book
            # Update best if we want to keep track of different configs with same score?
            # Not strictly necessary for score, but might help escape.
            # Let's just accept.
        else:
            # Worse, accept with probability
            if random.random() < np.exp(-delta / T):
                current_max_book = new_max_book
            else:
                # Revert
                current_edges.remove(new_edge)
                update_adj(new_edge, 0)
                current_edges.add(old_edge)
                update_adj(old_edge, 1)
                
        T *= alpha
        if T < T_min:
            T = T_min
            
    return [[u, v] for u, v in best_edges]
