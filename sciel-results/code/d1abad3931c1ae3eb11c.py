import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n = 40
    m = 401
    
    # 1. Initial construction: K_{20,20} + one internal edge (0, 1)
    # This is the unique graph with 401 edges that minimizes the max degree imbalance 
    # in the bipartition sense, but we need to optimize the book size.
    # Edges in K_{20,20}: all (i, j) where i < 20 <= j
    # Plus (0, 1)
    
    edges_set = set()
    for i in range(20):
        for j in range(20, 40):
            edges_set.add((i, j))
    edges_set.add((0, 1))
    
    # Verify count
    if len(edges_set) != m:
        raise ValueError(f"Expected {m} edges, got {len(edges_set)}")
    
    # Data structures for fast evaluation
    # adj[u] is a set of neighbors
    adj = [set() for _ in range(n)]
    for u, v in edges_set:
        adj[u].add(v)
        adj[v].add(u)
        
    # book_size[e] is the number of triangles containing edge e
    # We store edges as tuples (u, v) with u < v for consistency
    book_size = {}
    
    def update_book(u, v):
        """Calculate and store book size for edge (u, v). Assumes u < v."""
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        book_size[(u, v)] = len(common)
        
    # Initialize book sizes
    for u, v in edges_set:
        if u > v:
            u, v = v, u
        update_book(u, v)
        
    # Helper to get book size for an edge (handles ordering)
    def get_book(u, v):
        if u > v:
            u, v = v, u
        return book_size.get((u, v), 0)
        
    # Helper to get max book size
    def get_max_book():
        return max(book_size.values()) if book_size else 0

    best_edges = set(edges_set)
    best_max_book = get_max_book()
    
    # Local Search: Simulated Annealing
    # Move: Swap an edge e_in (in graph) with e_out (not in graph)
    # We want to minimize max book size.
    
    current_max_book = best_max_book
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Precompute all non-edges for sampling? 
    # Total edges possible: 40*39/2 = 780.
    # Non-edges: 780 - 401 = 379.
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    non_edges = set(all_possible_edges) - edges_set
    
    # To speed up swaps, we can pick a random edge from the graph and a random non-edge.
    # However, naive swap might disconnect or violate constraints? No, just swaps.
    
    # Optimization: Instead of recalculating max every time (O(M)), 
    # we can track the distribution or just recalculate if the affected edges change significantly.
    # Given N=40, M=401, recalculating max is O(400) which is fast.
    # Updating book sizes for affected edges is the main cost.
    
    # Affected edges when swapping (u1, v1) with (u2, v2):
    # Removing (u1, v1) changes book sizes of all edges incident to u1 or v1?
    # No, book size of edge (a, b) depends on common neighbors.
    # If we remove edge (u1, v1), u1 and v1 lose a neighbor.
    # This affects book_size of any edge (x, y) where x or y is u1 or v1?
    # Actually, book_size(a, b) = |N(a) ∩ N(b)|.
    # If we remove (u1, v1), then u1 is no longer in N(v1) and v1 is no longer in N(u1).
    # This changes N(u1) and N(v1).
    # Therefore, book_size(a, b) changes if:
    # 1. (a, b) is incident to u1 or v1? 
    #    If a=u1, N(a) changed. So book_size(u1, b) changes for all b in N(u1).
    #    If a=v1, N(a) changed. So book_size(v1, b) changes for all b in N(v1).
    # 2. Does it affect non-incident edges?
    #    book_size(a, b) = |N(a) ∩ N(b)|.
    #    N(a) and N(b) only change if a or b is u1 or v1.
    #    So only edges incident to u1 or v1 are affected by the removal of (u1, v1).
    #    Similarly for addition of (u2, v2).
    
    # So, for a swap (remove e1, add e2), we need to update book sizes for:
    # - All edges incident to endpoints of e1
    # - All edges incident to endpoints of e2
    # This is roughly deg(u1)+deg(v1)+deg(u2)+deg(v2) updates.
    # Average degree is ~20. So ~80 updates per swap.
    # Each update involves set intersection of size ~20.
    # This is very fast.
    
    steps = 0
    while time.time() < deadline:
        # Pick random edge to remove
        e1 = random.choice(list(edges_set))
        u1, v1 = e1
        
        # Pick random non-edge to add
        e2 = random.choice(list(non_edges))
        u2, v2 = e2
        
        # Ensure e1 != e2 (guaranteed by sets)
        
        # Perform Swap
        # 1. Remove e1
        edges_set.remove(e1)
        non_edges.add(e1)
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        
        # Update book sizes for edges incident to u1 or v1
        affected_edges = set()
        for neighbor in list(adj[u1]):
            affected_edges.add((min(u1, neighbor), max(u1, neighbor)))
        for neighbor in list(adj[v1]):
            affected_edges.add((min(v1, neighbor), max(v1, neighbor)))
            
        # Also, e1 itself is removed, so its book size entry should be removed or ignored.
        # But we are recalculating for remaining edges.
        
        # 2. Add e2
        edges_set.add(e2)
        non_edges.remove(e2)
        adj[u2].add(v2)
        adj[v2].add(u2)
        
        # Update book sizes for edges incident to u2 or v2
        for neighbor in list(adj[u2]):
            affected_edges.add((min(u2, neighbor), max(u2, neighbor)))
        for neighbor in list(adj[v2]):
            affected_edges.add((min(v2, neighbor), max(v2, neighbor)))
            
        # Recalculate book sizes for affected edges
        for e in affected_edges:
            if e in edges_set:
                update_book(e[0], e[1])
            else:
                # If it was removed, it shouldn't be in book_size anymore?
                # Actually, we only care about edges in the graph.
                # If e is not in edges_set, we don't care about its book size for the max calculation.
                # But if it was in book_size before, we should delete it to keep dict clean?
                # Or just ignore it in max calculation.
                # Let's delete to be safe and clean.
                if e in book_size:
                    del book_size[e]
                    
        # Calculate new max book size
        new_max_book = get_max_book()
        
        # Acceptance Criteria
        delta = new_max_book - current_max_book
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            # Accept move
            current_max_book = new_max_book
            if new_max_book < best_max_book:
                best_max_book = new_max_book
                best_edges = set(edges_set)
                # Optional: save time if we hit theoretical lower bound?
                # Theorem says >= 40/6 = 6.66, so integer >= 7.
                # If we hit 7, we can stop?
                # "The least achievable value for this finite size is not documented."
                # But 7 is the lower bound from the theorem for n=40, m=401.
                # If we find 7, it's likely optimal.
                if best_max_book == 7:
                    break
        else:
            # Reject move: Revert
            # Re-add e1
            edges_set.add(e1)
            non_edges.remove(e1)
            adj[u1].add(v1)
            adj[v1].add(u1)
            
            # Remove e2
            edges_set.remove(e2)
            non_edges.add(e2)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            # Recalculate book sizes for affected edges (same set)
            for e in affected_edges:
                if e in edges_set:
                    update_book(e[0], e[1])
                else:
                    if e in book_size:
                        del book_size[e]
                        
            # current_max_book remains unchanged (should be consistent)
            # Just in case, recalc? It's expensive. 
            # Since we reverted, the state is identical to before the swap.
            # So current_max_book is still correct.
            
        T *= cooling_rate
        steps += 1
        
    # Return best found
    return [[u, v] for u, v in best_edges]

