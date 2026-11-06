import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 40
    m = 401
    
    # Initial construction: K20,20 + 1 edge
    # Vertices 0-19 in A, 20-39 in B
    edges = set()
    for a in range(20):
        for b in range(20, 40):
            edges.add((a, b))
    # Add one intra-partition edge to reach 401
    # Let's pick (0, 1) initially
    edges.add((0, 1))
    
    # Build adjacency matrix for fast lookups
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
    
    # Helper to compute book size (common neighbors) for an edge
    def get_book_size(u, v):
        return np.sum(adj[u] & adj[v])
    
    # Compute current max book
    def compute_max_book():
        max_b = 0
        for u, v in edges:
            b = get_book_size(u, v)
            if b > max_b:
                max_b = b
        return max_b

    current_max_book = compute_max_book()
    best_max_book = current_max_book
    best_edges = set(edges)
    
    # Deterministic Greedy Local Search
    # Strategy: 
    # 1. Find edge e_max with maximum book size.
    # 2. Try to swap e_max with a candidate non-edge e_new.
    # 3. Candidate selection: 
    #    - We want to remove the "heavy" edge.
    #    - We want to add an edge that doesn't create a new heavy edge or increases max book less.
    #    - A good heuristic is to add an edge between two vertices that are "far" in terms of common neighbors, 
    #      or simply try a set of random non-edges and pick the one that yields the lowest resulting max book.
    
    # To make it deterministic but effective, we will:
    # 1. Identify the worst edge.
    # 2. Generate a pool of candidate non-edges (e.g., random sample + some structured ones).
    # 3. For each candidate, simulate the swap and compute the new max book.
    # 4. If the new max book is strictly smaller, accept.
    # 5. If no improvement, try a "kick" (random swap that might increase max book slightly) to escape local minima, 
    #      but only if we are stuck for a while.
    
    stuck_count = 0
    
    while time.time() < deadline:
        # 1. Find the edge with the maximum book size
        worst_edge = None
        worst_size = -1
        for u, v in edges:
            b = get_book_size(u, v)
            if b > worst_size:
                worst_size = b
                worst_edge = (u, v)
        
        if worst_size == 0:
            break # Optimal found (theoretical min is 7, but 0 is impossible for n=40, m=401? Actually min is 7. If we hit 7, we stop.)
        
        # If we found the theoretical minimum (7), we can stop early?
        # The theorem says >= 7. If we hit 7, it's optimal.
        if worst_size <= 7:
            break
            
        # 2. Generate candidates for the new edge
        # We remove worst_edge (u,v).
        # We need to add a non-edge (a,b).
        # To be fast, we sample a few random non-edges.
        # Also, consider edges that are "structurally similar" but less crowded?
        
        candidates = []
        # Random sampling of non-edges
        # There are ~780 non-edges. Sampling 50-100 is fast.
        for _ in range(100):
            a, b = random.sample(range(n), 2)
            if not adj[a, b]:
                candidates.append((a, b))
        
        # Also, specifically try edges within the same partition that are "isolated"
        # In K20,20 + 1, the intra-partition edges are the ones creating the triangles?
        # Wait, in K20,20, there are no triangles.
        # Adding (0,1) creates triangles (0,1,x) for x in N(0) ∩ N(1) = B (all 20 vertices).
        # So book size of (0,1) is 20.
        # The edges in K20,20 have book size 0 initially?
        # Edge (a,b) with a in A, b in B. Common neighbors: 
        # N(a) = B, N(b) = A. Intersection is empty.
        # So all bipartite edges have book size 0.
        # The single intra-edge (0,1) has book size 20.
        # So max book is 20.
        
        # If we remove (0,1) and add another intra-edge (2,3), book size of (2,3) is 20. Max is still 20.
        # To reduce max book, we need to break the structure where one edge has huge common neighbors.
        # We need to "spread out" the triangles.
        # This requires adding multiple intra-partition edges or breaking some bipartite edges.
        
        # The greedy swap:
        # Remove (u,v).
        # Add (a,b).
        # Check new max book.
        
        best_candidate = None
        best_new_max = worst_size + 1 # Must be strictly better than current worst
        
        for a, b in candidates:
            # Simulate swap
            # Remove (u,v)
            adj[u, v] = False
            adj[v, u] = False
            # Add (a,b)
            adj[a, b] = True
            adj[b, a] = True
            
            # Compute new max book
            # Optimization: Only edges incident to u, v, a, b change book size?
            # No, adding (a,b) changes N(a) and N(b).
            # Removing (u,v) changes N(u) and N(v).
            # Edges whose book size changes are those incident to u, v, a, b.
            # Plus, the new edge (a,b) itself.
            # And the removed edge (u,v) is gone.
            
            # Let's just compute max over all edges for simplicity in this prototype, 
            # as n=40 is small. 401 edges * 40x40 matrix op is fast.
            new_max = 0
            for x, y in edges:
                if x == u and y == v: continue # Removed
                if x == a and y == b: 
                    # New edge
                    pass
                # Actually, edges set needs to be updated for the loop to be correct.
                # Let's do a proper simulation.
                
            # Proper simulation:
            # Temporarily update edges set? No, too slow.
            # Just iterate.
            
            # Re-evaluate max book efficiently
            # The book size of an edge (x,y) is |N(x) ∩ N(y)|.
            # N(x) changed if x is u, v, a, or b.
            # So only edges incident to u, v, a, b need re-evaluation, plus the new edge (a,b).
            # Other edges have unchanged N(x) and N(y).
            
            # Let's collect the set of edges to check.
            check_edges = set()
            for z in range(n):
                if adj[u, z]: check_edges.add((min(u,z), max(u,z)))
                if adj[v, z]: check_edges.add((min(v,z), max(v,z)))
                if adj[a, z]: check_edges.add((min(a,z), max(a,z)))
                if adj[b, z]: check_edges.add((min(b,z), max(b,z)))
            
            # Add the new edge
            check_edges.add((min(a,b), max(a,b)))
            
            # Remove the old edge from consideration (it's not in adj anymore, so it won't be in check_edges from adj, 
            # but we need to ensure we don't pick it up if it was in the set before. 
            # Actually, check_edges is built from CURRENT adj (which has u,v removed, a,b added).
            # So (u,v) is not in check_edges unless it's incident to a or b? 
            # If u==a, then it's incident. But u,v removed.
            # So we just compute max over check_edges.
            
            max_b = 0
            for x, y in check_edges:
                # Ensure it's a valid edge in the new graph
                if adj[x, y]:
                    b_val = get_book_size(x, y)
                    if b_val > max_b:
                        max_b = b_val
            
            # Revert adj
            adj[u, v] = True
            adj[v, u] = True
            adj[a, b] = False
            adj[b, a] = False
            
            if max_b < best_new_max:
                best_new_max = max_b
                best_candidate = (a, b)
        
        if best_candidate is not None:
            # Apply the best swap
            a, b = best_candidate
            u, v = worst_edge
            
            adj[u, v] = False
            adj[v, u] = False
            adj[a, b] = True
            adj[b, a] = True
            
            edges.discard(worst_edge)
            edges.add((a, b))
            
            current_max_book = best_new_max
            if current_max_book < best_max_book:
                best_max_book = current_max_book
                best_edges = set(edges)
            
            stuck_count = 0
        else:
            # No improvement found in this round
            stuck_count += 1
            
            # If stuck, perform a random kick to escape local minimum
            if stuck_count > 10:
                # Pick a random edge to remove and random non-edge to add
                e_remove = random.choice(list(edges))
                u, v = e_remove
                
                added_edge = None
                for _ in range(50):
                    a, b = random.sample(range(n), 2)
                    if not adj[a, b]:
                        added_edge = (a, b)
                        break
                
                if added_edge:
                    a, b = added_edge
                    adj[u, v] = False
                    adj[v, u] = False
                    adj[a, b] = True
                    adj[b, a] = True
                    
                    edges.discard(e_remove)
                    edges.add(added_edge)
                    
                    # Update current_max_book (approximation or full calc)
                    current_max_book = compute_max_book()
                    
                    if current_max_book < best_max_book:
                        best_max_book = current_max_book
                        best_edges = set(edges)
                    
                    stuck_count = 0
        
        # Safety check for time
        if time.time() > deadline - 5:
            break

    # Return the best found
    result = [[u, v] for u, v in best_edges]
    
    # Sanity check
    if len(result) != 401:
        # Fallback to initial construction if something went wrong
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return result
