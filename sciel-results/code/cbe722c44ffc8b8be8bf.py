import time
import random

def construct():
    n = 40
    m = 401
    start_time = time.time()
    deadline = start_time + 115.0
    
    def try_greedy(B):
        # Returns edges set if successful, else None
        if time.time() > deadline - 5.0:
            return None
            
        edges = set()
        masks = [0] * n
        book_sizes = {}
        
        # Pre-generate all potential edges
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if len(edges) >= m:
                break
            
            # Check book size of new edge (u,v)
            # N(u) & N(v)
            common = masks[u] & masks[v]
            b_new = common.bit_count()
            if b_new > B:
                continue
            
            # Check neighbors of u
            valid = True
            # Iterate neighbors of u
            mask_u = masks[u]
            # We need to check edges (u, w) for w in N(u)
            # If we add v to N(u), book(u,w) increases if v in N(w)
            # i.e., if edge (v,w) exists
            # So for each w in N(u):
            #   if (u,w) is in edges (it is, by definition of N(u))
            #   if (v,w) is in edges (check mask_v & (1<<w))
            #   then book(u,w) + 1 > B?
            
            # Optimization: 
            # If we add (u,v), it creates triangles (u,v,w) for w in N(u) & N(v).
            # These triangles increase the book size of edges (u,w) and (v,w) by 1.
            # So we must ensure that for all w in N(u) & N(v):
            #   book(u,w) < B
            #   book(v,w) < B
            
            # Let T = N(u) & N(v)
            # For each w in T:
            #   if book_sizes[(min(u,w), max(u,w))] == B: fail
            #   if book_sizes[(min(v,w), max(v,w))] == B: fail
            
            # This is much faster than checking all neighbors.
            
            T = common
            while T:
                lsb = T & -T
                w = lsb.bit_length() - 1
                T ^= lsb
                
                e_uw = (min(u, w), max(u, w))
                e_vw = (min(v, w), max(v, w))
                
                # These edges must exist because w is in N(u) and N(v)
                if book_sizes.get(e_uw, 0) >= B:
                    valid = False
                    break
                if book_sizes.get(e_vw, 0) >= B:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Add edge
            edges.add((u, v))
            masks[u] |= 1 << v
            masks[v] |= 1 << u
            
            # Update book sizes for edges incident to u and v
            # Edges in T are (u,w) and (v,w). Their book size increases by 1.
            # Other edges incident to u or v do not change book size?
            # Book(u,w) = |N(u) & N(w)|.
            # N(u) gains v.
            # Intersection gains v if v in N(w).
            # v in N(w) iff (v,w) is edge.
            # If (v,w) is edge, then w is in N(v).
            # So w is in N(u) & N(v) = T.
            # So only edges (u,w) and (v,w) for w in T change.
            
            T = common # Reuse common mask
            while T:
                lsb = T & -T
                w = lsb.bit_length() - 1
                T ^= lsb
                
                e_uw = (min(u, w), max(u, w))
                e_vw = (min(v, w), max(v, w))
                
                book_sizes[e_uw] = book_sizes.get(e_uw, 0) + 1
                book_sizes[e_vw] = book_sizes.get(e_vw, 0) + 1
            
            # Store book size for new edge (u,v)
            book_sizes[(u, v)] = b_new
            
        if len(edges) == m:
            return edges
        return None

    # Binary search / Linear search for minimal B
    best_edges = None
    best_B = None
    
    # Theorem says B >= 40/6 ~ 6.6, so B >= 7.
    # Try B from 7 to 15
    for B in range(7, 16):
        if time.time() > deadline - 10.0:
            break
        
        # Try multiple times for each B to handle randomness in shuffle
        success = False
        for attempt in range(5):
            if time.time() > deadline - 10.0:
                break
            edges = try_greedy(B)
            if edges:
                best_edges = edges
                best_B = B
                success = True
                break
        
        if success:
            # We found a graph with max book <= B.
            # Since we are iterating B upwards, this is the minimal B for this greedy strategy.
            # We can break here as we found a valid solution.
            # However, maybe a random search could find B=6? Unlikely given theorem.
            # Let's verify the actual max book.
            break
            
    if best_edges is None:
        # Fallback to random + local search if greedy failed completely
        # Just return a random graph as last resort
        edges = set()
        while len(edges) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                edges.add((min(u, v), max(u, v)))
        best_edges = edges

    return [[u, v] for u, v in best_edges]
