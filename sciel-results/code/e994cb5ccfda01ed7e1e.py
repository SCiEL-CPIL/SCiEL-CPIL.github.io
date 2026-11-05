import random
import time

def construct():
    n = 20
    best_edges = []
    best_count = 0
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer

    def check_k33_for_edge(u, v, masks):
        """
        Check if adding edge (u, v) to a graph defined by masks creates a K3,3.
        Assumes (u, v) is NOT currently in the graph.
        """
        # Case 1: u and v are in the same part of K3,3
        # Need w such that w is connected to u and v.
        # And 3 vertices connected to u, v, and w.
        common_uv = masks[u] & masks[v]
        # w must be in common_uv
        # For each w in common_uv:
        #   C = (masks[u] & masks[v] & masks[w]) & ~(1 << u) & ~(1 << v) & ~(1 << w)
        #   if popcount(C) >= 3: return True
        
        # Optimization: 
        # common_uv contains all candidates for w.
        # We can iterate w.
        # To be faster, we can check if there exists ANY w such that 
        # popcount(masks[u] & masks[v] & masks[w]) >= 3 (excluding u,v,w).
        
        # Actually, if we add (u,v), u and v become connected.
        # The neighbors of u and v are fixed.
        
        w_mask = common_uv
        while w_mask:
            w = (w_mask & -w_mask).bit_length() - 1
            w_mask &= w_mask - 1
            # C is common neighbors of u, v, w
            c = masks[u] & masks[v] & masks[w]
            # Exclude u, v, w
            c &= ~(1 << u)
            c &= ~(1 << v)
            c &= ~(1 << w)
            if bin(c).count('1') >= 3:
                return True

        # Case 2: u and v are in different parts of K3,3
        # Need a1, a2 in N(v) \ {u} and b1, b2 in N(u) \ {v}
        # such that a1, a2 connected to b1, b2.
        
        # Let S_v = N(v) without u
        # Let S_u = N(u) without v
        
        sv_mask = masks[v] & ~(1 << u)
        su_mask = masks[u] & ~(1 << v)
        
        # We need to find a1, a2 in sv_mask such that 
        # | (N(a1) & N(a2) & su_mask) | >= 2
        
        # Iterate a1 in sv_mask
        a1_mask = sv_mask
        while a1_mask:
            a1 = (a1_mask & -a1_mask).bit_length() - 1
            a1_mask &= a1_mask - 1
            
            # a2 must be in sv_mask, a2 != a1
            a2_mask = sv_mask & ~(1 << a1)
            
            # We can optimize by iterating a2
            while a2_mask:
                a2 = (a2_mask & -a2_mask).bit_length() - 1
                a2_mask &= a2_mask - 1
                
                # Common neighbors of a1, a2 in su_mask
                c = masks[a1] & masks[a2] & su_mask
                if bin(c).count('1') >= 2:
                    return True

        return False

    def greedy_construction(order):
        """
        Build graph by adding vertices in 'order'.
        For each vertex v, connect to previous vertices u if valid.
        """
        masks = [0] * n
        edges = []
        
        for v in order:
            # Try to connect v to all previous vertices
            # We can try in reverse order of addition or random
            # Let's try all previous vertices
            for i in range(order.index(v)):
                u = order[i]
                # Check if adding (u, v) is valid
                if check_k33_for_edge(u, v, masks):
                    continue
                # Add edge
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                edges.append((u, v))
        
        # After processing all vertices in order, try to add any remaining edges?
        # The greedy step above only considers edges between v and previous vertices.
        # This builds a graph where edges only go from later to earlier in order?
        # No, it adds edges between v and any u that appeared before v.
        # This ensures we consider all pairs exactly once.
        
        return edges

    # Main loop: Try many random orderings
    while time.time() - start_time < time_limit:
        # Generate a random permutation
        order = list(range(n))
        random.shuffle(order)
        
        # Run greedy construction
        edges = greedy_construction(order)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [list(e) for e in edges]
            
        # Optional: Local improvement on the best found so far?
        # Or just keep generating.
        # Greedy construction is very fast. We can do many iterations.
        
    return best_edges
