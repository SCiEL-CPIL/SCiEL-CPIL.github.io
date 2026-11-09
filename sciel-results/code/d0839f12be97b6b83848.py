import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # Try to find a graph for n = 39 down to 27
    # If we find a valid graph for n, return it.
    # We prioritize higher n.
    
    # Known: R(3,10) > 30 is established.
    # Let's try to find one for n=30, 31, ... 39.
    
    best_n = 0
    best_edges = None
    
    # Helper: Check if graph is valid (triangle-free and alpha < 10)
    # We use bitmasks for efficiency.
    
    def is_valid_graph(n, edges):
        # Build adjacency bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Check triangle-free
        # A triangle exists if there is u < v < w such that u-v, v-w, u-w are edges.
        # Efficient check: For each edge (u, v), check if adj[u] & adj[v] has any bit set.
        # Actually, if adj[u] & adj[v] != 0, there is a common neighbor.
        # Since we build from edges, we just need to ensure no triangle.
        # Faster: During construction, we ensure no triangles. 
        # But for verification of a candidate:
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    # If u and v are connected, they must have no common neighbors
                    if adj[u] & adj[v]:
                        return False, None
        
        # Check alpha < 10
        # Branch and Bound
        # We want to know if there is an independent set of size 10.
        # If yes, invalid. If no, valid.
        
        # Order vertices by degree (ascending? descending?)
        # For independent set, low degree vertices are "harder" to include?
        # Actually, standard B&B for Max Independent Set:
        # Sort by degree descending.
        
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda i: -degrees[i])
        
        # Map new index to old index
        # We work on the ordered list.
        # adj_ordered[i] is the bitmask of neighbors in the original graph, 
        # but we need to map it to the new indices?
        # Easier: Keep original indices, but iterate in sorted order.
        
        # Let's use a recursive B&B on the set of available vertices.
        # available_mask: bits for vertices we can still pick.
        # current_set_size: size of independent set found so far.
        
        def has_independent_set_10(available_mask, current_size):
            if current_size >= 10:
                return True
            
            if available_mask == 0:
                return False
            
            # Bound: if current_size + popcount(available_mask) < 10, return False
            if current_size + bin(available_mask).count('1') < 10:
                return False
            
            # Pick a vertex
            # Heuristic: Pick vertex with lowest degree in the induced subgraph?
            # Or just pick the first one in sorted order?
            # Let's pick the vertex with the smallest index in `order` that is in `available_mask`.
            
            # To speed up, we can precompute the position of each vertex in the order.
            # But `order` is fixed.
            
            # Find the first vertex in `order` that is in `available_mask`
            # This might be slow if we do it naively.
            
            # Alternative: Iterate through vertices in `order`.
            # For each vertex v in order:
            #   if v is in available_mask:
            #       # Option 1: Include v
            #       # New available: available_mask & ~adj[v] & ~(1<<v)
            #       # Option 2: Exclude v
            #       # New available: available_mask & ~(1<<v)
            
            # To make it faster, we can use the property that if we include v, 
            # we remove v and its neighbors.
            
            # Let's iterate v in order
            for v in order:
                if (available_mask >> v) & 1:
                    # Include v
                    new_avail = available_mask & ~(1 << v) & ~adj[v]
                    if has_independent_set_10(new_avail, current_size + 1):
                        return True
                    
                    # Exclude v
                    available_mask &= ~(1 << v)
                    
                    # Prune: if current_size + popcount(available_mask) < 10
                    if current_size + bin(available_mask).count('1') < 10:
                        return False
                    
                    break # Only consider the first available vertex in the order?
                    # No, we must consider the branch where we exclude v and pick the next one.
                    # The loop continues to the next v if we didn't return.
                    # But we modified available_mask.
                    # This is a standard sequential branching.
            
            return False
        
        if has_independent_set_10((1 << n) - 1, 0):
            return False, None
        
        return True, edges

    # Search for n = 39 down to 27
    # We start with higher n to maximize score.
    for n in range(39, 26, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this n
        found = False
        for restart in range(500): # Max 500 restarts per n
            if time.time() > deadline:
                break
                
            # Initialize with a random triangle-free graph
            # Strategy: Start with empty, add random edges if no triangle.
            # This tends to produce sparse graphs.
            # Better: Start with a random graph, remove edges to break triangles.
            
            # Let's try: Random graph G(n, 0.5)
            adj = [0] * n
            edges = []
            
            # Generate random edges
            # We want to ensure triangle-free.
            # Iterative removal is safer.
            
            # Start with all possible edges
            # This is O(n^2) memory, n=39 is small.
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.append([u, v])
                        
            # Remove triangles
            # Loop until no triangles
            # To speed up, we can check triangles and remove one.
            # But checking all triangles is O(n^3) or O(E * degree).
            
            # Faster: 
            # For each u, for each v in adj[u], check if adj[u] & adj[v] != 0.
            # If yes, there is a triangle. Remove an edge.
            
            changed = True
            while changed:
                changed = False
                for u in range(n):
                    if time.time() > deadline:
                        break
                    # Get neighbors
                    # Iterate neighbors v
                    v_mask = adj[u]
                    while v_mask:
                        # Extract lowest set bit
                        lsb = v_mask & -v_mask
                        v = lsb.bit_length() - 1
                        v_mask ^= lsb
                        
                        if u > v: continue # Process each edge once? No, adj is symmetric.
                        # Check common neighbors
                        common = adj[u] & adj[v]
                        if common:
                            # There is a triangle u-v-w
                            # Pick a random edge to remove
                            # Options: (u,v), (u,w), (v,w)
                            # Let's remove (u,v) for simplicity, or random.
                            # Removing (u,v) is easy.
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            # Remove from edges list? 
                            # We can rebuild edges list at the end.
                            changed = True
                            break # Restart inner loop? Or continue?
                            # If we break, we restart the u loop?
                            # Let's just break the while loop and restart.
                            # Actually, breaking the while v_mask loop and continuing u is fine,
                            # but we might miss other triangles.
                            # Let's restart the whole while changed loop? No, too slow.
                            # Just break v_mask loop.
                    if changed:
                        break # Restart u loop? Or just continue?
                        # If we removed an edge, the graph changed.
                        # It's safer to restart the scan.
                
                if time.time() > deadline:
                    break
            
            # Rebuild edges list
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            
            # Check validity
            valid, _ = is_valid_graph(n, edges)
            if valid:
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            # We found a graph for this n. 
            # Since we iterate n downwards, this is the max n found so far.
            # We can stop if we want, or try to see if we can find a larger one?
            # No, we started from 39. So if we found one for 35, we should check 39, 38, 37, 36 first.
            # The loop `for n in range(39, 26, -1)` handles this.
            # If we find a valid graph for n=35, we return it?
            # No, we should continue to see if 36, 37, 38, 39 are possible?
            # Wait, the loop goes 39, 38, ..., 27.
            # If we find a valid graph for 39, we break out of the n loop?
            # Yes, because 39 is the max possible (R(3,10)<=41, but 40/41 might be hard).
            # Actually, R(3,10) is between 39 and 41.
            # So a graph on 39 vertices with alpha < 10 proves R(3,10) > 39.
            # A graph on 40 vertices proves R(3,10) > 40.
            # A graph on 41 vertices proves R(3,10) > 41 (which contradicts R(3,10)<=41).
            # So max n is 40.
            
            # We should break if we find a valid graph for the current n?
            # No, we want the MAXIMUM n.
            # So if we find a valid graph for n=30, we should still try n=31...39.
            # But the loop is `for n in range(39, 26, -1)`.
            # So it tries 39 first. If it finds one, it sets best_n=39 and `found=True`.
            # Then it breaks the restart loop.
            # Then it should break the n loop?
            # Yes, because we are iterating downwards. If we found a valid graph for 39, that's the best.
            # If we didn't find one for 39, we try 38, etc.
            # So if `found` is True, we can break the outer loop.
            break

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback: Return the Andrasfai graph (n=26)
        n = 26
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n % 3 == 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
