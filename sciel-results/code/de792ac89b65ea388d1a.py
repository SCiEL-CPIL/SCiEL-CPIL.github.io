import time
import random
import sys

def construct():
    deadline = time.time() + 145.0  # Safety margin
    
    def is_triangle_free(adj):
        """Check if graph is triangle-free using adjacency bitmasks."""
        for i in range(16):
            # For each edge (i, j), check if they share a common neighbor
            # adj[i] contains neighbors of i
            # We need to check if any neighbor j of i is also a neighbor of any other neighbor k of i?
            # No, triangle is i-j-k-i.
            # If (i,j) is an edge, then N(i) and N(j) must be disjoint (excluding i and j).
            # In bitmask: adj[i] & adj[j] should be 0.
            # Note: adj[i] does NOT contain i. adj[j] does NOT contain j.
            # If i is in adj[j] and j is in adj[i], that's fine.
            # But if k is in adj[i] AND k is in adj[j], then i-j-k-i is a triangle.
            # So we check adj[i] & adj[j] == 0.
            neighbors_i = adj[i]
            while neighbors_i:
                lsb = neighbors_i & (-neighbors_i)
                j = lsb.bit_length() - 1
                neighbors_i ^= lsb
                if adj[i] & adj[j]:
                    return False
        return True

    def is_3_colorable(adj):
        """Check if graph is 3-colorable using fast backtracking with precomputed conflict matrix.
        Returns True if 3-colorable, False if not (i.e., 4-chromatic or higher).
        """
        # Order vertices by degree (descending) to prune early
        degrees = [bin(adj[i]).count('1') for i in range(16)]
        order = sorted(range(16), key=lambda v: -degrees[v])
        
        # Precompute neighbor masks in the order
        order_neighbors = [adj[v] for v in order]
        
        colors = [-1] * 16
        color_masks = [0, 0, 0]
        
        def backtrack(pos):
            if pos == 16:
                return True
            
            v = order[pos]
            nmask = order_neighbors[pos]
            
            # Determine available colors
            forbidden = 0
            if nmask & color_masks[0]:
                forbidden |= 1
            if nmask & color_masks[1]:
                forbidden |= 2
            if nmask & color_masks[2]:
                forbidden |= 4
            
            available = 7 & (~forbidden)
            
            if available == 0:
                return False
            
            # Try each available color
            while available:
                lsb = available & (-available)
                c = 0 if lsb == 1 else (1 if lsb == 2 else 2)
                available ^= lsb
                
                color_masks[c] |= (1 << v)
                
                if backtrack(pos + 1):
                    return True
                
                color_masks[c] &= ~(1 << v)
            
            return False
        
        return backtrack(0)

    def count_edges(adj):
        count = 0
        for i in range(16):
            count += bin(adj[i]).count('1')
        return count // 2

    def adj_to_edges(adj):
        edges = []
        for i in range(16):
            for j in range(i + 1, 16):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def build_grotzsch():
        """Build the Grötzsch graph (Mycielski of C5) on vertices 0-10.
        Vertices 11-15 are isolated.
        """
        adj = [0] * 16
        # C5 on vertices 0-4
        for i in range(5):
            adj[i] |= (1 << ((i + 1) % 5))
            adj[(i + 1) % 5] |= (1 << i)
        
        # Mycielski: add vertices 5-9 as copies of 0-4, and vertex 10 as universal to copies
        for i in range(5):
            for j in range(5):
                if adj[i] & (1 << j):
                    adj[5 + i] |= (1 << j)
                    adj[j] |= (1 << (5 + i))
            adj[5 + i] |= (1 << 10)
            adj[10] |= (1 << (5 + i))
        
        return adj

    def try_add_edge(adj, u, v):
        """Try to add edge (u,v) if it doesn't create a triangle. Returns True if added."""
        if adj[u] & (1 << v):
            return False
        common = adj[u] & adj[v]
        if common:
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    def try_remove_edge(adj, u, v):
        """Remove edge (u,v). Returns True if removed."""
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return True
        return False

    def local_search(adj, deadline):
        """Local search to maximize edges while maintaining triangle-free and 4-chromatic."""
        current = adj[:]
        current_count = count_edges(current)
        
        # Initial check
        if not is_triangle_free(current) or is_3_colorable(current):
            return current, current_count
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Generate all possible edges not in current
            possible = []
            for i in range(16):
                for j in range(i + 1, 16):
                    if not (current[i] & (1 << j)):
                        # Check triangle-free condition
                        common = current[i] & current[j]
                        if not common:
                            possible.append((i, j))
            
            # Shuffle to avoid bias
            random.shuffle(possible)
            
            for u, v in possible:
                if time.time() > deadline:
                    break
                
                # Try adding
                current[u] |= (1 << v)
                current[v] |= (1 << u)
                
                # Check 3-colorability
                if not is_3_colorable(current):
                    current_count += 1
                    improved = True
                    # If improved, we might want to restart the loop to re-evaluate all edges
                    # but for now, continue scanning
                else:
                    # Revert
                    current[u] &= ~(1 << v)
                    current[v] &= ~(1 << u)
        
        return current, current_count

    best_adj = None
    best_count = 0
    
    # Start with Grotzsch graph
    grotzsch = build_grotzsch()
    if is_triangle_free(grotzsch) and not is_3_colorable(grotzsch):
        maxed, cnt = local_search(grotzsch, deadline)
        if cnt > best_count:
            best_count = cnt
            best_adj = maxed[:]
    
    # Try some random restarts if time permits
    # Generate random triangle-free graphs and see if any are 4-chromatic
    # This is rare, so we might skip or do very few
    
    # A better restart: Perturb the best_adj
    # Remove a random edge, then run local search again
    
    if best_adj is not None and time.time() < deadline - 10:
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            perturbed = best_adj[:]
            # Remove 1-3 random edges
            edges = adj_to_edges(perturbed)
            if not edges:
                continue
            for _ in range(2):
                u, v = random.choice(edges)
                try_remove_edge(perturbed, u, v)
            
            if is_triangle_free(perturbed) and not is_3_colorable(perturbed):
                maxed, cnt = local_search(perturbed, deadline)
                if cnt > best_count:
                    best_count = cnt
                    best_adj = maxed[:]

    if best_adj is None:
        # Fallback: Return Grotzsch (20 edges)
        return adj_to_edges(build_grotzsch())
    
    return adj_to_edges(best_adj)

# EVOLVE-BLOCK-END
