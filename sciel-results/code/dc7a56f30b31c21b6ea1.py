"""Optimized Tomita Max Clique algorithm on the complement graph."""
import sys

def construct():
    n = 256
    # Build conflict graph adjacency masks
    # conflict_mask[i] has bit j set if (i+1)*(j+1) % ((i+1)+(j+1)) == 0
    conflict_mask = [0] * n
    
    for a in range(1, n + 1):
        i = a - 1
        for b in range(a + 1, n + 1):
            j = b - 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
                
    # Build complement graph adjacency masks
    # An edge in complement graph means NO conflict.
    # comp_mask[i] has bit j set if i != j and no conflict between i and j.
    all_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        # All vertices except i, minus those that conflict with i
        comp_mask[i] = all_mask & ~(1 << i) & ~conflict_mask[i]
        
    # Precompute degrees in the complement graph for ordering
    comp_degrees = [comp_mask[i].bit_count() for i in range(n)]
    
    best_clique = []
    best_size = 0
    
    def color_sort(candidates_mask):
        """
        Greedy coloring of the subgraph induced by candidates_mask in the complement graph.
        Uses bitmask optimization for speed.
        Returns a list of (vertex, color) sorted by color ascending.
        """
        if candidates_mask == 0:
            return []
        
        # Extract vertices from mask
        verts = []
        m = candidates_mask
        while m:
            v_bit = m & (-m)
            v = v_bit.bit_length() - 1
            verts.append(v)
            m &= ~v_bit
            
        # Sort vertices by global degree in complement graph (descending)
        # This heuristic helps produce a tighter coloring bound faster
        verts.sort(key=lambda x: comp_degrees[x], reverse=True)
        
        color_masks = []
        colors = {}
        
        for v in verts:
            v_bit = 1 << v
            assigned = False
            # Try to assign the lowest possible color
            for k in range(len(color_masks)):
                # Check if v is non-adjacent to all vertices in color class k in G_comp
                # If (comp_mask[v] & color_masks[k]) == 0, then v has no neighbors in this class
                if (comp_mask[v] & color_masks[k]) == 0:
                    color_masks[k] |= v_bit
                    colors[v] = k + 1
                    assigned = True
                    break
            
            if not assigned:
                color_masks.append(v_bit)
                colors[v] = len(color_masks)
        
        # Create list of (vertex, color)
        result = [(v, colors[v]) for v in verts]
        # Sort by color ascending for the expand function
        result.sort(key=lambda x: x[1])
        return result

    def expand(candidates_mask, clique):
        nonlocal best_clique, best_size
        
        if candidates_mask == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
            
        # Get colored vertices
        colored_vertices = color_sort(candidates_mask)
        
        # Process in reverse order (highest color first)
        # The list is sorted by color ascending.
        for i in range(len(colored_vertices) - 1, -1, -1):
            v, color = colored_vertices[i]
            
            # Check if v is still in candidates
            if not (candidates_mask & (1 << v)):
                continue
                
            # Upper bound check
            # The maximum size of a clique in the subgraph induced by 
            # vertices with color <= color is at most 'color'.
            # So the max size we can achieve from this branch is len(clique) + color.
            if len(clique) + color <= best_size:
                break # All remaining vertices have lower or equal color, so they can't improve
                
            # Branch: include v
            new_candidates = candidates_mask & comp_mask[v]
            new_clique = clique + [v]
            expand(new_candidates, new_clique)
            
            # Exclude v: remove v from candidates for subsequent iterations
            candidates_mask &= ~(1 << v)

    # Start with all vertices
    initial_clique = []
    expand(all_mask, initial_clique)
    
    # Convert vertex indices to values (1-based)
    result = [v + 1 for v in best_clique]
    
    if not result:
        result = [1]
        
    return result
