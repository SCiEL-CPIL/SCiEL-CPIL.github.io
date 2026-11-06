<<<<<<< SEARCH
import time
import random
from itertools import combinations

# Precompute all possible 5-combinations for n up to 20
# This is used for fast random edge generation
MAX_N = 20
ALL_COMBS = []
for n in range(5, MAX_N + 1):
    combs = list(combinations(range(n), 5))
    ALL_COMBS.append(combs)

def check_non_2_colorable_fast(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Uses bitmask representation for speed.
    edges: list of tuples of 5 vertices.
    """
    if not edges:
        return False
    
    # Map vertices to indices 0..n-1
    # We represent the coloring as a bitmask of n bits.
    # For each edge, we need to check if all 5 bits are the same.
    
    # Optimization: Precompute the bitmasks for each edge?
    # An edge (v0, v1, v2, v3, v4) is monochromatic in mask M if:
    # (M & mask_edge) == 0 or (M & mask_edge) == mask_edge
    # where mask_edge has bits set for v0..v4.
    
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # If n is small (<= 15), brute force is extremely fast
    if n <= 15:
        for mask in range(1 << n):
            valid = True
            for em in edge_masks:
                # Check if edge is monochromatic
                # If it is, this coloring is invalid for "non-2-colorable" check
                # We are looking for a coloring where NO edge is monochromatic.
                # So if we find one where ALL edges are non-monochromatic, it IS 2-colorable.
                # Here we check if this mask makes the edge monochromatic.
                # If any edge is monochromatic, this mask is NOT a valid 2-coloring.
                # Wait, the definition of 2-colorable is: exists a coloring such that NO edge is monochromatic.
                # So we iterate masks. If we find a mask where for ALL edges, the edge is NOT monochromatic, 
                # then it is 2-colorable.
                # If we exhaust all masks and didn't find such a mask, it is NOT 2-colorable.
                
                # Let's rephrase:
                # Is there a mask M such that for all e in edges, M does not make e monochromatic?
                
                # Monochromatic check for edge e with mask M:
                # Let S be the set of vertices in e.
                # M restricts to S. If all bits in S are 0 or all are 1, it's mono.
                # val = M & em
                # if val == 0 or val == em: mono.
                
                if (mask & em) == 0 or (mask & em) == em:
                    valid = False
                    break
            if valid:
                # Found a valid 2-coloring
                return False 
        return True # No valid 2-coloring found

    else:
        # Backtracking with unit propagation for n > 15
        # This is a simplified version for robustness, but local search usually keeps n small (8-12)
        # For n > 15, we might need a more robust solver, but let's assume local search stays in small n.
        # If n gets large, we might fail. Let's force n to be <= 15 in the search if possible.
        # However, the problem allows up to 26. 
        # For now, if n > 15, we use a slower backtracking.
        
        # Simple Backtracking
        vertex_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                vertex_edges[v].append(i)
        
        colors = [-1] * n
        edge_counts = [[0, 0] for _ in edges]
        
        def backtrack(idx):
            # Find forced move or pick vertex
            target_v = -1
            target_c = -1
            forced = False
            
            for i in range(len(edges)):
                c0, c1 = edge_counts[i]
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and c1 == 0:
                    for v in edges[i]:
                        if colors[v] == -1:
                            target_v = v
                            target_c = 1
                            forced = True
                            break
                elif c1 == 4 and c0 == 0:
                    for v in edges[i]:
                        if colors[v] == -1:
                            target_v = v
                            target_c = 0
                            forced = True
                            break
                if forced:
                    break
            
            if not forced:
                # Pick first unassigned
                for v in range(n):
                    if colors[v] == -1:
                        target_v = v
                        break
                if target_v == -1:
                    return True
            
            # Apply
            colors[target_v] = target_c
            saved = []
            for i in vertex_edges[target_v]:
                saved.append((edge_counts[i][0], edge_counts[i][1]))
                if target_c == 0:
                    edge_counts[i][0] += 1
                else:
                    edge_counts[i][1] += 1
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            colors[target_v] = -1
            for i, (c0, c1) in zip(vertex_edges[target_v], saved):
                edge_counts[i] = [c0, c1]
            
            if forced:
                return False
            
            # Try other color
            other_c = 1 - target_c
            colors[target_v] = other_c
            saved = []
            for i in vertex_edges[target_v]:
                saved.append((edge_counts[i][0], edge_counts[i][1]))
                if other_c == 0:
                    edge_counts[i][0] += 1
                else:
                    edge_counts[i][1] += 1
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            colors[target_v] = -1
            for i, (c0, c1) in zip(vertex_edges[target_v], saved):
                edge_counts[i] = [c0, c1]
            
            return False
        
        return not backtrack(0)
=======
import time
import random
from itertools import combinations

def is_not_2_colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return False
    
    # Build vertex-to-edge index
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    colors = [-1] * n
    edge_counts = [[0, 0] for _ in range(len(edges))]
    
    def propagate():
        """Apply unit propagation. Returns False if conflict, True otherwise."""
        changed = True
        while changed:
            changed = False
            for i in range(len(edges)):
                c0, c1 = edge_counts[i]
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and c1 == 0:
                    # Force all uncolored vertices in this edge to color 1
                    for v in edges[i]:
                        if colors[v] == -1:
                            colors[v] = 1
                            for j in vertex_edges[v]:
                                edge_counts[j][1] += 1
                            changed = True
                elif c1 == 4 and c0 == 0:
                    # Force all uncolored vertices in this edge to color 0
                    for v in edges[i]:
                        if colors[v] == -1:
                            colors[v] = 0
                            for j in vertex_edges[v]:
                                edge_counts[j][0] += 1
                            changed = True
        return True
    
    def backtrack():
        # Check for conflicts
        for i in range(len(edges)):
            if edge_counts[i][0] == 5 or edge_counts[i][1] == 5:
                return False
        
        # Find uncolored vertex
        target_v = -1
        for v in range(n):
            if colors[v] == -1:
                target_v = v
                break
        
        if target_v == -1:
            return True  # All colored, no conflict
        
        # Try color 0
        saved = []
        for i in vertex_edges[target_v]:
            saved.append((edge_counts[i][0], edge_counts[i][1]))
            edge_counts[i][0] += 1
        colors[target_v] = 0
        
        if propagate() and backtrack():
            return True
        
        # Undo
        colors[target_v] = -1
        for i, (c0, c1) in zip(vertex_edges[target_v], saved):
            edge_counts[i] = [c0, c1]
        
        # Try color 1
        saved = []
        for i in vertex_edges[target_v]:
            saved.append((edge_counts[i][0], edge_counts[i][1]))
            edge_counts[i][1] += 1
        colors[target_v] = 1
        
        if propagate() and backtrack():
            return True
        
        # Undo
        colors[target_v] = -1
        for i, (c0, c1) in zip(vertex_edges[target_v], saved):
            edge_counts[i] = [c0, c1]
        
        return False
    
    return not (propagate() and backtrack())
>>>>>>> REPLACE

