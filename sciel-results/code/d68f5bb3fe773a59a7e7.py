import random
import time
from itertools import combinations

def is_2_colorable(edges, n):
    """
    Check if the hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    # Map vertices to lists of edges they are in
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # colors: -1 unassigned, 0, 1
    colors = [-1] * n
    
    # Precompute edge vertices as sets for fast lookup? 
    # Or just lists.
    
    def propagate(v, c):
        """
        Assign color c to vertex v.
        Returns False if contradiction, True otherwise.
        Modifies colors.
        """
        if colors[v] != -1:
            return colors[v] == c
        
        colors[v] = c
        
        # Check edges containing v
        for e_idx in vertex_edges[v]:
            e = edges[e_idx]
            # Count colors in this edge
            count_c = 0
            count_1_minus_c = 0
            unassigned = -1
            
            for u in e:
                if colors[u] == c:
                    count_c += 1
                elif colors[u] == 1 - c:
                    count_1_minus_c += 1
                else:
                    unassigned = u
            
            # If 4 vertices are 1-c, the last must be c
            if count_1_minus_c == 4:
                if unassigned != -1:
                    if not propagate(unassigned, c):
                        return False
            # If 4 vertices are c, the last must be 1-c
            elif count_c == 4:
                if unassigned != -1:
                    if not propagate(unassigned, 1 - c):
                        return False
            
            # If all 5 are c, contradiction
            if count_c == 5:
                return False
            # If all 5 are 1-c, contradiction
            if count_1_minus_c == 5:
                return False
                
        return True

    def backtrack():
        # Find unassigned vertex with highest degree (most edges)
        best_v = -1
        best_deg = -1
        for v in range(n):
            if colors[v] == -1:
                deg = len(vertex_edges[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
        
        if best_v == -1:
            return True # All assigned, no contradictions found
        
        # Try color 0
        # Save state
        old_colors = colors[:]
        if propagate(best_v, 0):
            if backtrack():
                return True
        # Restore
        colors[:] = old_colors
        
        # Try color 1
        old_colors = colors[:]
        if propagate(best_v, 1):
            if backtrack():
                return True
        
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initialize with a known valid solution: Complete 5-uniform hypergraph on 9 vertices.
    # n=9, edges = all C(9,5) = 126 edges.
    # This is non-2-colorable because any 2-coloring of 9 vertices has a monochromatic set of size >= 5.
    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_count = len(best_edges)
    
    # Strategy: Try different n.
    # For each n, start with empty, add random edges.
    # Stop when non-2-colorable.
    
    for trial in range(200): # Max trials
        if time.time() > deadline - 5.0:
            break
            
        # Choose n
        # We know n=9 complete (126 edges) is valid.
        # We try to find smaller.
        # n=10, 11 might allow sparser non-2-colorable hypergraphs?
        # Actually, higher n usually requires MORE edges to be non-2-colorable?
        # No, the density needs to be high enough.
        # For n=9, density 1.0 (126/126) is required? No, we proved 126 is the total.
        # Can we have a non-2-colorable 5-uniform hypergraph on 9 vertices with < 126 edges?
        # Yes. We just need to hit all "bad" colorings.
        # There are 2^9 = 512 colorings.
        # A coloring is "bad" (i.e., fails to be proper) if it has a mono edge.
        # We want to select edges such that EVERY coloring has a mono edge.
        # This is a hitting set problem.
        
        n = random.randint(9, 12)
        
        # Generate all possible edges
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        current_edges = []
        
        for e in all_edges:
            if time.time() > deadline - 5.0:
                break
            
            # If we already have more edges than best, stop
            if len(current_edges) >= best_count:
                break
                
            current_edges.append(e)
            
            # Check colorability
            if not is_2_colorable(current_edges, n):
                count = len(current_edges)
                if count < best_count:
                    best_count = count
                    best_n = n
                    best_edges = [list(e) for e in current_edges]
                break # This ordering yielded a result, try next
        
    return {'n': best_n, 'edges': best_edges}
