import time
import random
from itertools import combinations

def is_2_colorable(edges, n):
    """
    Returns True if the hypergraph is 2-colorable.
    """
    if not edges:
        return True
        
    # Map vertex to list of edge indices
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
            
    # Precompute edge vertex sets for fast intersection?
    # We will use a simple recursive backtracking with propagation.
    
    colors = [-1] * n
    
    # Propagation queue
    # If we assign v to c, we check all edges containing v.
    # If an edge has 4 vertices of color 1-c, the 5th MUST be c.
    # If an edge has 5 vertices of color c, contradiction.
    
    def propagate(v, c):
        if colors[v] != -1:
            return colors[v] == c
        if colors[v] == c:
            return True
            
        colors[v] = c
        # Check all edges containing v
        for e_idx in vertex_edges[v]:
            e = edges[e_idx]
            # Count colors in this edge
            count_c = 0
            count_1c = 0
            unassigned = -1
            has_conflict = False
            
            for u in e:
                cu = colors[u]
                if cu == c:
                    count_c += 1
                elif cu == 1 - c:
                    count_1c += 1
                elif cu == -1:
                    unassigned = u
                # Note: if multiple unassigned, we only track one for now? 
                # No, if count_c < 4 and count_1c < 4, no immediate force.
                # But if count_c == 4, the remaining MUST be 1-c.
                # If count_1c == 4, the remaining MUST be c.
                
            # Check for immediate contradiction
            if count_c == 5:
                return False
            if count_1c == 5:
                return False
                
            # Check for forced assignment
            # If 4 are c, the 5th must be 1-c
            if count_c == 4:
                if unassigned != -1:
                    if colors[unassigned] != -1 and colors[unassigned] != 1 - c:
                        return False
                    if colors[unassigned] == -1:
                        if not propagate(unassigned, 1 - c):
                            return False
            # If 4 are 1-c, the 5th must be c
            elif count_1c == 4:
                if unassigned != -1:
                    if colors[unassigned] != -1 and colors[unassigned] != c:
                        return False
                    if colors[unassigned] == -1:
                        if not propagate(unassigned, c):
                            return False
                            
        return True

    def backtrack():
        # Find unassigned vertex with highest degree
        best_v = -1
        best_deg = -1
        for v in range(n):
            if colors[v] == -1:
                deg = len(vertex_edges[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                    
        if best_v == -1:
            return True # All assigned
            
        # Save state
        old_colors = colors[:]
        
        # Try color 0
        if propagate(best_v, 0):
            if backtrack():
                return True
        # Restore
        colors[:] = old_colors
        
        # Try color 1
        if propagate(best_v, 1):
            if backtrack():
                return True
                
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial best solution: K_9^(5)
    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_count = len(best_edges)
    
    # Local Search Strategy:
    # Start with a dense non-2-colorable hypergraph.
    # Try to remove edges one by one.
    # If the hypergraph remains non-2-colorable, keep the removal.
    # This finds locally minimal non-2-colorable hypergraphs.
    
    for trial in range(100):
        if time.time() > deadline - 5.0:
            break
            
        # Choose starting n
        # Smaller n means fewer possible edges, but denser required.
        # n=9 is 126 edges. n=10 is 252 edges.
        # Let's stick to n=9, 10, 11 initially.
        n = random.choice([9, 10, 11])
        
        # Start with all edges
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        # We want to start with a non-2-colorable set.
        # For n=9, the complete hypergraph is non-2-colorable.
        # For n>9, the complete hypergraph is non-2-colorable (Ramsey).
        current_edges = [list(e) for e in all_edges]
        
        # Verify it's non-2-colorable (it should be for these sizes)
        if is_2_colorable(current_edges, n):
            continue # Should not happen for complete graphs of this size
            
        # Greedy Removal
        # We iterate through edges and try to remove them.
        # To be efficient, we can shuffle the edges and try removing each one.
        # If removal keeps it non-2-colorable, we remove it.
        
        # We do this in multiple passes to find deeper minima.
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            random.shuffle(current_edges)
            
            for i in range(len(current_edges)):
                if time.time() > deadline - 5.0:
                    break
                if len(current_edges) <= best_count:
                    break
                    
                e_to_remove = current_edges[i]
                
                # Try removing e_to_remove
                # Create a new list without it
                # Optimization: If we are checking many times, copying the list is O(E).
                # E is ~100-200. This is fine.
                
                new_edges = current_edges[:i] + current_edges[i+1:]
                
                # Check if new_edges is non-2-colorable
                if not is_2_colorable(new_edges, n):
                    # It became colorable, so we cannot remove this edge.
                    pass
                else:
                    # It is still non-2-colorable. Remove it.
                    current_edges = new_edges
                    improved = True
                    # We can break and restart the pass to re-shuffle
                    # Or just continue. Re-shuffling is better to avoid bias.
                    break
            
        # Update best
        if len(current_edges) < best_count:
            best_count = len(current_edges)
            best_n = n
            best_edges = current_edges

    return {'n': best_n, 'edges': best_edges}
