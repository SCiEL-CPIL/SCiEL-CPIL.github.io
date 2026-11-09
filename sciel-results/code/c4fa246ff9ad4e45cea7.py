import itertools
import time

def _is_2_colorable(n, edges):
    """
    Checks if a 5-uniform hypergraph on n vertices is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    # Preprocess edges to sets for faster lookup? 
    # Actually, for propagation, we need to check colors of vertices in an edge.
    # We can just iterate through edges.
    
    # colors: -1 unknown, 0, 1
    colors = [-1] * n
    
    # For faster propagation, we can maintain counts of 0s and 1s in each edge?
    # With small n and small edge count, simple check is fine.
    
    def propagate():
        """
        Applies unit propagation. Returns False if contradiction.
        """
        changed = True
        while changed:
            changed = False
            for edge in edges:
                # Count colors in this edge
                c0 = 0
                c1 = 0
                unknown = -1
                for v in edge:
                    if colors[v] == 0:
                        c0 += 1
                    elif colors[v] == 1:
                        c1 += 1
                    else:
                        unknown = v
                        break # If more than 1 unknown, no unit clause
                # If we broke early, there might be >1 unknown. 
                # Let's count properly.
                c0 = 0
                c1 = 0
                unknowns = []
                for v in edge:
                    if colors[v] == 0:
                        c0 += 1
                    elif colors[v] == 1:
                        c1 += 1
                    else:
                        unknowns.append(v)
                
                # Contradiction: all 5 are same color
                if c0 == 5 or c1 == 5:
                    return False
                
                # Unit clause: 4 of one color, 1 unknown
                if len(unknowns) == 1:
                    if c0 == 4:
                        if colors[unknowns[0]] == 1:
                            return False
                        colors[unknowns[0]] = 1
                        changed = True
                    elif c1 == 4:
                        if colors[unknowns[0]] == 0:
                            return False
                        colors[unknowns[0]] = 0
                        changed = True
        return True

    def backtrack(idx):
        """
        Recursively assign colors to vertices.
        """
        # Find next uncolored vertex
        # Heuristic: pick vertex with most constraints? 
        # For simplicity, pick the first uncolored one.
        # Better: pick the one involved in the most "tight" edges.
        # Let's just pick the first uncolored vertex for now.
        v = -1
        for i in range(n):
            if colors[i] == -1:
                v = i
                break
        
        if v == -1:
            # All colored, check if valid (should be, due to propagation)
            # But we must ensure no edge is monochromatic.
            # Propagation handles monochromatic detection during assignment.
            # If we reach here, it's a valid coloring.
            return True
        
        # Try color 0
        colors[v] = 0
        if propagate():
            if backtrack(idx + 1):
                return True
        
        # Try color 1
        colors[v] = 1
        if propagate():
            if backtrack(idx + 1):
                return True
        
        # Backtrack
        colors[v] = -1
        return False

    # Initial propagation
    if not propagate():
        return False
    
    if backtrack(0):
        return True
    else:
        return False

def _minimize_hypergraph(n):
    """
    Finds a minimal non-2-colorable 5-uniform hypergraph on n vertices
    by starting with the complete hypergraph and greedily removing edges.
    """
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    current_edges = all_edges[:]
    
    # Verify it's non-2-colorable (it should be for n>=9)
    if _is_2_colorable(n, current_edges):
        return None # Should not happen for n>=9
    
    # Greedy removal
    i = 0
    while i < len(current_edges):
        edge = current_edges[i]
        # Try removing edge
        trial_edges = current_edges[:i] + current_edges[i+1:]
        
        # Check if trial_edges is 2-colorable
        if _is_2_colorable(n, trial_edges):
            # It became 2-colorable, so we must keep this edge
            i += 1
        else:
            # It is still non-2-colorable, so we can remove this edge
            del current_edges[i]
            # Do not increment i, as the next edge has shifted into position i
    
    return current_edges

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_n = 9
    best_edges = None
    best_count = float('inf')
    
    # We know n=9 works with 126 edges.
    # Let's try to minimize for n=9, 10, 11, 12.
    
    for n in range(9, 15):
        if time.time() > deadline:
            break
            
        # Generate minimized hypergraph
        edges = _minimize_hypergraph(n)
        
        if edges is not None:
            if len(edges) < best_count:
                best_count = len(edges)
                best_n = n
                best_edges = edges
                
        # If we found a very small one, we can stop early?
        # Theoretical min is >28. If we find 29, we stop.
        if best_count <= 29:
            break

    # Fallback if something went wrong or time ran out before finding anything
    if best_edges is None:
        # Return the seed's result: complete 5-uniform on 9 vertices
        return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
