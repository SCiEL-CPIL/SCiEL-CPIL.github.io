"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def _is_2_colorable(edges):
    """
    Check if a 3-uniform hypergraph is 2-colorable.
    Uses backtracking with constraint propagation for efficiency.
    Returns True if 2-colorable, False otherwise.
    """
    n_vertices = 9
    
    # Map vertex -> list of edges it belongs to
    vertex_edges = [[] for _ in range(n_vertices)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # Order vertices by degree (descending) to prune earlier
    # This is a heuristic for better performance, though for 9 vertices it's trivial
    order = sorted(range(n_vertices), key=lambda x: len(vertex_edges[x]), reverse=True)
    
    # Precompute edge index mapping for quick lookup if needed, 
    # but we iterate edges directly in checks.
    
    def backtrack(idx, coloring):
        if idx == n_vertices:
            return True
        
        v = order[idx]
        
        # Try color 0
        # Check if assigning color 0 to v creates a monochromatic edge
        valid_0 = True
        for e_idx in vertex_edges[v]:
            e = edges[e_idx]
            # Check other two vertices in the edge
            other_vertices = [u for u in e if u != v]
            # If both other vertices are assigned color 0, then this edge would be mono-0
            # If either is unassigned, it's not mono yet.
            # If one is assigned 1, it's not mono.
            c1 = coloring[other_vertices[0]]
            c2 = coloring[other_vertices[1]]
            if c1 == 0 and c2 == 0:
                valid_0 = False
                break
            # Note: if c1 or c2 is -1 (unassigned), we can't decide yet, but we must ensure
            # that they *can* be non-0. Since we are backtracking, we just check immediate conflicts.
            # Actually, if c1=0 and c2=-1, it's currently valid, but c2 must eventually be 1.
            # The check "c1==0 and c2==0" is the only hard conflict at this step.
            
        if valid_0:
            coloring[v] = 0
            if backtrack(idx + 1, coloring):
                return True
            coloring[v] = -1
            
        # Try color 1
        valid_1 = True
        for e_idx in vertex_edges[v]:
            e = edges[e_idx]
            other_vertices = [u for u in e if u != v]
            c1 = coloring[other_vertices[0]]
            c2 = coloring[other_vertices[1]]
            if c1 == 1 and c2 == 1:
                valid_1 = False
                break
        
        if valid_1:
            coloring[v] = 1
            if backtrack(idx + 1, coloring):
                return True
            coloring[v] = -1
            
        return False

    coloring = [-1] * n_vertices
    return backtrack(0, coloring)

def construct():
    """
    Returns a 3-chromatic 3-uniform hypergraph on 9 vertices (0..8)
    with the minimum number of edges.
    
    The Fano plane is the unique minimal 3-chromatic 3-uniform hypergraph.
    It has 7 edges on 7 vertices.
    We embed it on vertices 0..6 and leave 7, 8 isolated.
    """
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Verified 3-coloring for the Fano plane
    # c[0]=0, c[1]=1, c[2]=2
    # c[3]=1, c[4]=2, c[5]=0
    # c[6]=1
    # c[7]=0, c[8]=0 (isolated)
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Sanity Check 1: Ensure provided coloring is valid (no monochromatic edges)
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            raise ValueError("Provided coloring is invalid: monochromatic edge found")
            
    # Sanity Check 2: Ensure it is NOT 2-colorable
    # Using the optimized backtracker
    if _is_2_colorable(edges):
        raise ValueError("Graph is 2-colorable, violates 3-chromatic requirement")

    # Ensure types are strictly Python int and list/tuple
    # The task says sequences must be list or tuple. Tuples are fine.
    # Integers must be Python int.
    final_edges = [tuple(int(v) for v in e) for e in edges]
    final_colors = [int(c) for c in colors]
    
    return {'edges': final_edges, 'colors': final_colors}

# EVOLVE-BLOCK-END
