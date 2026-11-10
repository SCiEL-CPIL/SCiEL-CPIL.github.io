<<<<<<< SEARCH
def _min_enclosing_hex_side(centers):
    """
    Find the minimum side length s of a regular hexagon (centered at origin, angle 0)
    that contains all points at distance up to 1 from each center.
    """
    normals = _hex_edge_normals(0.0)
    unit_verts = _hex_vertices(0, 0, 1.0, 0.0)
    
    max_required_s = 0.0
    for center in centers:
        for n in normals:
            support = np.max(unit_verts @ n)
            required_s = (center @ n + support) / (np.sqrt(3) / 2)
            if required_s > max_required_s:
                max_required_s = required_s
    
    return max_required_s
=======
def _min_enclosing_hex_side(centers):
    """
    Find the minimum side length s of a regular hexagon (centered at origin, angle 0)
    that contains all points at distance up to 1 from each center.
    """
    normals = _hex_edge_normals(0.0)
    unit_verts = _hex_vertices(0, 0, 1.0, 0.0)
    
    max_required_s = 0.0
    for center in centers:
        for n in normals:
            support = np.max(unit_verts @ n)
            required_s = (center @ n + support) / (np.sqrt(3) / 2)
            if required_s > max_required_s:
                max_required_s = required_s
    
    return max_required_s

def _objective_fast(params, outer_verts_cache=None):
    """Fast vectorized objective function."""
    inner_xy = params[:24].reshape(12, 2)
    s_val = params[24]
    
    if s_val < np.sqrt(12):
        return 1e6
    
    # Compute all inner hexagon vertices
    all_verts = np.zeros((12, 6, 2))
    for i in range(12):
        all_verts[i] = _hex_vertices(inner_xy[i, 0], inner_xy[i, 1], 1.0, 0.0)
    
    penalty = 0.0
    
    # Check each inner hexagon is inside outer
    outer_verts = _hex_vertices(0, 0, s_val, 0)
    for i in range(12):
        verts = all_verts[i]
        for v in verts:
            for j in range(6):
                p1 = outer_verts[j]
                p2 = outer_verts[(j+1) % 6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < 0:
                    penalty += abs(cross)
    
    # Check separation between all pairs
    for i in range(12):
        for j in range(i+1, 12):
            gap = _min_gap_sat(all_verts[i], all_verts[j])
            if gap < 0:
                penalty += gap ** 2
    
    return s_val + 1000 * penalty
>>>>>>> REPLACE

