<<<<<<< SEARCH
def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Returns (center_x, center_y, side_length, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Collect all vertices of all inner hexagons
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    # Try different angles for the outer hexagon
    # Due to symmetry, 0 and 30 are the most likely optimal orientations for a hexagonal cluster.
    for angle_deg in [0.0, 30.0]:
        # The normals of the outer hexagon are at angle_deg + 30 + 60*k
        # We need to find a center (cx, cy) that minimizes the max projection.
        
        # Start with centroid of vertices
        cx_mean = sum(v[0] for v in all_vertices) / len(all_vertices)
        cy_mean = sum(v[1] for v in all_vertices) / len(all_vertices)
        
        # Refine center using a fine grid search around the centroid
        # The optimal center is within the convex hull of the points.
        # A grid step of 0.01 should be sufficient for high precision.
        range_search = 2.0
        step = 0.05
        
        best_local_side = float('inf')
        best_local_center = (cx_mean, cy_mean)
        
        # Coarse to fine grid search
        for r in [range_search, range_search/2, range_search/4, range_search/8]:
            for cx in [cx_mean + dx * r for dx in [-1, -0.5, 0, 0.5, 1]]:
                for cy in [cy_mean + dy * r for dy in [-1, -0.5, 0, 0.5, 1]]:
                    required_inradius = 0.0
                    for k in range(6):
                        a = math.radians(angle_deg + 30 + 60 * k)
                        nx = math.cos(a)
                        ny = math.sin(a)
                        
                        max_proj = -float('inf')
                        min_proj = float('inf')
                        for vx, vy in all_vertices:
                            proj = (vx - cx) * nx + (vy - cy) * ny
                            if proj > max_proj:
                                max_proj = proj
                            if proj < min_proj:
                                min_proj = proj
                        
                        extent = max(max_proj, -min_proj)
                        if extent > required_inradius:
                            required_inradius = extent
                    
                    side = required_inradius / (math.sqrt(3) / 2)
                    if side < best_local_side:
                        best_local_side = side
                        best_local_center = (cx, cy)
            
            cx_mean, cy_mean = best_local_center
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_center = best_local_center
            best_angle = angle_deg
            
    return best_center[0], best_center[1], best_side, best_angle
=======
def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Returns (center_x, center_y, side_length, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Collect all vertices of all inner hexagons
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    # Try different angles for the outer hexagon with finer resolution
    # Due to symmetry, 0 and 30 are the most likely optimal orientations, but try nearby angles too
    for angle_deg in [0.0, 15.0, 30.0, 45.0, 60.0]:
        # The normals of the outer hexagon are at angle_deg + 30 + 60*k
        # We need to find a center (cx, cy) that minimizes the max projection.
        
        # Start with centroid of vertices
        cx_mean = sum(v[0] for v in all_vertices) / len(all_vertices)
        cy_mean = sum(v[1] for v in all_vertices) / len(all_vertices)
        
        best_local_side = float('inf')
        best_local_center = (cx_mean, cy_mean)
        
        # Multi-resolution grid search with finer final step
        for r in [3.0, 1.5, 0.75, 0.375, 0.1875, 0.09375]:
            n_grid = 5  # 5x5 grid at each level
            step = r / (n_grid // 2)
            best_at_level = float('inf')
            best_cx, best_cy = cx_mean, cy_mean
            
            for i in range(n_grid):
                for j in range(n_grid):
                    cx = cx_mean + (i - n_grid // 2) * step
                    cy = cy_mean + (j - n_grid // 2) * step
                    
                    required_inradius = 0.0
                    for k in range(6):
                        a = math.radians(angle_deg + 30 + 60 * k)
                        nx = math.cos(a)
                        ny = math.sin(a)
                        
                        max_proj = -float('inf')
                        min_proj = float('inf')
                        for vx, vy in all_vertices:
                            proj = (vx - cx) * nx + (vy - cy) * ny
                            if proj > max_proj:
                                max_proj = proj
                            if proj < min_proj:
                                min_proj = proj
                        
                        extent = max(max_proj, -min_proj)
                        if extent > required_inradius:
                            required_inradius = extent
                    
                    side = required_inradius / (math.sqrt(3) / 2)
                    if side < best_at_level:
                        best_at_level = side
                        best_cx, best_cy = cx, cy
            
            cx_mean, cy_mean = best_cx, best_cy
            if best_at_level < best_local_side:
                best_local_side = best_at_level
                best_local_center = (best_cx, best_cy)
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_center = best_local_center
            best_angle = angle_deg
            
    return best_center[0], best_center[1], best_side, best_angle
>>>>>>> REPLACE

