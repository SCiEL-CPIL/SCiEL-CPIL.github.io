<<<<<<< SEARCH
def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    return 2.0 * max_projection / SQRT3
=======
def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Precompute normals for the outer hexagon (in its local frame)
    normals = np.array([[math.cos(math.radians(30 + i * 60)), math.sin(math.radians(30 + i * 60))] for i in range(6)])
    
    max_projection = 0.0
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    # Precompute vertex offsets for a unit hexagon at angle 0
    # Then rotate them for each inner hexagon
    # Actually, let's just vectorize the whole thing per inner hexagon for speed, or all at once.
    # Let's try to do it in chunks or all at once if memory permits.
    # N=11, 6 verts each, 6 normals. 11*6*6 = 396 projections. Very small.
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        # Vectorize projection for this hexagon
        # verts is list of 6 tuples. Convert to array.
        v_arr = np.array(verts)
        # dx, dy relative to outer center
        dx = v_arr[:, 0] - ox
        dy = v_arr[:, 1] - oy
        
        # Rotate into outer frame
        # lx = dx * cos_a + dy * sin_a
        # ly = -dx * sin_a + dy * cos_a
        lx = dx * cos_a + dy * sin_a
        ly = -dx * sin_a + dy * cos_a
        
        # Project onto normals
        # proj = lx * nx + ly * ny
        # lx, ly shape (6,), normals shape (6, 2)
        # We need max over all 6x6 projections
        # proj_matrix = lx[:, None] * normals[0] + ly[:, None] * normals[1]
        # max_proj = np.max(proj_matrix)
        
        # Efficient: 
        # proj = lx[:, None] * normals[0, :] + ly[:, None] * normals[1, :]
        # This is (6, 6)
        # max = proj.max()
        
        # Even faster:
        # The projection of a point (lx, ly) onto normal (nx, ny) is lx*nx + ly*ny.
        # We want max over all vertices and all normals.
        # This is equivalent to max over normals of (max over vertices of proj).
        # Or max over vertices of (max over normals of proj).
        
        # Let's just compute the (6,6) matrix.
        proj_matrix = lx[:, None] * normals[0, :] + ly[:, None] * normals[1, :]
        local_max = float(np.max(proj_matrix))
        
        if local_max > max_projection:
            max_projection = local_max
    
    return 2.0 * max_projection / SQRT3
>>>>>>> REPLACE
