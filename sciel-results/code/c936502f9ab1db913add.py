<<<<<<< SEARCH
    if best_result is None:
        # Fallback to initial guess
        best_result = x0

    # Extract final parameters
    ox, oy, o_ang, R = best_result[0], best_result[1], best_result[2], best_result[3]
    
    inner = []
    for i in range(N):
        ix = best_result[4 + 3*i]
        iy = best_result[5 + 3*i]
        i_ang = best_result[6 + 3*i]
        inner.append([ix, iy, i_ang])
        
    return inner, (ox, oy), R, o_ang
=======
    if best_result is None:
        # Fallback to initial guess
        best_result = x0

    # Extract final parameters
    ox, oy, o_ang, R_opt = best_result[0], best_result[1], best_result[2], best_result[3]
    
    # Calculate the minimum required side length R to contain all inner hexagons
    # This ensures the solution is valid regardless of optimizer convergence issues
    outer_center = np.array([ox, oy])
    normal_angles = np.radians(o_ang) + np.pi/3 * np.arange(6)
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    max_apothem_required = 0.0
    
    for i in range(N):
        ix = best_result[4 + 3*i]
        iy = best_result[5 + 3*i]
        i_ang = best_result[6 + 3*i]
        
        inner_center = np.array([ix, iy])
        verts = hex_vertices(inner_center, i_ang, 1.0)
        verts_rel = verts - outer_center
        
        # For each normal direction, find the max projection
        # The apothem must be at least this max projection
        for k in range(6):
            n = normals[k]
            proj = np.max(verts_rel @ n)
            if proj > max_apothem_required:
                max_apothem_required = proj
    
    # Required R = 2 * apothem / sqrt(3)
    R_required = 2.0 * max_apothem_required / np.sqrt(3.0)
    
    # Use the larger of the optimized R and the required R to ensure validity
    R_final = max(R_opt, R_required)
    
    # Small buffer to avoid floating point issues in evaluator
    R_final += 1e-9

    inner = []
    for i in range(N):
        ix = best_result[4 + 3*i]
        iy = best_result[5 + 3*i]
        i_ang = best_result[6 + 3*i]
        inner.append([ix, iy, i_ang])
        
    return inner, (ox, oy), R_final, o_ang
>>>>>>> REPLACE
