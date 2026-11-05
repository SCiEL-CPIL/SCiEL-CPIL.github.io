# Replace _smooth_objective and related helpers with a faster vectorized version

def _fast_smooth_objective(v, t0):
    """
    Fast smooth objective for L-BFGS.
    v: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, cx, cy, out_angle]
    """
    # Unpack
    cfg = np.zeros((N, 3))
    for i in range(N):
        cfg[i, 0] = v[3*i]
        cfg[i, 1] = v[3*i+1]
        cfg[i, 2] = v[3*i+2]
    
    cx = v[3*N]
    cy = v[3*N+1]
    out_angle_deg = v[3*N+2]
    
    # 1. Calculate outer side requirement (smooth max of projections)
    # Outer hexagon normals
    a_out = math.radians(out_angle_deg)
    # Normals of outer hexagon edges (pointing inward? No, outward from center to define boundary)
    # The condition is: for all vertices p of inner hexagons, proj(p - center) <= S * apothem
    # So S >= max(proj(p - center) / apothem)
    # We want to minimize S.
    
    # Compute all vertices of all inner hexagons
    # N hexagons, 6 vertices each -> 72 vertices
    # We can vectorize the rotation and translation
    
    # Precompute unit hexagon vertices (local frame)
    # _UNIT_HEX_VERTS is (6, 2)
    
    # Rotation matrices for each inner hexagon
    angles_rad = np.deg2rad(cfg[:, 2])
    cos_a = np.cos(angles_rad)
    sin_a = np.sin(angles_rad)
    
    # R = [[cos, -sin], [sin, cos]]
    # Vertices_local = _UNIT_HEX_VERTS (6, 2)
    # Vertices_global = (Vertices_local @ R.T + centers)
    
    # Expand centers to (6, 2) by repeating
    centers = cfg[:, :2] # (N, 2)
    centers_rep = np.repeat(centers, 6, axis=0) # (72, 2)
    
    # Expand rotation params
    cos_rep = np.repeat(cos_a, 6) # (72,)
    sin_rep = np.repeat(sin_a, 6) # (72,)
    
    # Local vertices (6, 2) -> (72, 2) by tiling
    local_verts = np.tile(_UNIT_HEX_VERTS, (N, 1)) # (72, 2)
    
    # Rotate
    # x' = x cos - y sin
    # y' = x sin + y cos
    lx = local_verts[:, 0]
    ly = local_verts[:, 1]
    rx = lx * cos_rep - ly * sin_rep
    ry = lx * sin_rep + ly * cos_rep
    
    # Translate
    gx = rx + centers_rep[:, 0]
    gy = ry + centers_rep[:, 1]
    
    # Relative to outer center
    gx_rel = gx - cx
    gy_rel = gy - cy
    
    # Project onto outer hexagon normals
    # Outer hexagon normals are at angles 90 + 60k + out_angle
    # Normal k: (cos(theta_k), sin(theta_k))
    # theta_k = a_out + pi/2 + k*pi/3
    thetas = a_out + np.pi/2 + np.arange(6) * np.pi/3
    normals = np.column_stack((np.cos(thetas), np.sin(thetas))) # (6, 2)
    
    # Projections: (72, 1) @ (1, 6) -> (72, 6)
    # proj[i, k] = gx_rel[i] * normals[k, 0] + gy_rel[i] * normals[k, 1]
    proj = gx_rel[:, np.newaxis] * normals[0][np.newaxis, :] + gy_rel[:, np.newaxis] * normals[1][np.newaxis, :]
    # Wait, normals is (6,2). 
    # proj[i, k] = sum_j rel[i, j] * normals[k, j]
    # Using matmul: rel (72, 2) @ normals.T (2, 6) -> (72, 6)
    proj = np.dot(np.column_stack((gx_rel, gy_rel)), normals.T)
    
    # We need max over i for each k
    # max_proj (6,)
    max_proj = np.max(proj, axis=0)
    
    # Smooth max to make it differentiable? 
    # The max function is non-differentiable. 
    # However, L-BFGS can often handle kinks if we use a smooth approximation or if the gradient is defined subgradient-wise (scipy handles this somewhat).
    # But for better performance, let's use log-sum-exp.
    beta = 10.0 # Sharpness
    # smooth_max = (1/beta) * log(sum(exp(beta * proj)))
    # To avoid overflow, subtract max
    m = np.max(proj, axis=0)
    smooth_max_proj = (1.0 / beta) * np.log(np.sum(np.exp(beta * (proj - m)), axis=0)) + m
    
    # Required side S = max_k (smooth_max_proj[k] / apothem)
    apothem = math.sqrt(3) / 2.0
    S_req = np.max(smooth_max_proj) / apothem
    
    # 2. Overlap Penalty
    # Calculate pairwise distances between centers? 
    # A simple penalty: if centers are closer than 2*sqrt(3) (touching distance for aligned hexes), penalize.
    # But they can rotate. 
    # A better penalty: use the minimum distance between edges. 
    # For speed, let's use a soft penalty based on center distance.
    # d_min_center = 2 * sqrt(3) approx 3.46
    # This is a heuristic. It might allow overlaps if rotated poorly.
    # Let's try a stronger penalty based on a quick SDF check for only a few pairs? 
    # No, let's stick to center distance for speed in L-BFGS, and rely on SA to fix the rest.
    # Or, use the existing _smooth_collision_penalty but make it faster?
    # The existing one calls _min_distance_hexagons which is O(N^2 * V^2). Very slow.
    
    # Let's use a cheaper overlap penalty:
    # Sum of max(0, (d_min_center - dist_center)^2)
    # This is smooth.
    centers_arr = cfg[:, :2]
    # Compute pairwise distances
    # dist[i, j] = ||c_i - c_j||
    # We can use scipy.spatial.distance or just numpy
    # For N=12, 66 pairs.
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = centers_arr[i, 0] - centers_arr[j, 0]
            dy = centers_arr[i, 1] - centers_arr[j, 1]
            d = math.hypot(dx, dy)
            # Threshold: minimum distance for unit hexes is sqrt(3) (side touching) ? 
            # No, if two unit hexes touch at a vertex, distance is 2.
            # If they touch at a side, distance is sqrt(3).
            # Minimum possible distance between centers of two non-overlapping unit hexes is sqrt(3) (if aligned and side-touching).
            # If rotated, it can be smaller? No, the inradius is sqrt(3)/2. 
            # The distance between centers must be at least 2 * inradius = sqrt(3) if they are parallel.
            # If rotated, the "width" in the direction of the center line might be less?
            # Actually, the minimum distance between centers of two non-overlapping regular hexagons of side 1 is sqrt(3).
            # Proof: The distance from center to any point in hex is <= 1 (circumradius).
            # So d >= 1+1=2? No.
            # The inradius is sqrt(3)/2 ~ 0.866.
            # If two hexes touch at a side, the distance between centers is 2 * inradius = sqrt(3) ~ 1.732.
            # Can it be smaller? If they touch at a vertex?
            # Distance between centers when touching at a vertex: 
            # Consider two hexes, one at (0,0), one at (d, 0).
            # Vertex of first at (1, 0). Vertex of second at (d-1, 0).
            # If they touch, d-1 = 1 => d=2.
            # Side touching: First hex right side is x=sqrt(3)/2. Second hex left side is x=d-sqrt(3)/2.
            # Touching: d - sqrt(3)/2 = sqrt(3)/2 => d = sqrt(3).
            # So sqrt(3) is the minimum distance for side-touching.
            # Is it possible to have d < sqrt(3) without overlap?
            # The "width" of a hexagon in any direction is between sqrt(3) and 2.
            # If we align the narrowest direction (width sqrt(3)) with the center line, we get d=sqrt(3).
            # So d_min = sqrt(3).
            
            d_min = math.sqrt(3)
            if d < d_min:
                penalty += (d_min - d)**2
    
    # Weight the penalty heavily
    total = S_req + 10.0 * penalty
    
    return total
