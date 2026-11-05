<<<<<<< SEARCH
def _compute_S_for_rotation(verts, oangle):
    """
    Compute minimal side length S of outer hexagon at origin with rotation oangle
    containing all vertices in 'verts' (N x 6 x 2).
    """
    # Transform to container local frame (rotate by -oangle)
    cos_a = np.cos(-oangle)
    sin_a = np.sin(-oangle)
    
    rx = verts[..., 0] * cos_a - verts[..., 1] * sin_a
    ry = verts[..., 0] * sin_a + verts[..., 1] * cos_a
    
    # Normals for unit hexagon facets (apothem directions)
    theta_n = np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(theta_n)
    ny = np.sin(theta_n)
    
    # Projections
    proj = rx[..., np.newaxis] * nx[np.newaxis, np.newaxis, :] + \
           ry[..., np.newaxis] * ny[np.newaxis, np.newaxis, :]
    
    max_proj = np.max(np.abs(proj))
    apothem = np.sqrt(3) / 2
    return max_proj / apothem

def _find_best_outer(verts):
    """
    Given all inner hexagon vertices (N x 6 x 2), find the minimal enclosing regular hexagon.
    Returns (best_S, best_oangle).
    """
    def S_of_angle(oangle):
        return _compute_S_for_rotation(verts, oangle)
    
    # Coarse search
    best_S = float('inf')
    best_angle = 0.0
    n_coarse = 36
    for i in range(n_coarse):
        ang = i * (np.pi / 3) / n_coarse
        s_val = S_of_angle(ang)
        if s_val < best_S:
            best_S = s_val
            best_angle = ang
    
    # Fine search around best
    result = minimize_scalar(S_of_angle, bounds=(best_angle - 0.1, best_angle + 0.1), 
                             method='bounded', options={'xatol': 1e-10})
    if result.success:
        best_S = result.fun
        best_angle = result.x
    
    # Ensure angle is in [0, pi/3)
    best_angle = best_angle % (np.pi / 3)
    
    return best_S, best_angle
=======
def _compute_S_for_rotation(verts, oangle):
    """
    Compute minimal side length S of outer hexagon at origin with rotation oangle
    containing all vertices in 'verts' (N x 6 x 2).
    Uses the support function: S = max over 6 facet normals of (max_proj - min_proj) / (2 * apothem)
    """
    # Transform to container local frame (rotate by -oangle)
    cos_a = np.cos(-oangle)
    sin_a = np.sin(-oangle)
    
    rx = verts[..., 0] * cos_a - verts[..., 1] * sin_a
    ry = verts[..., 0] * sin_a + verts[..., 1] * cos_a
    
    # Normals for unit hexagon facets (apothem directions)
    theta_n = np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(theta_n)
    ny = np.sin(theta_n)
    
    # Projections
    proj = rx[..., np.newaxis] * nx[np.newaxis, np.newaxis, :] + \
           ry[..., np.newaxis] * ny[np.newaxis, np.newaxis, :]
    
    # Width in each normal direction
    widths = np.max(proj, axis=(0, 1)) - np.min(proj, axis=(0, 1))
    apothem = np.sqrt(3) / 2
    return np.max(widths) / (2.0 * apothem)

def _find_best_outer_discrete(verts, n_bins=720):
    """
    Discretize the outer hexagon rotation into n_bins within [0, pi/3) and
    analytically compute S for each bin using the support function.
    Returns (best_S, best_oangle).
    """
    angles = np.linspace(0, np.pi / 3, n_bins, endpoint=False)
    
    # Vectorized computation: rotate all vertices once per angle
    # For efficiency, process in chunks
    best_S = float('inf')
    best_angle = 0.0
    
    chunk_size = 100
    for start in range(0, n_bins, chunk_size):
        end = min(start + chunk_size, n_bins)
        chunk_angles = angles[start:end]
        
        cos_vals = np.cos(-chunk_angles)  # (chunk,)
        sin_vals = np.sin(-chunk_angles)  # (chunk,)
        
        # verts shape: (N, 6, 2)
        # For each angle, compute rotated coords and projections
        for ci in range(end - start):
            ca = cos_vals[ci]
            sa = sin_vals[ci]
            rx = verts[..., 0] * ca - verts[..., 1] * sa
            ry = verts[..., 0] * sa + verts[..., 1] * ca
            
            theta_n = np.pi / 6 + np.arange(6) * np.pi / 3
            nx = np.cos(theta_n)
            ny = np.sin(theta_n)
            
            proj = rx[..., np.newaxis] * nx[np.newaxis, np.newaxis, :] + \
                   ry[..., np.newaxis] * ny[np.newaxis, np.newaxis, :]
            
            widths = np.max(proj, axis=(0, 1)) - np.min(proj, axis=(0, 1))
            apothem = np.sqrt(3) / 2
            s_val = np.max(widths) / (2.0 * apothem)
            
            if s_val < best_S:
                best_S = s_val
                best_angle = chunk_angles[ci]
    
    # Fine-tune around the best discrete angle
    if n_bins >= 36:
        step = (np.pi / 3) / n_bins
        result = minimize_scalar(
            lambda a: _compute_S_for_rotation(verts, a),
            bounds=(best_angle - step, best_angle + step),
            method='bounded', options={'xatol': 1e-12}
        )
        if result.success and result.fun < best_S:
            best_S = result.fun
            best_angle = result.x
    
    best_angle = best_angle % (np.pi / 3)
    return best_S, best_angle
>>>>>>> REPLACE

