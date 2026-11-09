# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _compute_min_outer_side(inner, outer_center, outer_angle):
    """Compute the minimum side length of outer hexagon that contains all inner hexagons."""
    inner = np.array(inner, dtype=float)
    outer_center = np.array(outer_center, dtype=float)
    
    # Shift inner centers by minus outer center
    shifted_centers = inner[:, :2] - outer_center
    
    # Get all vertices of all inner hexagons
    all_verts = []
    for i in range(12):
        verts = _hex_vertices(shifted_centers[i, 0], shifted_centers[i, 1], 1.0, inner[i, 2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # For a regular hexagon centered at origin with angle a and side s,
    # the constraint is that for each edge, the cross product test must pass.
    a_rad = np.radians(outer_angle)
    unit_outer_verts = np.column_stack([
        np.cos(a_rad + np.radians(60) * np.arange(6)),
        np.sin(a_rad + np.radians(60) * np.arange(6))
    ])
    
    max_proj = 0
    for k in range(6):
        p1 = unit_outer_verts[k]
        p2 = unit_outer_verts[(k+1) % 6]
        edge = p2 - p1
        # Inward normal: perpendicular to edge, pointing toward center
        normal = np.array([edge[1], -edge[0]])  # Rotate -90 degrees
        normal = normal / (np.linalg.norm(normal) + 1e-15)
        
        for v in all_verts:
            proj = np.dot(v, normal)
            if proj > max_proj:
                max_proj = proj
    
    # s * sqrt(3)/2 >= max_proj
    # s >= max_proj * 2 / sqrt(3)
    min_s = max_proj * 2.0 / np.sqrt(3)
    
    return max(min_s, np.sqrt(12))


def _overlap_penalty(inner):
    """Compute overlap penalty for inner hexagons using center distance approximation."""
    penalty = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
            # For unit hexagons, minimum center distance for non-overlap depends on orientation
            # Use a conservative threshold
            if dist < 1.9:
                penalty += (1.9 - dist) ** 2
    return penalty


def _is_valid(inner, outer, s):
    """Check validity of the packing configuration."""
    inner = np.array(inner, dtype=float)
    outer = np.array(outer, dtype=float)
    s = float(s)
    
    # Check finite values
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    
    # Check s >= sqrt(12)
    if s < np.sqrt(12) - 1e-9:
        return False
    
    # Check outer vertices are distinct
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
    
    # Check unit edges of inner hexagons have length ~1
    for i in range(12):
        verts = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for j in range(6):
            edge_len = np.linalg.norm(verts[j] - verts[(j + 1) % 6])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    # Check no two inner hexagons overlap (separating axis test)
    for i in range(12):
        for j in range(i + 1, 12):
            verts_i = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            verts_j = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            
            # Collect all edge normals from both hexagons
            normals = []
            for v in [verts_i, verts_j]:
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    normal = normal / (np.linalg.norm(normal) + 1e-15)
                    normals.append(normal)
            
            # Check separating axis
            overlap = True
            for normal in normals:
                proj_i = np.dot(verts_i, normal)
                proj_j = np.dot(verts_j, normal)
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                # If intervals don't overlap, we have separation
                if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                    overlap = False
                    break
            
            if overlap:
                return False
    
    # Check all inner vertices are inside or on the outer hexagon
    for i in range(12):
        verts_i = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for v in verts_i:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit
    
    # Hexagonal grid arrangement (like a honeycomb pattern)
    # Row spacing: 3*sqrt(3)/2 ≈ 2.598
    # Column spacing: 3
    # This gives a natural hexagonal packing
    row_spacing = 3.0 * np.sqrt(3) / 2
    col_spacing = 3.0
    
    initial_inner = np.array([
        [0.0, 0.0, 0.0],                          # center
        [-col_spacing, 0.0, 0.0],                 # left
        [col_spacing, 0.0, 0.0],                  # right
        [-col_spacing/2, row_spacing, 0.0],       # top-left
        [col_spacing/2, row_spacing, 0.0],        # top-right
        [-col_spacing/2, -row_spacing, 0.0],      # bottom-left
        [col_spacing/2, -row_spacing, 0.0],       # bottom-right
        [-col_spacing*1.5, row_spacing, 0.0],     # far top-left
        [col_spacing*1.5, row_spacing, 0.0],      # far top-right
        [-col_spacing*1.5, -row_spacing, 0.0],    # far bottom-left
        [col_spacing*1.5, -row_spacing, 0.0],     # far bottom-right
        [0.0, -2*row_spacing, 0.0],               # far bottom-center
    ])
    
    best_inner = initial_inner.copy()
    best_outer_center = np.array([0.0, 0.0])
    best_outer_angle = 0.0
    best_s = _compute_min_outer_side(best_inner, best_outer_center, best_outer_angle)
    
    # Try different outer hexagon angles
    for angle_deg in [0, 30, 60, 90, 120, 150]:
        s = _compute_min_outer_side(initial_inner, best_outer_center, angle_deg)
        if s < best_s:
            best_s = s
            best_outer_angle = angle_deg
    
    # Optimize inner positions to minimize the required outer side length
    from scipy.optimize import minimize
    
    def objective(params):
        """Objective: minimize the required outer hexagon side length plus overlap penalty."""
        if time.time() > deadline - 5.0:
            return 1e10  # Timeout penalty
        
        inner = params.reshape(12, 3).copy()
        
        # Compute required outer side
        s = _compute_min_outer_side(inner, best_outer_center, best_outer_angle)
        
        # Add overlap penalty
        penalty = _overlap_penalty(inner)
        
        return s + penalty
    
    # Nelder-Mead optimization with multiple trials
    x0 = initial_inner.flatten()
    best_obj = objective(x0)
    best_result = x0.copy()
    
    n_trials = 8
    for trial in range(n_trials):
        if time.time() > deadline - 30.0:
            break
        
        if trial == 0:
            x_start = x0.copy()
        else:
            np.random.seed(trial * 12345 + 7)
            x_start = x0 + np.random.randn(36) * 0.3
        
        try:
            result = minimize(
                objective, x_start,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x.copy()
                
        except Exception:
            pass
    
    # Extract best result
    candidate_inner = best_result.reshape(12, 3).copy()
    candidate_s = _compute_min_outer_side(candidate_inner, best_outer_center, best_outer_angle)
    
    # Verify validity
    if _is_valid(candidate_inner, np.array([best_outer_center[0], best_outer_center[1], best_outer_angle]), candidate_s):
        if candidate_s < best_s:
            best_inner = candidate_inner
            best_s = candidate_s
    else:
        # If invalid, try to fix by adjusting positions slightly
        # Fall back to initial if needed
        pass
    
    # Final check and return
    final_outer_center = np.array([0.0, 0.0])
    final_outer_angle = best_outer_angle
    
    # Compute final s
    final_s = _compute_min_outer_side(best_inner, final_outer_center, final_outer_angle)
    
    # Ensure validity
    if not _is_valid(best_inner, np.array([final_outer_center[0], final_outer_center[1], final_outer_angle]), final_s):
        # Fall back to initial
        best_inner = initial_inner
        final_s = _compute_min_outer_side(best_inner, final_outer_center, final_outer_angle)
    
    inner_hex_data = best_inner.astype(float)
    outer_hex_data = np.array([final_outer_center[0], final_outer_center[1], final_outer_angle])
    outer_hex_side_length = float(final_s)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
