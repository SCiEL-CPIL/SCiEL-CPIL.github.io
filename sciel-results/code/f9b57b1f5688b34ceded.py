# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


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
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
    
    # Check unit edges of inner hexagons have length ~1
    for i in range(12):
        verts = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for j in range(6):
            edge_len = np.linalg.norm(verts[j] - verts[(j+1)%6])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    # Check no two inner hexagons overlap (separating axis test)
    for i in range(12):
        for j in range(i+1, 12):
            verts_i = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            verts_j = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            
            # Collect all edge normals from both hexagons
            normals = []
            for v in [verts_i, verts_j]:
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k+1)%6]
                    edge = p2 - p1
                    # Normal to edge
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
            # Check against each outer edge
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True


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
    
    a_rad = np.radians(outer_angle)
    unit_outer_verts = np.column_stack([
        np.cos(a_rad + np.radians(60) * np.arange(6)),
        np.sin(a_rad + np.radians(60) * np.arange(6))
    ])
    
    max_proj = 0
    for k in range(6):
        p1 = unit_outer_verts[k]
        p2 = unit_outer_verts[(k+1)%6]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-15)
        
        # Vectorized: compute all projections at once
        projs = all_verts @ normal
        max_k = np.max(projs)
        if max_k > max_proj:
            max_proj = max_k
    
    min_s = max_proj * 2.0 / np.sqrt(3)
    return max(min_s, np.sqrt(12))


def _overlap_penalty(inner):
    """Compute penalty for overlapping hexagons."""
    penalty = 0.0
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
            if dist < 2.0:
                penalty += (2.0 - dist) ** 2
    return penalty


def _lattice_positions(lattice_params):
    """
    Generate 12 positions on a triangular lattice.
    lattice_params: [a1x, a1y, a2x, a2y, global_rotation_deg]
    The 12 positions are integer combinations of a1 and a2 from a set of 12 lattice vectors.
    We use a hexagonal arrangement: center + 6 neighbors + 5 more in second shell.
    """
    a1x, a1y, a2x, a2y, rot_deg = lattice_params
    rot = np.radians(rot_deg)
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    
    # Lattice basis vectors
    a1 = np.array([a1x, a1y])
    a2 = np.array([a2x, a2y])
    
    # 12 lattice points: center, 6 first neighbors, 5 second shell
    # In a triangular lattice, the 12 points can be:
    # (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1),
    # (2,0), (-2,0), (1,2), (-1,-2), (2,1)
    lattice_coords = np.array([
        [0, 0],
        [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1],
        [2, 0], [-2, 0], [1, 2], [-1, -2], [2, 1]
    ], dtype=float)
    
    # Transform to Cartesian: pos = m*a1 + n*a2, then rotate
    positions = lattice_coords[:, 0:1] * a1[None, :] + lattice_coords[:, 1:2] * a2[None, :]
    
    # Apply global rotation
    rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    positions = positions @ rot_matrix.T
    
    # All hexagons have same orientation (0 degrees)
    angles = np.zeros(12)
    
    return np.column_stack([positions[:, 0], positions[:, 1], angles])


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses lattice-based optimization with only 5 parameters (lattice basis + rotation).
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    # Initial lattice: standard triangular lattice with spacing ~2.0 for unit hexagons
    # For unit hexagons, minimum center distance is 2*sqrt(3)/2 = sqrt(3) ≈ 1.732 for edge-to-edge
    # But for hexagons of side 1, the distance between centers of adjacent hexagons in a honeycomb
    # is 2*sqrt(3)/2 * 2 = 2*sqrt(3) ≈ 3.464? No, let me reconsider.
    # Two unit hexagons sharing an edge: center distance = 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732
    # Wait, apothem of unit hexagon = sqrt(3)/2. So center distance for edge-sharing = 2 * apothem = sqrt(3).
    # But that would cause overlap in other directions. Let me use a larger spacing.
    
    # For a triangular lattice, the basis vectors should be at 60 degrees.
    # Initial guess: spacing = 2.0 (safe, no overlap)
    spacing = 2.0
    a1_init = np.array([spacing, 0.0])
    a2_init = spacing * np.array([0.5, np.sqrt(3)/2])
    
    initial_params = np.array([a1_init[0], a1_init[1], a2_init[0], a2_init[1], 0.0])
    
    best_params = initial_params.copy()
    best_inner = _lattice_positions(initial_params)
    best_outer_center = np.array([0.0, 0.0])
    best_outer_angle = 0.0
    best_s = _compute_min_outer_side(best_inner, best_outer_center, best_outer_angle)
    
    # Try different outer hexagon angles
    for angle_deg in [0, 30, 60, 90, 120, 150]:
        s = _compute_min_outer_side(best_inner, best_outer_center, angle_deg)
        if s < best_s:
            best_s = s
            best_outer_angle = angle_deg
    
    # Simulated Annealing on the 5 lattice parameters
    current_params = initial_params.copy()
    current_s = _compute_min_outer_side(_lattice_positions(current_params), best_outer_center, best_outer_angle)
    
    T_initial = 0.3
    T_min = 0.0001
    n_iterations = 50000
    
    np.random.seed(42)
    
    for it in range(n_iterations):
        if time.time() > deadline - 60.0:
            break
        
        T = T_initial * (T_min / T_initial) ** (it / n_iterations)
        
        # Generate a random perturbation of the 5 parameters
        new_params = current_params.copy()
        # Perturb with decreasing step size
        step = 0.05 * (T / T_initial) ** 0.5
        new_params += np.random.randn(5) * step
        
        # Keep rotation in [0, 60)
        new_params[4] = new_params[4] % 60.0
        
        new_inner = _lattice_positions(new_params)
        new_s = _compute_min_outer_side(new_inner, best_outer_center, best_outer_angle)
        
        # Check validity
        new_valid = _is_valid(new_inner, np.array([best_outer_center[0], best_outer_center[1], best_outer_angle]), new_s)
        
        if new_valid:
            if new_s < current_s:
                current_params = new_params
                current_s = new_s
                if current_s < best_s:
                    best_params = new_params.copy()
                    best_inner = new_inner.copy()
                    best_s = current_s
            else:
                delta = new_s - current_s
                if np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                    current_params = new_params
                    current_s = new_s
        else:
            # Invalid: accept with small probability if close to valid
            penalty = _overlap_penalty(new_inner)
            if penalty < 0.5 and np.random.rand() < 0.05:
                current_params = new_params
                current_s = new_s + penalty
    
    # Final refinement with Nelder-Mead on the 5 parameters
    from scipy.optimize import minimize
    
    def objective(params):
        if time.time() > deadline - 10.0:
            return 1e10
        params = params.copy()
        params[4] = params[4] % 60.0
        inner = _lattice_positions(params)
        s = _compute_min_outer_side(inner, best_outer_center, best_outer_angle)
        if not _is_valid(inner, np.array([best_outer_center[0], best_outer_center[1], best_outer_angle]), s):
            return s + _overlap_penalty(inner)
        return s
    
    x0 = best_params.copy()
    try:
        result = minimize(
            objective, x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if result.fun < best_s:
            candidate_params = result.x.copy()
            candidate_params[4] = candidate_params[4] % 60.0
            candidate_inner = _lattice_positions(candidate_params)
            candidate_s = _compute_min_outer_side(candidate_inner, best_outer_center, best_outer_angle)
            if _is_valid(candidate_inner, np.array([best_outer_center[0], best_outer_center[1], best_outer_angle]), candidate_s):
                best_params = candidate_params
                best_inner = candidate_inner
                best_s = candidate_s
    except Exception:
        pass
    
    # Final check and return
    final_outer_center = np.array([0.0, 0.0])
    final_outer_angle = best_outer_angle
    
    # Compute final s
    final_s = _compute_min_outer_side(best_inner, final_outer_center, final_outer_angle)
    
    # Ensure validity
    if not _is_valid(best_inner, np.array([final_outer_center[0], final_outer_center[1], final_outer_angle]), final_s):
        # Fall back to initial
        best_inner = _lattice_positions(initial_params)
        final_s = _compute_min_outer_side(best_inner, final_outer_center, final_outer_angle)
    
    inner_hex_data = best_inner.astype(float)
    outer_hex_data = np.array([final_outer_center[0], final_outer_center[1], final_outer_angle])
    outer_hex_side_length = float(final_s)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
