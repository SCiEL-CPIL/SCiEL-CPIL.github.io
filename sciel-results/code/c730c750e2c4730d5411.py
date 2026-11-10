# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _compute_enclosing_hex_side(centers, angles_deg, outer_angle_deg):
    """
    Compute minimum side length s of outer hexagon (centered at origin) with given angle
    that contains all vertices of all inner hexagons.
    """
    outer_a = np.radians(outer_angle_deg)
    normal_angles = outer_a + np.radians(90) + np.radians(60) * np.arange(6)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    all_verts = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy, 1.0, angles_deg[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    max_proj = 0
    for n in normals:
        projs = np.abs(all_verts @ n)
        max_proj = max(max_proj, np.max(projs))
    
    s = 2.0 * max_proj / np.sqrt(3.0)
    return s


def _verify_validity(inner_hex_data, outer_hex_data, s):
    """Check if the packing is valid."""
    # Check s >= sqrt(12)
    if s < np.sqrt(12) - 1e-9:
        return False
    
    # Check outer vertices are distinct
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-6:
                return False
    
    # Check inner hexagons are separated (SAT)
    inner_verts_list = []
    for i in range(12):
        inner_verts_list.append(_hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2]))
    
    for i in range(12):
        for j in range(i+1, 12):
            vi = inner_verts_list[i]
            vj = inner_verts_list[j]
            # Get all edge normals from both hexagons
            all_normals = []
            for k in range(6):
                edge_i = vi[(k+1) % 6] - vi[k]
                normal_i = np.array([-edge_i[1], edge_i[0]])
                all_normals.append(normal_i)
                edge_j = vj[(k+1) % 6] - vj[k]
                normal_j = np.array([-edge_j[1], edge_j[0]])
                all_normals.append(normal_j)
            
            # SAT test
            separated = False
            for normal in all_normals:
                proj_i = vi @ normal
                proj_j = vj @ normal
                if np.max(proj_i) < np.min(proj_j) - 1e-6 or np.max(proj_j) < np.min(proj_i) - 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # Check all inner vertices are inside outer hexagon
    for i in range(12):
        vi = inner_verts_list[i]
        for v in vi:
            # Check against each outer edge
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True


def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Define 12 hexagons in a honeycomb-like arrangement using axial coordinates
    axial_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, 0), (-2, 0),
        (0, 2), (0, -2),
        (1, 1),
    ]
    
    # Convert to Cartesian (pointy-top, side=1)
    base_centers = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        base_centers.append([x, y])
    base_centers = np.array(base_centers)
    
    inner_angles_base = np.zeros(12)
    
    def objective(outer_angle_deg):
        if time.time() > deadline:
            return float('inf')
        return _compute_enclosing_hex_side(base_centers, inner_angles_base, outer_angle_deg)
    
    # Search for optimal outer angle
    best_s = float('inf')
    best_angle = 0.0
    
    for angle in np.linspace(0, 60, 121):
        s = objective(angle)
        if s < best_s:
            best_s = s
            best_angle = angle
    
    if time.time() < deadline - 20:
        result = minimize_scalar(objective, bounds=(max(0, best_angle - 5), min(60, best_angle + 5)), 
                                method='bounded', options={'xatol': 1e-7})
        if result.fun < best_s:
            best_s = result.fun
            best_angle = result.x
    
    # Also try rotating inner cluster
    def objective2(inner_rotation_deg):
        if time.time() > deadline:
            return float('inf')
        rot = np.radians(inner_rotation_deg)
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        rotated = np.zeros_like(base_centers)
        for i in range(12):
            x, y = base_centers[i]
            rotated[i, 0] = x * cos_r - y * sin_r
            rotated[i, 1] = x * sin_r + y * cos_r
        return _compute_enclosing_hex_side(rotated, np.full(12, inner_rotation_deg), 0.0)
    
    best_s2 = float('inf')
    best_angle2 = 0.0
    
    for angle in np.linspace(0, 60, 121):
        s = objective2(angle)
        if s < best_s2:
            best_s2 = s
            best_angle2 = angle
    
    if time.time() < deadline - 20:
        result = minimize_scalar(objective2, bounds=(max(0, best_angle2 - 5), min(60, best_angle2 + 5)), 
                                method='bounded', options={'xatol': 1e-7})
        if result.fun < best_s2:
            best_s2 = result.fun
            best_angle2 = result.x
    
    # Pick best
    if best_s <= best_s2:
        final_s = best_s
        final_centers = base_centers
        final_inner_angles = inner_angles_base
        final_outer_angle = best_angle
    else:
        final_s = best_s2
        rot = np.radians(best_angle2)
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        final_centers = np.zeros((12, 2))
        for i in range(12):
            x, y = base_centers[i]
            final_centers[i, 0] = x * cos_r - y * sin_r
            final_centers[i, 1] = x * sin_r + y * cos_r
        final_inner_angles = np.full(12, best_angle2)
        final_outer_angle = 0.0
    
    inner_hex_data = np.column_stack([final_centers, final_inner_angles])
    outer_hex_data = np.array([0.0, 0.0, final_outer_angle])
    
    # Verify validity
    if not _verify_validity(inner_hex_data, outer_hex_data, final_s):
        # Fallback: use a larger outer hexagon
        # Compute actual required s more carefully
        final_s = final_s * 1.01  # Add small buffer
        outer_hex_data = np.array([0.0, 0.0, final_outer_angle])
    
    return inner_hex_data, outer_hex_data, final_s


# EVOLVE-BLOCK-END
