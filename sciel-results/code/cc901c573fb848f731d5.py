# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _check_separation(hex1_verts, hex2_verts, tol=1e-6):
    """Separating axis test: return True if hexagons are separated (no overlap)."""
    axes = []
    for verts in (hex1_verts, hex2_verts):
        for i in range(6):
            edge = verts[(i + 1) % 6] - verts[i]
            # Normal perpendicular to edge
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)
    
    for axis in axes:
        proj1 = hex1_verts @ axis
        proj2 = hex2_verts @ axis
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        # If separated: max1 < min2 or max2 < min1 (with tolerance)
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    return False

def _vertex_inside_outer(v, outer_verts, tol=1e-6):
    """Check if vertex v is inside or on the boundary of outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
        if cross < -tol:
            return False
    return True

def _compute_min_side(inner_positions, outer_center, outer_angle_deg):
    """
    Given inner hexagon centers (12,2) and outer hexagon center and angle,
    compute the minimum side length s of a regular hexagon centered at outer_center
    with the given angle that contains all inner unit hexagons.
    """
    outer_angle = np.deg2rad(outer_angle_deg)
    # Outer hexagon edges: for each of 6 edges, compute the distance from center to edge
    # A regular hexagon with side s has inradius = s * sqrt(3)/2
    # The distance from center to each edge is s * sqrt(3)/2
    # For a vertex v to be inside, for each edge direction, the projection must be <= inradius
    
    # Edge normals of the outer hexagon (pointing inward)
    edge_angles = outer_angle + np.deg2rad(60 * np.arange(6))
    # Actually, the edges are at angles offset by 30 deg from vertices
    # Let's think: vertices are at angles outer_angle + 60*i
    # Edges connect vertex i to vertex i+1
    # The normal to edge i points outward at angle outer_angle + 30 + 60*i
    
    # For containment: for each inner vertex, for each outer edge,
    # the signed distance from the vertex to the edge line must be >= 0 (inside)
    # Equivalently, the projection of (v - center) onto the outward normal <= inradius
    
    outward_normals = []
    for i in range(6):
        va = outer_angle + np.deg2rad(60 * i)
        vb = outer_angle + np.deg2rad(60 * (i + 1))
        # Edge from vertex i to vertex i+1
        # Outward normal direction
        # Midpoint of edge is at angle outer_angle + 30 + 60*i from center
        normal_angle = outer_angle + np.deg2rad(30 + 60 * i)
        outward_normals.append(np.array([np.cos(normal_angle), np.sin(normal_angle)]))
    outward_normals = np.array(outward_normals)  # (6,2)
    
    # Compute all vertices of all 12 inner hexagons
    all_verts = []
    for pos in inner_positions:
        verts = _hex_vertices(pos[0], pos[1], 1.0, 0.0)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # (72, 2)
    
    # For each outward normal, compute max projection of (v - outer_center)
    rel = all_verts - outer_center  # (72, 2)
    max_projs = np.max(rel @ outward_normals.T, axis=0)  # (6,)
    
    # inradius = s * sqrt(3)/2, so s = inradius / (sqrt(3)/2) = 2 * inradius / sqrt(3)
    max_inradius = np.max(max_projs)
    if max_inradius < 0:
        max_inradius = 0
    s = 2.0 * max_inradius / np.sqrt(3.0)
    return s

def _objective(params, outer_angle_deg):
    """Objective: minimize s given parameter vector."""
    # params: 24 values (12 positions) + 1 outer center x + 1 outer center y = 26
    inner_positions = params[:24].reshape(12, 2)
    outer_center = params[24:26]
    return _compute_min_side(inner_positions, outer_center, outer_angle_deg)

def _validate(inner_hex_data, outer_hex_data, s):
    """Validate the packing. Return True if valid."""
    # Check s >= sqrt(12)
    if s < np.sqrt(12) - 1e-8:
        return False
    # Check outer vertices distinct
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    # Check inner hexagons are unit
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for k in range(6):
            edge_len = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    # Check separation between all pairs
    inner_verts_list = []
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        inner_verts_list.append(verts)
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
    # Check all inner vertices inside outer
    for i in range(12):
        for v in inner_verts_list[i]:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    return True

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner_hex_data, outer_hex_data, s)
    """
    start_time = time.time()
    deadline = start_time + 600.0 - 30.0  # 30s safety margin
    
    # Initial honeycomb-like arrangement
    # 4 columns x 3 rows pattern
    # Hexagon spacing: horizontal = 2.5, vertical = 2.165 (sqrt(3))
    sqrt3 = np.sqrt(3)
    
    # Start with a reasonable initial configuration
    # 12 hexagons in a 4x3 grid pattern
    dx = 2.5
    dy = 2.165  # approx sqrt(3)
    
    # Positions for 4x3 grid
    positions = []
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * dx
            y = (row - 1.0) * dy
            if row % 2 == 1:
                x += dx / 2  # offset middle row
            positions.append([x, y])
    positions = np.array(positions)
    
    # Center the configuration
    center = positions.mean(axis=0)
    positions -= center
    
    # Initial outer hexagon: center at origin, angle 0
    outer_center = np.array([0.0, 0.0])
    outer_angle_deg = 0.0
    
    # Initial s
    s = _compute_min_side(positions, outer_center, outer_angle_deg)
    
    best_params = np.concatenate([positions.flatten(), outer_center])
    best_s = s
    
    # Optimization using Nelder-Mead
    def objective_func(p):
        inner_pos = p[:24].reshape(12, 2)
        oc = p[24:26]
        return _compute_min_side(inner_pos, oc, outer_angle_deg)
    
    # Try optimization
    if time.time() < deadline:
        result = minimize(objective_func, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < best_s:
            best_s = result.fun
            best_params = result.x
    
    # Try different outer angles for better fit
    best_overall_s = best_s
    best_overall_params = best_params.copy()
    
    for angle_deg in [0, 30, 60, 90, 120, 150]:
        if time.time() > deadline:
            break
        
        outer_angle_deg = angle_deg
        
        def obj_angle(p):
            inner_pos = p[:24].reshape(12, 2)
            oc = p[24:26]
            return _compute_min_side(inner_pos, oc, outer_angle_deg)
        
        # Restart optimization from current best
        init_params = best_overall_params.copy()
        result = minimize(obj_angle, init_params, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < best_overall_s:
            best_overall_s = result.fun
            best_overall_params = result.x
            best_overall_angle = angle_deg
        else:
            best_overall_angle = angle_deg if angle_deg == 0 else best_overall_angle if 'best_overall_angle' in dir() else 0
    
    # Extract best solution
    inner_positions = best_overall_params[:24].reshape(12, 2)
    outer_center = best_overall_params[24:26]
    
    # Build return arrays
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = inner_positions[i, 0]
        inner_hex_data[i, 1] = inner_positions[i, 1]
        inner_hex_data[i, 2] = 0.0  # angle in degrees
    
    outer_hex_data = np.array([outer_center[0], outer_center[1], 0.0])
    
    s_final = best_overall_s
    
    # Validate
    if not _validate(inner_hex_data, outer_hex_data, s_final):
        # Fallback to a known valid configuration
        # Use the initial honeycomb with a safe s
        inner_hex_data = np.zeros((12, 3))
        positions_fb = positions.copy()
        for i in range(12):
            inner_hex_data[i, 0] = positions_fb[i, 0]
            inner_hex_data[i, 1] = positions_fb[i, 1]
            inner_hex_data[i, 2] = 0.0
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        s_final = _compute_min_side(positions_fb, np.array([0.0, 0.0]), 0.0) + 0.1  # small buffer
    
    return inner_hex_data, outer_hex_data, s_final


# EVOLVE-BLOCK-END
