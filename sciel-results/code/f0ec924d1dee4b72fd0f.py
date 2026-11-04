import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, s, angle_deg):
    """Return vertices of regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = angle_rad + k * np.pi / 3
        verts.append((cx + s * np.cos(theta), cy + s * np.sin(theta)))
    return np.array(verts)

def _is_inside_hex(point, cx, cy, S, angle_deg):
    """Check if point is inside regular hexagon centered at (cx,cy), side S, rotation angle_deg."""
    angle_rad = np.radians(angle_deg)
    # Transform to local frame
    dx = point[0] - cx
    dy = point[1] - cy
    # Rotate by -angle_rad
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    u = dx * cos_a - dy * sin_a
    v = dx * sin_a + dy * cos_a
    # Apothem
    a = S * np.sqrt(3) / 2
    # 6 edge normals at 30 + k*60 degrees
    for k in range(6):
        phi = np.radians(30 + k * 60)
        proj = u * np.cos(phi) + v * np.sin(phi)
        if proj > a + 1e-9:
            return False
    return True

def _max_extent_in_direction(vertices, direction):
    """Max projection of vertices onto direction (unit vector)."""
    return np.max(vertices @ direction)

def _min_extent_in_direction(vertices, direction):
    """Min projection of vertices onto direction (unit vector)."""
    return np.min(vertices @ direction)

def _outer_side_required(inner_positions, inner_angles, outer_center, outer_angle_deg):
    """Compute the minimum outer side length S such that all inner hexagons fit inside outer hexagon."""
    cx, cy = outer_center
    angle_rad = np.radians(outer_angle_deg)
    # Outer hexagon edge normals in global frame
    normals = []
    for k in range(6):
        phi = np.radians(30 + k * 60) + angle_rad
        normals.append((np.cos(phi), np.sin(phi)))
    
    # For each inner hexagon, compute its extent in each normal direction
    # The outer hexagon constraint: for each normal n_k, the max projection of all points <= cx*n_kx + cy*n_ky + a
    # where a = S * sqrt(3)/2
    # So S >= 2/sqrt(3) * (max_proj - (cx*n_kx + cy*n_ky))
    
    max_S = 0.0
    for i in range(len(inner_positions)):
        px, py, pa = inner_positions[i], inner_angles[i]
        verts = _hex_vertices(px, py, 1.0, pa)
        for k in range(6):
            nx, ny = normals[k]
            # Max projection of this hexagon's vertices onto normal k
            proj = verts[:, 0] * nx + verts[:, 1] * ny
            max_proj = np.max(proj)
            # Required: max_proj <= cx*nx + cy*ny + a => a >= max_proj - cx*nx - cy*ny
            required_a = max_proj - (cx * nx + cy * ny)
            if required_a > 0:
                S_candidate = 2.0 * required_a / np.sqrt(3)
                if S_candidate > max_S:
                    max_S = S_candidate
    return max_S

def _objective(params):
    """Objective: outer side length. params = [x0,y0,a0, x1,y1,a1, ..., outer_cx, outer_cy, outer_angle_deg]"""
    n_params = 3 * N + 3
    inner_x = params[0:3*N:3]
    inner_y = params[1:3*N:3]
    inner_angles = params[2:3*N:3]
    outer_cx = params[3*N]
    outer_cy = params[3*N+1]
    outer_angle = params[3*N+2]
    
    inner_positions = list(zip(inner_x, inner_y))
    S = _outer_side_required(inner_positions, inner_angles, (outer_cx, outer_cy), outer_angle)
    return S

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Initial guess: honeycomb arrangement
    # Place hexagons in a roughly hexagonal cluster
    # For 12 hexagons, use a 3x4 or similar arrangement
    # Honeycomb spacing: horizontal distance = sqrt(3), vertical distance = 1.5
    # Offset rows by sqrt(3)/2
    
    positions = []
    angles = [0.0] * N
    
    # Arrange in 4 rows of 3
    # Row y-offsets: 0, 1.5, 3.0, 4.5
    # Even rows: x = 0, sqrt(3), 2*sqrt(3)
    # Odd rows: x = sqrt(3)/2, 3*sqrt(3)/2, 5*sqrt(3)/2
    
    sqrt3 = np.sqrt(3)
    idx = 0
    for row in range(4):
        y = row * 1.5
        if row % 2 == 0:
            x_offsets = [0, sqrt3, 2*sqrt3]
        else:
            x_offsets = [sqrt3/2, 3*sqrt3/2, 5*sqrt3/2]
        for x_off in x_offsets:
            if idx < N:
                positions.append((x_off, y, 0.0))
                idx += 1
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx0 = (min(xs) + max(xs)) / 2
    cy0 = (min(ys) + max(ys)) / 2
    
    params0 = []
    for px, py, pa in positions:
        params0.append(px - cx0)
        params0.append(py - cy0)
        params0.append(pa)
    params0.append(0.0)  # outer center x
    params0.append(0.0)  # outer center y
    params0.append(0.0)  # outer angle
    
    best_params = np.array(params0)
    best_S = _objective(best_params)
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(_objective, best_params, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        if result.fun < best_S:
            best_S = result.fun
            best_params = result.x
    except Exception:
        pass
    
    if time.time() > deadline:
        pass  # Skip further optimization
    
    # Extract results
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        angle = best_params[3*i+2]
        inner.append([x, y, angle])
    
    outer_cx = best_params[3*N]
    outer_cy = best_params[3*N+1]
    outer_angle = best_params[3*N+2]
    outer_side = _outer_side_required([(p[0], p[1]) for p in inner], 
                                      [p[2] for p in inner], 
                                      (outer_cx, outer_cy), 
                                      outer_angle)
    
    return inner, (outer_cx, outer_cy), outer_side, outer_angle
# EVOLVE-BLOCK-END
