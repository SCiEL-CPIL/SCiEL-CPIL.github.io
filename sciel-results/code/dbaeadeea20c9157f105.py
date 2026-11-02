"""
Hexagon packing: 12 unit hexagons in a regular hexagon, minimize outer side length.
Strategy:
- Generate many compact hexagonal-lattice initial configurations.
- For each: coarse global-rotation search, then local Nelder-Mead on (positions, angles, outer side)
  with a penalty for overlaps / containment violation.
- Final high-precision Nelder-Mead polish.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # safety margin

# Unit hexagon vertices (pointy-top: angles 30, 90, 150, 210, 270, 330)
UNIT_VERTS = np.array([
    [np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)
])

# Outward normals for the 6 edges of a regular hexagon (pointing to edge midpoints)
# For pointy-top, edge midpoints at 0, 60, 120, 180, 240, 300 degrees
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])


def get_vertices(centers):
    """
    centers: (N, 3) array of [x, y, angle_deg]
    Returns: (N, 6, 2) array of vertices
    """
    n = len(centers)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        x, y, a_deg = centers[i]
        a_rad = np.radians(a_deg)
        c = np.cos(a_rad)
        s = np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        verts[i] = UNIT_VERTS @ R.T + np.array([x, y])
    return verts


def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    verts = get_vertices(np.array(centers, dtype=float))
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    max_extent = 0.0
    for i in range(len(centers)):
        v = verts[i]
        v_rel = v - outer_center
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        max_extent = max(max_extent, np.max(max_proj))
    
    return max_extent / (np.sqrt(3) / 2)


def sat_min_separation(v1, v2):
    """Compute minimum separation between two hexagons using SAT. Returns scalar."""
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    axes = np.vstack([edges1, edges2])
    axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15)
    
    proj1 = v1 @ axes.T
    proj2 = v2 @ axes.T
    
    sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                    proj2.min(axis=0) - proj1.max(axis=0))
    
    return np.max(sep)


def min_pair_separation(centers):
    """Compute minimum separation among all pairs of inner hexagons."""
    verts = get_vertices(np.array(centers, dtype=float))
    n = verts.shape[0]
    min_sep = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            s = sat_min_separation(verts[i], verts[j])
            if s < min_sep:
                min_sep = s
    return min_sep


def min_outer_clearance(centers, outer_center, outer_angle, outer_side):
    """Compute minimum clearance from inner hexagons to outer boundary."""
    verts = get_vertices(np.array(centers, dtype=float))
    apothem = outer_side * np.sqrt(3) / 2
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    min_cl = float('inf')
    for i in range(len(centers)):
        v = verts[i]
        v_rel = v - outer_center
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        cl = apothem - max_proj
        if np.min(cl) < min_cl:
            min_cl = np.min(cl)
    return min_cl


def penalty_objective(params, outer_center, outer_angle, beta=20.0):
    """
    Objective: outer_side + penalty for violations.
    params: [x0,y0,a0, ..., x11,y11,a11, side]
    """
    n = N
    centers = []
    for i in range(n):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        centers.append([x, y, a])
    side = params[3*n]
    
    if side < 1.0:
        return 1e6
    
    min_sep = min_pair_separation(centers)
    min_cl = min_outer_clearance(centers, outer_center, outer_angle, side)
    
    # Penalty for negative clearances
    penalty = 0.0
    if min_sep < 0:
        penalty += (-min_sep) ** 2 * 50.0
    if min_cl < 0:
        penalty += (-min_cl) ** 2 * 50.0
    
    return side + penalty


def generate_hex_lattice_arrangements():
    """Generate several compact hexagonal lattice arrangements of 12 hexagons."""
    arrangements = []
    
    # Hexagonal lattice vectors for pointy-top hexagons
    # Distance between adjacent centers = 2 * apothem = sqrt(3)
    a1 = np.array([np.sqrt(3), 0])
    a2 = np.array([np.sqrt(3) / 2, 1.5])
    
    # Pattern 1: 3-4-3-2
    pattern1 = [
        (0, 0),
        (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1),
        (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3)
    ]
    
    # Pattern 2: 2-3-4-3
    pattern2 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    
    # Pattern 3: 1-3-4-3-1
    pattern3 = [
        (0, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3),
        (0, 4)
    ]
    
    # Pattern 4: 4-4-4
    pattern4 = [
        (-1.5, 0), (-0.5, 0), (0.5, 0), (1.5, 0),
        (-1.0, 1.5), (0.0, 1.5), (1.0, 1.5), (2.0, 1.5),
        (-1.5, 3.0), (-0.5, 3.0), (0.5, 3.0), (1.5, 3.0)
    ]
    
    # Pattern 5: 2-4-4-2
    pattern5 = [
        (-0.5, 0), (0.5, 0),
        (-1.5, 1.5), (-0.5, 1.5), (0.5, 1.5), (1.5, 1.5),
        (-1.5, 3.0), (-0.5, 3.0), (0.5, 3.0), (1.5, 3.0),
        (-0.5, 4.5), (0.5, 4.5)
    ]
    
    # Pattern 6: Center + ring1 + partial ring2
    pattern6 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (2, 2)
    ]
    
    # Pattern 7: 3-3-3-3 staggered
    pattern7 = [
        (-1, 0), (0, 0), (1, 0),
        (-0.5, 1.5), (0.5, 1.5), (1.5, 1.5),
        (-1, 3.0), (0, 3.0), (1, 3.0),
        (-0.5, 4.5), (0.5, 4.5), (1.5, 4.5)
    ]
    
    # Pattern 8: 2-3-3-2-2
    pattern8 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    
    # Pattern 9: 3-4-4-1
    pattern9 = [
        (-1, 0), (0, 0), (1, 0),
        (-1.5, 1.5), (-0.5, 1.5), (0.5, 1.5), (1.5, 1.5),
        (-1.5, 3.0), (-0.5, 3.0), (0.5, 3.0), (1.5, 3.0),
        (0, 4.5)
    ]
    
    # Pattern 10: 4-3-3-2
    pattern10 = [
        (-1.5, 0), (-0.5, 0), (0.5, 0), (1.5, 0),
        (-1.0, 1.5), (0.0, 1.5), (1.0, 1.5),
        (-1.5, 3.0), (-0.5, 3.0), (0.5, 3.0),
        (-0.5, 4.5), (0.5, 4.5)
    ]
    
    for pattern in [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7, pattern8, pattern9, pattern10]:
        centers = []
        for (i, j) in pattern:
            pos = i * a1 + j * a2
            centers.append([pos[0], pos[1], 0.0])
        arrangements.append(centers)
    
    return arrangements


def optimize_global_rotation(centers, outer_center, num_steps=120):
    """
    Find the best global rotation of the inner arrangement to minimize outer side length.
    """
    best_side = float('inf')
    best_angle = 0.0
    
    angles = np.linspace(0, 60, num_steps)
    
    for rot_deg in angles:
        rotated_centers = []
        for c in centers:
            rotated_centers.append([c[0], c[1], c[2] + rot_deg])
        
        side = required_outer_side(rotated_centers, outer_center, 0.0)
        
        if side < best_side:
            best_side = side
            best_angle = rot_deg
    
    return best_side, best_angle


def refine_rotation(centers, outer_center, center_angle, num_steps=50, range_deg=5.0):
    """Fine-tune the rotation angle around a given center angle."""
    best_side = float('inf')
    best_angle = center_angle
    
    angles = np.linspace(center_angle - range_deg, center_angle + range_deg, num_steps)
    
    for rot_deg in angles:
        rotated_centers = []
        for c in centers:
            rotated_centers.append([c[0], c[1], c[2] + rot_deg])
        
        side = required_outer_side(rotated_centers, outer_center, 0.0)
        
        if side < best_side:
            best_side = side
            best_angle = rot_deg
    
    return best_side, best_angle


def nelder_mead_polish(centers, outer_center, outer_angle, time_limit=15.0):
    """
    Polish a configuration using Nelder-Mead on all parameters + outer side.
    """
    start_time = time.time()
    
    req_side = required_outer_side(centers, outer_center, outer_angle)
    
    params = np.zeros(3 * N + 1)
    for i in range(N):
        params[3*i] = centers[i][0]
        params[3*i+1] = centers[i][1]
        params[3*i+2] = centers[i][2]
    params[3*N] = req_side
    
    def obj(p):
        if time.time() - start_time > time_limit:
            return 1e6
        return penalty_objective(p, outer_center, outer_angle)
    
    result = minimize(
        obj, params, method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )
    
    new_centers = []
    for i in range(N):
        new_centers.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
    new_side = result.x[3*N]
    
    return new_centers, new_side


def high_precision_polish(centers, outer_center, outer_angle, time_limit=10.0):
    """
    Final high-precision polish with very tight tolerances.
    """
    start_time = time.time()
    
    req_side = required_outer_side(centers, outer_center, outer_angle)
    
    params = np.zeros(3 * N + 1)
    for i in range(N):
        params[3*i] = centers[i][0]
        params[3*i+1] = centers[i][1]
        params[3*i+2] = centers[i][2]
    params[3*N] = req_side
    
    def obj(p):
        if time.time() - start_time > time_limit:
            return 1e6
        return penalty_objective(p, outer_center, outer_angle, beta=30.0)
    
    result = minimize(
        obj, params, method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    
    new_centers = []
    for i in range(N):
        new_centers.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
    new_side = result.x[3*N]
    
    return new_centers, new_side


def check_validity(centers, outer_center, outer_side, outer_angle, tol=1e-8):
    """Check if packing is valid: all inner hexagons inside outer, no overlaps."""
    # Check containment
    actual_side = required_outer_side(centers, outer_center, outer_angle)
    if actual_side > outer_side + tol:
        return False
    
    # Check pairwise separation
    min_sep = min_pair_separation(centers)
    if min_sep < -tol:
        return False
    
    return True


def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    configs = generate_hex_lattice_arrangements()
    
    for idx, config in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 25:
            break
        
        # Phase 1: Global rotation optimization
        rot_time = min(5.0, DEADLINE - elapsed - 20)
        if rot_time < 2:
            rot_time = 2
        
        try:
            coarse_side, coarse_angle = optimize_global_rotation(config, outer_center, num_steps=60)
            fine_side, fine_angle = refine_rotation(config, outer_center, coarse_angle, num_steps=30, range_deg=3.0)
        except Exception:
            fine_side = required_outer_side(config, outer_center, outer_angle)
            fine_angle = 0.0
        
        # Apply best rotation
        rotated_config = []
        for c in config:
            rotated_config.append([c[0], c[1], c[2] + fine_angle])
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
        
        # Phase 2: Nelder-Mead polish
        polish_time = min(12.0, DEADLINE - elapsed - 8)
        if polish_time < 5:
            polish_time = 5
        
        try:
            polished, polished_side = nelder_mead_polish(rotated_config, outer_center, outer_angle, time_limit=polish_time)
        except Exception:
            polished = rotated_config
            polished_side = required_outer_side(polished, outer_center, outer_angle)
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 8:
            break
        
        # Validate
        valid = check_validity(polished, outer_center, polished_side, outer_angle)
        
        if valid and polished_side < best_score:
            best_score = polished_side
            best_result = (polished, outer_center, polished_side, outer_angle)
    
    # Final high-precision polish if we have time
    if best_result is not None and time.time() - start_time < DEADLINE - 12:
        try:
            polished_centers, polished_side = high_precision_polish(
                best_result[0], outer_center, outer_angle, time_limit=10.0
            )
            valid = check_validity(polished_centers, outer_center, polished_side, outer_angle)
            if valid and polished_side < best_score:
                best_score = polished_side
                best_result = (polished_centers, outer_center, polished_side, outer_angle)
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    
    # Fallback: circular arrangement
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = required_outer_side(inner_default, outer_center, outer_angle)
    return inner_default, outer_center, required_side, outer_angle
