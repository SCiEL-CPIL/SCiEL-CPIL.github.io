"""Packing 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 11
# Vertices of a unit regular hexagon (side length 1), centered at origin
# Using flat-top orientation: vertices at angles 0, 60, 120, 180, 240, 300 degrees
_HEX_VERTS = np.array([
    [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)
])

# The 6 outward normals of the unit hexagon (for support function)
# For a regular hexagon with flat-top, the normals point at 0, 60, 120, 180, 240, 300 degrees
# Actually for a hexagon with vertices at 0,60,..., the face normals are at 30, 90, 150, 210, 270, 330
_HEX_NORMALS = np.array([
    [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(30, 360, 60)
])

def _transform_hex(cx, cy, angle_deg):
    """Return vertices of a unit hexagon centered at (cx,cy) rotated by angle_deg."""
    theta = np.deg2rad(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return cx + _HEX_VERTS[:, 0] * np.cos(theta) - _HEX_VERTS[:, 1] * np.sin(theta), \
           cy + _HEX_VERTS[:, 0] * np.sin(theta) + _HEX_VERTS[:, 1] * np.cos(theta)

def _hex_support(cx, cy, angle_deg, normal):
    """Support function of a unit hexagon at (cx,cy) rotated by angle_deg in direction normal."""
    theta = np.deg2rad(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    # Support = max over vertices of dot(vertex, normal) + dot(center, normal)
    verts = R @ _HEX_VERTS.T
    return np.max(verts.T @ normal) + cx * normal[0] + cy * normal[1]

def _min_distance_hex_hex(p1, a1, p2, a2):
    """
    Compute minimum distance between two unit hexagons.
    p1, p2: (x,y) centers; a1, a2: angles in degrees.
    Returns the minimum gap (positive if disjoint, negative if overlapping).
    We use the support function approach: two convex polygons are disjoint iff
    there exists a separating axis. For regular hexagons, we check all 12 edge normals.
    """
    theta1 = np.deg2rad(a1)
    theta2 = np.deg2rad(a2)
    R1 = np.array([[np.cos(theta1), -np.sin(theta1)],
                   [np.sin(theta1), np.cos(theta1)]])
    R2 = np.array([[np.cos(theta2), -np.sin(theta2)],
                   [np.sin(theta2), np.cos(theta2)]])
    
    # Get all 12 candidate separating axes (6 from each hexagon)
    axes1 = (R1 @ _HEX_NORMALS.T).T  # 6 x 2
    axes2 = (R2 @ _HEX_NORMALS.T).T  # 6 x 2
    axes = np.vstack([axes1, axes2])
    
    # For each axis, compute the gap
    d = p2 - p1
    min_gap = np.inf
    for axis in axes:
        # Project both hexagons onto this axis
        proj1 = np.max((R1 @ _HEX_VERTS.T).T @ axis) + p1 @ axis
        proj2 = np.min((R2 @ _HEX_VERTS.T).T @ axis) + p2 @ axis
        gap = proj2 - proj1
        if gap < min_gap:
            min_gap = gap
        # Also check the other direction
        proj1b = np.min((R1 @ _HEX_VERTS.T).T @ axis) + p1 @ axis
        proj2b = np.max((R2 @ _HEX_VERTS.T).T @ axis) + p2 @ axis
        gapb = proj1b - proj2b
        if gapb < min_gap:
            min_gap = gapb
    
    return min_gap

def _is_valid(positions, angles, outer_side, outer_center, outer_angle_deg, tol=1e-6):
    """
    Check if all hexagons are disjoint and inside the outer hexagon.
    positions: (N,2) array of centers
    angles: (N,) array of angles in degrees
    outer_side: side length of outer hexagon
    outer_center: (2,) center of outer hexagon
    outer_angle_deg: rotation of outer hexagon
    """
    # Check pairwise disjointness
    for i in range(N):
        for j in range(i+1, N):
            gap = _min_distance_hex_hex(positions[i], angles[i], positions[j], angles[j])
            if gap < -tol:
                return False
    
    # Check containment in outer hexagon
    # The outer hexagon has side length outer_side, so its "radius" (center to vertex) is outer_side
    # and its "inradius" (center to face midpoint) is outer_side * sqrt(3)/2
    # A unit hexagon centered at (cx,cy) rotated by angle has support in direction n:
    #   max over its vertices of dot(v, n)
    # For it to be inside the outer hexagon, for each of the 6 face normals of the outer hex:
    #   support_inner <= inradius_outer - 0 (actually the face is at distance inradius from center)
    # More precisely: the outer hexagon in direction n (face normal) extends to inradius_outer.
    # The inner hexagon must satisfy: max_v dot(v - outer_center, n) <= inradius_outer
    # i.e., dot(outer_center, n) + max_v dot(v, n) <= inradius_outer
    # Wait, let me be more careful.
    # Outer hexagon: center oc, side s, rotation oa. Its face normals are at angles oa+30, oa+90, ...
    # The distance from center to each face is s * sqrt(3)/2.
    # For a point p to be inside: for each face normal n_k, dot(p - oc, n_k) <= s*sqrt(3)/2
    # For a convex body B with support h_B(n) = max_{x in B} dot(x, n):
    #   B is inside the outer hex iff for each face normal n_k:
    #   h_B(n_k) <= dot(oc, n_k) + s*sqrt(3)/2
    #   where h_B(n_k) = max_{x in B} dot(x, n_k)
    
    theta_o = np.deg2rad(outer_angle_deg)
    R_o = np.array([[np.cos(theta_o), -np.sin(theta_o)],
                    [np.sin(theta_o), np.cos(theta_o)]])
    outer_normals = (R_o @ _HEX_NORMALS.T).T  # 6 x 2
    inradius_outer = outer_side * np.sqrt(3) / 2.0
    
    for i in range(N):
        theta_i = np.deg2rad(angles[i])
        R_i = np.array([[np.cos(theta_i), -np.sin(theta_i)],
                        [np.sin(theta_i), np.cos(theta_i)]])
        # Support of inner hex i in each outer face normal direction
        verts_i = (R_i @ _HEX_VERTS.T).T  # 6 x 2
        for nk in outer_normals:
            support = np.max(verts_i @ nk) + positions[i] @ nk
            limit = outer_center[0] * nk[0] + outer_center[1] * nk[1] + inradius_outer
            if support > limit + tol:
                return False
    
    return True

def _objective(params, outer_center, outer_angle_deg):
    """
    Objective: minimize outer_side such that all hexagons fit.
    params: 3*N values: x1,y1,a1, x2,y2,a2, ..., xN,yN,aN
    Returns the minimal outer side needed, or a large penalty if infeasible.
    """
    positions = params[0::3].copy()
    angles = params[1::3].copy()
    # Actually let's restructure: params = [x1,y1,a1, x2,y2,a2, ...]
    # That's 3 per hexagon
    positions = np.array([[params[3*i], params[3*i+1]] for i in range(N)])
    angles = np.array([params[3*i+2] for i in range(N)])
    
    # Compute the minimal outer side needed
    # For each outer face normal direction, find the max support of all inner hexagons
    # outer_side_min = max over all face normals of: (max_i support_i(n) - dot(oc, n)) / (sqrt(3)/2)
    theta_o = np.deg2rad(outer_angle_deg)
    R_o = np.array([[np.cos(theta_o), -np.sin(theta_o)],
                    [np.sin(theta_o), np.cos(theta_o)]])
    outer_normals = (R_o @ _HEX_NORMALS.T).T  # 6 x 2
    
    max_needed = 0.0
    for nk in outer_normals:
        max_support = -np.inf
        for i in range(N):
            theta_i = np.deg2rad(angles[i])
            R_i = np.array([[np.cos(theta_i), -np.sin(theta_i)],
                            [np.sin(theta_i), np.cos(theta_i)]])
            verts_i = (R_i @ _HEX_VERTS.T).T
            sup = np.max(verts_i @ nk) + positions[i][0] * nk[0] + positions[i][1] * nk[1]
            if sup > max_support:
                max_support = sup
        needed = (max_support - outer_center[0] * nk[0] - outer_center[1] * nk[1]) / (np.sqrt(3) / 2.0)
        if needed > max_needed:
            max_needed = needed
    
    # Also check pairwise disjointness; if any overlap, add penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            gap = _min_distance_hex_hex(positions[i], angles[i], positions[j], angles[j])
            if gap < 0:
                penalty += (-gap) ** 2
    
    return max(0.0, max_needed) + 100.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: place hexagons in a roughly hexagonal arrangement
    # A unit hexagon has width 2 (vertex to vertex) and height sqrt(3) (face to face)
    # For a hexagonal packing of 11 hexagons: center + ring of 6 + partial second ring
    # Let's start with a compact arrangement
    
    # Hexagonal lattice spacing: for face-to-face, center-to-center distance is 2*sqrt(3)/2 = sqrt(3)
    # Actually for regular hexagons of side 1, the distance between centers in a hexagonal packing
    # (sharing edges) is 2 * (sqrt(3)/2) = sqrt(3)
    
    # Let's use a hexagonal arrangement: 1 in center, 6 around, 4 more
    # Center at origin
    # Ring 1: 6 hexagons at distance sqrt(3) in 6 directions (30, 90, 150, 210, 270, 330)
    # Actually for hexagonal packing, the neighbors are at angles 0, 60, 120, 180, 240, 300
    # with distance sqrt(3)
    
    sqrt3 = np.sqrt(3)
    initial_positions = []
    initial_angles = []
    
    # Center
    initial_positions.append([0.0, 0.0])
    initial_angles.append(0.0)
    
    # Ring 1: 6 hexagons
    for k in range(6):
        ang = np.deg2rad(k * 60)
        initial_positions.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang)])
        initial_angles.append(0.0)
    
    # Ring 2: 4 more hexagons (to make 11 total)
    # Place them at the "corners" of the ring-1 hexagons
    for k in range(4):
        ang = np.deg2rad(k * 90 + 45)  # spread them out
        initial_positions.append([2.5 * np.cos(ang), 2.5 * np.sin(ang)])
        initial_angles.append(0.0)
    
    initial_params = np.array([v for pos, ang in zip(initial_positions, initial_angles) 
                               for v in [pos[0], pos[1], ang]])
    
    best_side = 40.0
    best_params = initial_params.copy()
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    
    # Phase 1: Differential Evolution for global search
    elapsed = time.time() - start_time
    if elapsed < deadline - 60.0:
        bounds = [(-5, 5), (-5, 5), (-180, 180)] * N
        try:
            de_result = differential_evolution(
                _objective, bounds, args=(outer_center, outer_angle),
                maxiter=50, popsize=15, seed=42, tol=1e-6,
                mutation=(0.5, 1.0), recombination=0.7,
                init='sobol', polish=False,
                callback=None, update_interval=10, disp=False
            )
            if de_result.fun < best_side:
                best_side = de_result.fun
                best_params = de_result.x
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 30.0:
        # Phase 2: Local refinement with Nelder-Mead
        try:
            nm_result = minimize(
                _objective, best_params, args=(outer_center, outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if nm_result.fun < best_side:
                best_side = nm_result.fun
                best_params = nm_result.x
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 15.0:
        # Phase 3: Try different outer angles
        for oa in [0, 30, 60, 90, 120, 150]:
            if time.time() - start_time > deadline - 10.0:
                break
            oc = np.array([0.0, 0.0])
            try:
                nm_result = minimize(
                    _objective, best_params, args=(oc, float(oa)),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if nm_result.fun < best_side:
                    best_side = nm_result.fun
                    best_params = nm_result.x
                    outer_angle = float(oa)
            except Exception:
                pass
    
    # Extract final solution
    final_positions = np.array([[best_params[3*i], best_params[3*i+1]] for i in range(N)])
    final_angles = np.array([best_params[3*i+2] for i in range(N)])
    
    # Compute the actual outer side length
    theta_o = np.deg2rad(outer_angle)
    R_o = np.array([[np.cos(theta_o), -np.sin(theta_o)],
                    [np.sin(theta_o), np.cos(theta_o)]])
    outer_normals = (R_o @ _HEX_NORMALS.T).T
    inradius_factor = np.sqrt(3) / 2.0
    
    max_needed = 0.0
    for nk in outer_normals:
        max_support = -np.inf
        for i in range(N):
            theta_i = np.deg2rad(final_angles[i])
            R_i = np.array([[np.cos(theta_i), -np.sin(theta_i)],
                            [np.sin(theta_i), np.cos(theta_i)]])
            verts_i = (R_i @ _HEX_VERTS.T).T
            sup = np.max(verts_i @ nk) + final_positions[i][0] * nk[0] + final_positions[i][1] * nk[1]
            if sup > max_support:
                max_support = sup
        needed = (max_support - outer_center[0] * nk[0] - outer_center[1] * nk[1]) / inradius_factor
        if needed > max_needed:
            max_needed = needed
    
    outer_side = max(0.0, max_needed)
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([float(final_positions[i][0]), float(final_positions[i][1]), float(final_angles[i])])
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
