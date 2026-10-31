"""Stronger solver: rigid-chain packing of 11 unit hexagons via scipy differential_evolution."""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 11
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

# Unit regular hexagon vertices (pointy-top), side length 1
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees
_HEX_VERTS = np.array([[np.cos(np.pi/3*k), np.sin(np.pi/3*k)] for k in range(6)])

# The 6 outward normals of a regular hexagon (pointy-top orientation)
# For a hexagon centered at origin with pointy top, the faces are between vertices
# Normal to face between vertex k and k+1:
_HEX_NORMALS = np.array([[-np.sin(np.pi/3*k + np.pi/6), np.cos(np.pi/3*k + np.pi/6)] for k in range(6)])
# Actually let me compute properly. For a regular hexagon, the 6 outward normals
# point to the midpoints of the 6 edges.
# Vertex k is at angle pi/3*k from center.
# Edge midpoint between vertex k and k+1 is at angle pi/3*(k+0.5)
# So normal direction for edge k is at angle pi/3*(k+0.5)
_HEX_NORMALS = np.array([[np.cos(np.pi/3*(k+0.5)), np.sin(np.pi/3*(k+0.5))] for k in range(6)])


def _hex_transform(center, angle_deg):
    """Return rotation matrix and translation for a unit hexagon."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    return R, np.array(center)


def _get_transformed_vertices(center, angle_deg):
    """Get the 6 vertices of a unit hexagon at given center and rotation."""
    R, t = _hex_transform(center, angle_deg)
    return R @ _HEX_VERTS.T + t[:, np.newaxis]  # shape (2, 6)


def _check_overlap(v1, v2, eps=1e-9):
    """Check if two convex hexagons overlap using SAT. v1, v2 are (2,6) arrays."""
    for verts in (v1, v2):
        for i in range(6):
            p1 = verts[:, i]
            p2 = verts[:, (i+1) % 6]
            # Edge vector
            edge = p2 - p1
            # Normal (perpendicular)
            n = np.array([-edge[1], edge[0]])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            # Project both polygons onto this normal
            proj1 = n @ v1
            proj2 = n @ v2
            if proj1.max() < proj2.min() + eps or proj2.max() < proj1.min() + eps:
                return False  # Separating axis found
    return True  # No separating axis, they overlap


def _compute_min_side(params):
    """
    Given parameters [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10],
    compute the minimum outer hexagon side length needed.
    Returns (side_length, valid_flag) where valid_flag is False if any overlap.
    """
    # Check for overlaps
    verts_list = []
    for i in range(N):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        v = _get_transformed_vertices((x, y), a)
        verts_list.append(v)
    
    for i in range(N):
        for j in range(i+1, N):
            if _check_overlap(verts_list[i], verts_list[j]):
                return (1e10, False)
    
    # Compute the required outer hexagon side length
    # The outer hexagon is regular, centered at (cx, cy), with side length S, rotation theta.
    # For a point p to be inside the outer hexagon, for each of the 6 normals n_k:
    #   n_k . (p - center) <= S * cos(pi/6) = S * sqrt(3)/2
    # (The inradius of a regular hexagon with side S is S * sqrt(3)/2)
    
    # We need to find center (cx, cy) and rotation theta that minimize S
    # such that for all points p in the union and all normals n_k:
    #   n_k(theta) . (p - (cx,cy)) <= S * sqrt(3)/2
    
    # This is equivalent to: for each normal direction, the max projection of all points
    # minus the center projection must be <= S * sqrt(3)/2
    
    # For a given theta, the optimal center is the midpoint of the bounding box in each normal direction.
    # S = max over all normals k of (max_p n_k . p - min_p n_k . p) / (2 * sqrt(3)/2)
    #   = max over k of (range_k) / sqrt(3)
    
    # All vertices of all hexagons
    all_verts = np.hstack(verts_list)  # (2, 6*N)
    
    best_S = 1e10
    # Search over theta
    for theta_deg in np.linspace(0, 60, 121):  # 0 to 60 degrees (hexagon symmetry)
        theta = np.radians(theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        # Rotated normals
        R = np.array([[c, -s], [s, c]])
        normals = R @ _HEX_NORMALS  # (6, 2)
        
        # Project all vertices onto each normal
        # proj shape: (6, 6*N)
        proj = normals @ all_verts
        ranges = proj.max(axis=1) - proj.min(axis=1)
        S_candidate = ranges.max() / np.sqrt(3)
        if S_candidate < best_S:
            best_S = S_candidate
    
    return (best_S, True)


def construct():
    global START_TIME
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT:
        # Return a trivial valid solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Initial guess: hexagons in a row
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = 4.0 * (i - N//2)
        x0[3*i+1] = 0.0
        x0[3*i+2] = 0.0
    
    # Bounds for differential evolution
    bounds = []
    for i in range(N):
        bounds.append((-20, 20))  # x
        bounds.append((-20, 20))  # y
        bounds.append((-180, 180))  # angle
    
    # Objective function
    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT:
            return 1e10
        S, valid = _compute_min_side(params)
        return S
    
    # Run differential evolution with time check
    best_result = None
    best_score = 1e10
    
    def callback(xk, convergence):
        nonlocal best_result, best_score
        if time.time() - START_TIME > TIME_LIMIT:
            return True  # Stop
        S, valid = _compute_min_side(xk)
        if valid and S < best_score:
            best_score = S
            best_result = xk.copy()
        return False
    
    try:
        result = differential_evolution(
            objective, bounds,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            callback=callback,
            seed=42,
            polish=True,
            workers=1
        )
    except Exception as e:
        pass
    
    if best_result is not None:
        params = best_result
    else:
        params = x0
        S, valid = _compute_min_side(params)
        if not valid:
            # Fallback to trivial
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N//2)]
            return inner, (0.0, 0.0), 40.0, 0.0
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([float(params[3*i]), float(params[3*i+1]), float(params[3*i+2])])
    
    # Compute final side length and find optimal center/rotation
    verts_list = []
    for i in range(N):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        v = _get_transformed_vertices((x, y), a)
        verts_list.append(v)
    
    all_verts = np.hstack(verts_list)
    
    best_S = 1e10
    best_center = (0.0, 0.0)
    best_theta = 0.0
    
    for theta_deg in np.linspace(0, 60, 361):
        theta = np.radians(theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        normals = R @ _HEX_NORMALS
        
        proj = normals @ all_verts
        ranges = proj.max(axis=1) - proj.min(axis=1)
        S_candidate = ranges.max() / np.sqrt(3)
        
        if S_candidate < best_S:
            best_S = S_candidate
            # Optimal center: for each normal, center should be at midpoint
            # Use the normal that gives the max range
            k_max = np.argmax(ranges)
            n_k = normals[k_max]
            proj_k = n_k @ all_verts
            center_proj = (proj_k.max() + proj_k.min()) / 2
            # Center in 2D: we need to find (cx, cy) such that n_k . (cx,cy) = center_proj
            # This is underdetermined; use all normals to find best center
            # Actually, for a hexagon, the center is uniquely determined by the midpoints
            # of the 3 pairs of opposite faces
            # For normal k and normal k+3 (opposite):
            #   n_k . c = (max_k + min_k)/2
            #   n_{k+3} . c = (max_{k+3} + min_{k+3})/2
            # Since n_{k+3} = -n_k, these are consistent only if the ranges are equal
            # Let's just use the average of all 6 midpoint projections
            center_projs = []
            for k in range(6):
                proj_kk = normals[k] @ all_verts
                cp = (proj_kk.max() + proj_kk.min()) / 2
                center_projs.append(cp)
            
            # Solve for center: we have 6 equations n_k . c = cp_k
            # Use least squares
            A = normals.T  # (2, 6)
            b = np.array(center_projs)
            # c = A^T (A A^T)^{-1} b  (since A is 2x6, use pseudoinverse)
            c_center = np.linalg.lstsq(A, b, rcond=None)[0]
            best_center = (float(c_center[0]), float(c_center[1]))
            best_theta = theta_deg
    
    return inner, best_center, float(best_S), float(best_theta)
