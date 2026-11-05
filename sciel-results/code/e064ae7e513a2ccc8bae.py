"""Packing 12 unit hexagons in a regular hexagon."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 115.0  # Safety margin before 120s limit

def _hexagon_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    # Unit hexagon vertices (flat-top, side 1)
    base = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    # Scale by s
    verts = base * s
    # Rotate
    R = np.array([[ca, -sa], [sa, ca]])
    verts = (R @ verts.T).T
    # Translate
    verts[:, 0] += cx
    verts[:, 1] += cy
    return verts

def _signed_dist_point_to_polygon(p, poly):
    """
    Return signed distance from point p to polygon poly (CCW).
    Negative if inside, positive if outside.
    """
    n = len(poly)
    min_dist = float('inf')
    inside = True
    for i in range(n):
        v1 = poly[i]
        v2 = poly[(i+1) % n]
        e = v2 - v1
        # Distance to edge segment
        t = np.dot(p - v1, e) / np.dot(e, e)
        t = np.clip(t, 0.0, 1.0)
        closest = v1 + t * e
        d = np.linalg.norm(p - closest)
        min_dist = min(min_dist, d)
        # Check if point is to the right of edge (outside for CCW)
        cross = np.cross(e, p - v1)
        if cross < -1e-10:
            inside = False
    if inside:
        return -min_dist
    else:
        return min_dist

def _polygon_overlap_penalty(v1, v2):
    """
    Compute smooth overlap penalty between two convex polygons v1, v2 (each 6 vertices).
    Returns a scalar penalty that is 0 if disjoint, >0 if overlapping.
    Uses sum of squared negative signed distances of vertices of each to the other.
    """
    pen = 0.0
    for v in v1:
        d = _signed_dist_point_to_polygon(v, v2)
        if d < 0:
            pen += d * d
    for v in v2:
        d = _signed_dist_point_to_polygon(v, v1)
        if d < 0:
            pen += d * d
    return pen

def _bounding_hexagon_side(points, center, angle_deg):
    """
    Given a set of points (vertices of all inner hexagons), find the minimum side length
    of a regular hexagon centered at `center` with rotation `angle_deg` that contains all points.
    """
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    # Local coordinates
    lx = ca * (points[:, 0] - center[0]) + sa * (points[:, 1] - center[1])
    ly = -sa * (points[:, 0] - center[0]) + ca * (points[:, 1] - center[1])
    # For a regular hexagon with flat-top orientation (in local frame), 
    # the constraint for a point (x,y) to be inside a hexagon of side S centered at origin:
    # The hexagon is defined by |x| <= S, |sqrt(3)*y| <= S*(2 - |x|/S) ... 
    # Actually, easier: the hexagon has support function.
    # For a regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300 (pointy-top) or 
    # 30, 90, 150, 210, 270, 330 (flat-top).
    # Let's use the 6 supporting lines. For a hexagon with flat top (vertices at 30,90,...),
    # the normal directions are 0, 60, 120, 180, 240, 300 degrees? 
    # Actually, for a regular hexagon, the width in any direction is S*(2*cos(theta) + sqrt(3)*sin(theta)) ... no.
    
    # Simplest: A regular hexagon of side S centered at origin, rotated by angle a,
    # can be described as the intersection of 6 half-planes.
    # The 6 outward normal directions are at angles: a + k*60 degrees, k=0..5.
    # The distance from center to each edge (apothem) is S*sqrt(3)/2.
    # So the constraint is: for each normal direction n_k, n_k . p <= S*sqrt(3)/2.
    # We need the smallest S such that for all points p, and all k, n_k . (p - center) <= S*sqrt(3)/2.
    # So S = max_{p,k} (n_k . (p - center)) / (sqrt(3)/2).
    
    # Normal directions in local frame (after rotating by -a, the hexagon is unrotated)
    # Wait, we already rotated points into the hexagon's frame. So in local frame, the hexagon is unrotated.
    # For a flat-top hexagon, the edges are horizontal (top/bottom) and slanted.
    # The outward normals are at angles 0, 60, 120, 180, 240, 300? 
    # Let's verify: a flat-top hexagon has a horizontal top edge. The outward normal of the top edge is (0,1), angle 90.
    # So normals are at 90, 150, 210, 270, 330, 30 degrees. i.e., 30 + 60*k.
    
    normal_angles = np.radians(30 + 60 * np.arange(6))
    normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1)  # 6x2
    
    # Project all points onto all normals
    # proj[k, i] = n_k . (p_i)
    proj = points[:, :2] @ normals.T  # shape (n_points, 6)
    
    # We need max over all points and all normals of proj, but we also need to consider 
    # that the point could be on the other side? No, for a convex body containing origin,
    # the support function in direction n is max(p . n). But we also need min? 
    # No, the hexagon is centered, so we need both:
    # For each normal direction n, we need:
    #   max(p . n) <= S*sqrt(3)/2
    #   min(p . n) >= -S*sqrt(3)/2
    # So S >= max(|max(p . n)|, |min(p . n)|) / (sqrt(3)/2)
    
    # Actually, simpler: the constraint is |p . n_k| <= S*sqrt(3)/2 for all k.
    # So S = max_{k, p} |p . n_k| / (sqrt(3)/2)
    
    max_abs = np.max(np.abs(proj))
    S = max_abs / (np.sqrt(3) / 2)
    return S

def _objective(params):
    """
    params: 2D array of shape (12, 3) or flat (36,).
    Each row: (x, y, angle_deg).
    """
    if params.ndim == 1:
        params = params.reshape(N, 3)
    
    # Compute all vertices
    all_verts = []
    for i in range(N):
        v = _hexagon_vertices(params[i, 0], params[i, 1], params[i, 2])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)  # 72 points
    
    # Center of bounding box (use mean of all vertices as approximation, or fixed at 0)
    center = (0.0, 0.0)
    angle_deg = 0.0  # Try to keep outer hexagon fixed orientation for simplicity
    
    S = _bounding_hexagon_side(all_verts, center, angle_deg)
    
    # Overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            penalty += _polygon_overlap_penalty(all_verts[i*6:(i+1)*6], all_verts[j*6:(j+1)*6])
    
    return S + 10.0 * penalty

def construct():
    start_time = time.time()
    
    # Initial guess: honeycomb-like arrangement
    # 12 hexagons in a 4x3 grid? Or a compact cluster.
    # Let's use a simple grid: 4 columns, 3 rows, with appropriate spacing.
    # Side length 1 hexagons: horizontal spacing = 1.5 (for pointy-top) or 1.5 for flat-top?
    # For flat-top: width = 2, height = sqrt(3).
    # Horizontal spacing: 1.5 (since width is 2, but they interlock). Actually for flat-top:
    # dx = 1.5, dy = sqrt(3) for adjacent rows? No.
    # Let's just use a compact 3x4 grid with spacing 1.5 in x and sqrt(3) in y, with alternating rows shifted.
    
    # Actually, let's start with a simple configuration that is definitely valid (non-overlapping)
    # and then optimize.
    
    # 4 rows of 3? Or 3 rows of 4?
    # Let's do 3 rows of 4.
    # Row 0: x = 0, 1.5, 3.0, 4.5; y = 0
    # Row 1: x = 0.75, 2.25, 3.75, 5.25; y = sqrt(3)
    # Row 2: x = 0, 1.5, 3.0, 4.5; y = 2*sqrt(3)
    
    sqrt3 = np.sqrt(3)
    initial = np.zeros((N, 3))
    for i in range(N):
        row = i // 4
        col = i % 4
        x = col * 1.5 + (0.75 if row % 2 == 1 else 0.0)
        y = row * sqrt3
        initial[i, 0] = x
        initial[i, 1] = y
        initial[i, 2] = 0.0
    
    # Center the configuration
    cx = np.mean(initial[:, 0])
    cy = np.mean(initial[:, 1])
    initial[:, 0] -= cx
    initial[:, 1] -= cy
    
    best_params = initial.copy()
    best_score = _objective(initial.flatten())
    
    # Try Nelder-Mead
    try:
        res = minimize(_objective, initial.flatten(), method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        if res.fun < best_score:
            best_params = res.x.reshape(N, 3)
            best_score = res.fun
    except Exception:
        pass
    
    if time.time() > DEADLINE:
        # Return best so far
        inner = [[float(best_params[i, 0]), float(best_params[i, 1]), float(best_params[i, 2])] for i in range(N)]
        # Compute actual bounding hexagon side
        all_verts = []
        for i in range(N):
            v = _hexagon_vertices(best_params[i, 0], best_params[i, 1], best_params[i, 2])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        S = _bounding_hexagon_side(all_verts, (0.0, 0.0), 0.0)
        return inner, (0.0, 0.0), float(S), 0.0
    
    # Try a few random restarts
    for restart in range(3):
        if time.time() > DEADLINE - 10:
            break
        
        # Perturb initial guess
        perturbed = initial.copy()
        perturbed[:, :2] += np.random.normal(0, 0.5, (N, 2))
        perturbed[:, 2] += np.random.normal(0, 30, N)
        
        try:
            res = minimize(_objective, perturbed.flatten(), method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_score:
                best_params = res.x.reshape(N, 3)
                best_score = res.fun
        except Exception:
            pass
    
    inner = [[float(best_params[i, 0]), float(best_params[i, 1]), float(best_params[i, 2])] for i in range(N)]
    
    # Compute actual bounding hexagon side (without penalty)
    all_verts = []
    for i in range(N):
        v = _hexagon_vertices(best_params[i, 0], best_params[i, 1], best_params[i, 2])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    S = _bounding_hexagon_side(all_verts, (0.0, 0.0), 0.0)
    
    return inner, (0.0, 0.0), float(S), 0.0
