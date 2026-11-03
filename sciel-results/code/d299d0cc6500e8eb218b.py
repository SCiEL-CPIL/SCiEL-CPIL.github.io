import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Unit hexagon vertices at angle 0 (flat-top)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    hex_verts = np.array([[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)])
    # Reorder to be CCW starting from (1,0)
    # Actually cos(0)=1, sin(0)=0 -> (1,0)
    # cos(60)=0.5, sin(60)=sqrt(3)/2 -> (0.5, 0.866)
    # This is CCW. Good.

    def support_hex(v, theta):
        """Support function of unit hexagon in direction theta (normal vector)."""
        # Support function h(theta) = max_i (v_i . n) where n = (cos theta, sin theta)
        n = np.array([math.cos(theta), math.sin(theta)])
        return np.max(hex_verts @ n)

    def support_composite(pts, theta, delta):
        """
        Support function of the composite body (Minkowski sum of point set and hexagon).
        pts: array of shape (n, 2) - centers of inner hexagons
        theta: direction angle for support
        delta: rotation of the composite body (inner hexagons)
        The outer container has normals at angles beta + k*pi/3.
        We want to minimize the outer side length S.
        
        The support of the composite in direction u (unit vector) is:
        h_comp(u) = max_i (center_i . u) + h_hex(u - delta)
        where h_hex is the support function of the unit hexagon.
        
        Wait, the inner hexagons are rotated by delta. So the support of an inner hexagon
        centered at c_i, rotated by delta, in direction u is:
        c_i . u + h_hex(u - delta)
        
        The composite support in direction u is:
        max_i (c_i . u) + h_hex(u - delta)
        
        The minimal enclosing regular hexagon with side S, centered at origin (wlog), 
        rotated by beta, has support function:
        h_outer(u) = S * APOTHEM / cos(0) ... 
        Actually, a regular hexagon with side S has apothem a = S * sqrt(3)/2.
        Its support function in direction u is a / cos(phi) where phi is the angle 
        between u and the nearest normal.
        The normals are at angles beta + k*pi/3.
        For a given direction u (angle theta), the support is a / cos(theta - beta_k)
        where beta_k is the closest normal angle to theta.
        
        Actually, it's simpler: the hexagon is the intersection of 6 half-planes.
        The constraint is that for all 6 normal directions n_k (at angles beta + k*pi/3),
        h_comp(n_k) <= a = S * APOTHEM.
        
        So S = max_k (h_comp(n_k) / APOTHEM).
        
        We want to minimize S over beta (and the point set configuration).
        """
        # Compute max_i (c_i . u) for each of the 6 normal directions
        # But we need h_comp(n_k) = max_i (c_i . n_k) + h_hex(n_k - delta)
        # The term h_hex(n_k - delta) is the same for all i, so:
        # h_comp(n_k) = max_i (c_i . n_k) + h_hex(n_k - delta)
        
        # Let's compute for each normal direction
        max_projs = []
        for k in range(6):
            ang = beta + k * math.pi / 3
            n = np.array([math.cos(ang), math.sin(ang)])
            # max_i (c_i . n)
            projs = pts @ n
            max_proj = np.max(projs)
            # h_hex(n - delta)
            h_hex_val = support_hex(n, ang - delta)
            max_projs.append(max_proj + h_hex_val)
        
        # S = max_k (h_comp(n_k) / APOTHEM)
        S = max(max_projs) / APOTHEM
        return S

    # But we can also rotate the point set. Let's fix the point set and optimize beta and delta.
    # Actually, the point set can be translated. Let's center it at origin.
    # The relative positions matter.
    
    # Candidate point sets in axial coordinates
    candidates = []

    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)

    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    axial_28 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)

    axial_29 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1), (0, 2)
    ]
    candidates.append(axial_30)

    # New candidates
    axial_31 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (1, 1)
    ]
    candidates.append(axial_32)

    axial_33 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_33)

    axial_34 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_34)

    axial_35 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_35)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def optimize_config(pts):
        """Optimize beta and delta for a given point set."""
        # Center the points
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        pts_centered = pts - np.array([cx, cy])
        
        best_S = float('inf')
        best_beta = 0.0
        best_delta = 0.0
        
        # Grid search over beta and delta
        for i in range(37):
            beta = i * (math.pi / 3) / 36
            for j in range(37):
                delta = j * (math.pi / 3) / 36
                S = support_composite(pts_centered, beta, delta)
                if S < best_S:
                    best_S = S
                    best_beta = beta
                    best_delta = delta
        
        # Refine with minimize
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    beta, delta = params
                    return support_composite(pts_centered, beta, delta)
                
                # Try multiple starting points
                for start in [(best_beta, best_delta), (0.0, 0.0), (math.pi/6, math.pi/6)]:
                    res = minimize(objective, start, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_beta = res.x[0]
                        best_delta = res.x[1]
        except Exception:
            pass
        
        return best_S, best_beta, best_delta, (cx, cy)

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_delta = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, delta, center = optimize_config(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_delta = delta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        best_global_delta = 0.0
        best_global_S = support_composite(best_global_pts, 0.0, 0.0)

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    inner_angle_deg = math.degrees(best_global_delta)

    # Scale points slightly to ensure no overlap
    SCALE = 1.0 + 1e-10
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    # Recalculate S
    pts_centered_scaled = pts_scaled - np.array([cx, cy])
    S_final = support_composite(pts_centered_scaled, best_global_beta, best_global_delta)

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(inner_angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)

    return inner, center, side, rotation
