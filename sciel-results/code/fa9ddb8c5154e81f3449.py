import math
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates) for 11 unit hexagons
    # Each set represents the centers of the 11 hexagons in axial coordinates
    # The composite body is the Minkowski sum of these points with a unit hexagon

    candidates = []

    # Candidate 1: Central hexagon + 6 neighbors + 4 more (flower-like)
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_1)

    # Candidate 2: 3x3 grid minus 2 corners + 2 extras
    axial_2 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_2)

    # Candidate 3: 4x3 grid
    axial_3 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_3)

    # Candidate 4: Stacked rows
    axial_4 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_4)

    # Candidate 5: Another flower variant
    axial_5 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_5)

    # Candidate 6: 5+6 strip
    axial_6 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_6)

    # Candidate 7: 4+3+4 pattern
    axial_7 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_7)

    # Candidate 8: 3+5+3 pattern
    axial_8 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_8)

    # Candidate 9: 2+3+3+3 pattern
    axial_9 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_9)

    # Candidate 10: 3+4+4 pattern
    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_10)

    # Candidate 11: 5+4+2 pattern
    axial_11 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2)
    ]
    candidates.append(axial_11)

    # Candidate 12: 3+3+5 pattern
    axial_12 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)
    ]
    candidates.append(axial_12)

    # Candidate 13: 4+4+3 pattern
    axial_13 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_13)

    # Candidate 14: Compact 5x3 minus corners
    axial_14 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_14)

    # Candidate 15: 6+5 pattern
    axial_15 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1)
    ]
    candidates.append(axial_15)

    def axial_to_cartesian(axial_list):
        """Convert axial coordinates to Cartesian coordinates."""
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def support_function(pts, angle):
        """
        Compute the support function of the Minkowski sum of pts with a unit hexagon
        at a given angle. The support function h(theta) = max over all points p in
        the composite body of p . (cos(theta), sin(theta)).
        
        For the Minkowski sum of point set P with a unit hexagon H,
        h_{P+H}(theta) = max_{p in P} (p . u) + h_H(theta)
        where u = (cos(theta), sin(theta)) and h_H is the support function of the unit hexagon.
        
        The support function of a unit regular hexagon (side 1) centered at origin:
        h_H(theta) = APOTHEM * max over 6 face normals of (u . n_k)
        where n_k are the outward unit normals of the hexagon faces.
        
        Actually, for a convex body K, h_K(theta) = max_{x in K} x . u(theta).
        For a regular hexagon with apothem a = sqrt(3)/2, the support function is:
        h_H(theta) = a / cos((theta mod pi/3) - pi/6) for theta in [0, 2*pi)
        More precisely, if we align the hexagon with one face normal along the x-axis,
        h_H(theta) = APOTHEM / cos(theta_mod) where theta_mod is the angle modulo pi/3,
        adjusted to be in [-pi/6, pi/6].
        """
        u = np.array([math.cos(angle), math.sin(angle)])
        
        # Max of p . u over all points
        max_dot = np.max(pts @ u)
        
        # Support function of unit hexagon at angle 'angle'
        # The unit hexagon has 6 faces. The support function is periodic with period pi/3.
        # Reduce angle to [0, pi/3)
        theta_mod = angle % (math.pi / 3)
        # The angle relative to the nearest face normal
        delta = theta_mod - math.pi / 6
        delta = abs(delta)
        # If delta > pi/6, wrap around
        if delta > math.pi / 6:
            delta = math.pi / 3 - delta
        
        h_hex = APOTHEM / math.cos(delta)
        
        return max_dot + h_hex

    def enclosing_hexagon_side(pts, beta):
        """
        Compute the side length of the minimal enclosing regular hexagon with orientation beta.
        
        The enclosing hexagon has 6 faces. For each face with outward normal n_k,
        the face must contain all points of the composite body.
        
        The distance from the center of the enclosing hexagon to each face is the apothem A.
        For the composite body centered at some point c, we need:
        max_{p in composite} (p - c) . n_k <= A for all k
        
        So A >= max over all 6 normals of max_{p in composite} (p - c) . n_k
        
        But we can choose c to minimize A. The optimal c for a given orientation is
        the center of the minimal enclosing hexagon, which can be found by linear programming,
        but for the support function approach, we note that:
        
        The width of the composite body in direction n_k is:
        w_k = max_{p in composite} (p . n_k) - min_{p in composite} (p . n_k)
        
        For a regular hexagon with apothem A, the width in each face-normal direction is 2A.
        So we need 2A >= w_k for all k, i.e., A >= max_k (w_k / 2).
        
        But this is only valid if the hexagon is centered at the midpoint of the widths.
        The side length S = A / APOTHEM (since for a unit hexagon, apothem = APOTHEM and side = 1).
        So S = max_k (w_k / 2) / APOTHEM.
        
        Wait, let me reconsider. The composite body is the Minkowski sum of the point set P
        (centers of the 11 hexagons) with a unit hexagon H. So the composite body C = P + H.
        
        The width of C in direction n is:
        w_C(n) = max_{p in C} (p . n) - min_{p in C} (p . n)
               = max_{p in P, h in H} ((p + h) . n) - min_{p in P, h in H} ((p + h) . n)
               = (max_{p in P} (p . n) + max_{h in H} (h . n)) - (min_{p in P} (p . n) + min_{h in H} (h . n))
               = (max_{p in P} (p . n) - min_{p in P} (p . n)) + (max_{h in H} (h . n) - min_{h in H} (h . n))
               = w_P(n) + w_H(n)
        
        For a unit hexagon H, w_H(n) = 2 * APOTHEM / cos(delta) where delta is the angle
        between n and the nearest face normal of H, reduced to [-pi/6, pi/6].
        
        Actually, for a regular hexagon with side 1, the width in any direction is:
        w_H(n) = 2 * APOTHEM / cos(delta) where delta is in [-pi/6, pi/6].
        
        But wait, the width of a convex body in direction n is h(n) + h(-n).
        For the hexagon, h_H(n) = APOTHEM / cos(delta) as computed above.
        So w_H(n) = h_H(n) + h_H(n + pi) = 2 * APOTHEM / cos(delta) since h is even for symmetric bodies.
        
        So w_C(n) = w_P(n) + 2 * APOTHEM / cos(delta(n))
        
        where delta(n) is the angular distance from n to the nearest face normal of the unit hexagon H.
        
        Now, for the enclosing regular hexagon with orientation beta (rotation angle),
        its 6 face normals are at angles beta + pi/6 + k*pi/3 for k = 0, ..., 5.
        
        The apothem A of the enclosing hexagon must satisfy:
        A >= w_C(n_k) / 2 for each face normal n_k
        
        So A_min(beta) = max_k (w_C(n_k) / 2)
        
        And the side length S = A_min(beta) / APOTHEM.
        
        We want to minimize S over beta, which is equivalent to minimizing A_min(beta) over beta.
        Since A_min(beta) is periodic with period pi/3, we only need to search beta in [0, pi/3).
        """
        # The 6 face normals of the enclosing hexagon with orientation beta
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Compute widths of P in each normal direction
        max_A = 0.0
        for nx, ny in normals:
            # Width of P in direction (nx, ny)
            dots = pts @ np.array([nx, ny])
            w_P = np.max(dots) - np.min(dots)
            
            # Width of unit hexagon in this direction
            # The angle of the normal relative to the hexagon's face normals
            # The unit hexagon H has face normals at angles pi/6 + k*pi/3 (if unrotated)
            # But H is the fixed unit hexagon (the shape of each inner hexagon), not rotated.
            # So we need w_H(n) for the fixed unit hexagon.
            
            # For the fixed unit hexagon with face normals at pi/6 + k*pi/3,
            # the width in direction n at angle ang is:
            # Find the nearest face normal angle to ang
            # The face normals are at pi/6 + k*pi/3 for k=0,...,5
            # Reduce ang modulo pi/3
            ang_mod = ang % (math.pi / 3)
            delta = ang_mod - math.pi / 6
            delta = abs(delta)
            if delta > math.pi / 6:
                delta = math.pi / 3 - delta
            
            w_H = 2.0 * APOTHEM / math.cos(delta)
            
            w_C = w_P + w_H
            A_k = w_C / 2.0
            if A_k > max_A:
                max_A = A_k
        
        S = max_A / APOTHEM
        return S

    def optimize_orientation(pts):
        """Find the optimal orientation beta that minimizes the enclosing hexagon side length."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0
        
        # Coarse grid search
        best_S = float('inf')
        best_beta = 0.0
        n_grid = 120
        for i in range(n_grid):
            beta = i * (math.pi / 3) / n_grid
            S = enclosing_hexagon_side(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
        
        # Fine grid search around the best
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            center = best_beta
            span = math.pi / 3 / n_grid
            for i in range(200):
                beta = center - span + i * (2 * span) / 200
                beta = beta % (math.pi / 3)
                S = enclosing_hexagon_side(pts, beta)
                if S < best_S:
                    best_S = S
                    best_beta = beta
        
        # Use minimize_scalar for even finer optimization
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            try:
                res = minimize_scalar(lambda b: enclosing_hexagon_side(pts, b % (math.pi / 3)),
                                     bounds=(0, math.pi / 3), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x % (math.pi / 3)
            except Exception:
                pass
        
        return best_S, best_beta

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta = optimize_orientation(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta

    if best_global_pts is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_beta = 0.0
        best_global_S = enclosing_hexagon_side(best_global_pts, best_global_beta)

    # Compute the center of the composite body for output
    # The center of the enclosing hexagon should be at the centroid of the composite body
    # For the Minkowski sum P + H, the centroid is centroid(P) + centroid(H) = centroid(P) since H is centered at origin
    cx = float(np.mean(best_global_pts[:, 0]))
    cy = float(np.mean(best_global_pts[:, 1]))
    
    angle_deg = math.degrees(best_global_beta)
    
    # Small safety margin to ensure no overlap due to floating point
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    # Recompute S with the scaled points
    S_final = enclosing_hexagon_side(pts_scaled, best_global_beta)
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
