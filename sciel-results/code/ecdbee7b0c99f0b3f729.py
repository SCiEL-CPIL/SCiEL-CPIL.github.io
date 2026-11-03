import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def convex_hull_2d(points):
        """Compute convex hull of 2D points (Andrew's monotone chain)."""
        points = np.array(points)
        points = points[np.lexsort((points[:, 1], points[:, 0]))]
        if len(points) <= 2:
            return points
        
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        
        return np.vstack([lower[:-1], upper[:-1]])

    def compute_convex_hull_with_hexagon_offsets(pts, hex_vertices):
        """
        Compute the convex hull of the Minkowski sum of the point set with the unit hexagon.
        This is equivalent to the convex hull of all points offset by each hexagon vertex.
        """
        all_points = []
        for p in pts:
            for v in hex_vertices:
                all_points.append(p + v)
        all_points = np.array(all_points)
        return convex_hull_2d(all_points)

    def support_function(hull, direction):
        """Compute support function of convex hull in a given direction."""
        return np.max(hull @ direction)

    def min_enclosing_hexagon_side(hull, beta):
        """
        Compute the minimum side length of a regular hexagon (with rotation beta)
        that encloses the convex hull.
        
        The outer hexagon has apothem D such that for each of the 6 face normals,
        the support function in that direction (and its negative) is <= D.
        Side length S = (D + APOTHEM) / APOTHEM where D is the max projection.
        """
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # For each normal, we need max(hull @ normal) <= D and max(hull @ (-normal)) <= D
        # i.e., max(|hull @ normal|) <= D
        projs = hull @ normals.T  # (n_hull, 6)
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def min_enclosing_hexagon_optimal(hull, beta):
        """
        For a given orientation beta, find the optimal center and D for the enclosing hexagon.
        The center should be at the midpoint of the min and max projections for each axis.
        We use the center that minimizes the maximum projection.
        
        For a regular hexagon with given orientation, the optimal center is the
        centroid of the "critical" points. We can compute this via LP or by
        finding the center that equalizes the max positive and negative projections.
        
        Actually, for a fixed orientation, the minimum enclosing hexagon center is
        determined by the LP: minimize D subject to -D <= p_i @ n_k - c @ n_k <= D for all i,k.
        This is equivalent to: c @ n_k is between (min_i p_i @ n_k + max_i p_i @ n_k)/2... 
        No, that's not right. Let me think again.
        
        For each direction n_k, we need:
          min_i (p_i - c) @ n_k >= -D
          max_i (p_i - c) @ n_k <= D
        i.e., c @ n_k >= max_i p_i @ n_k - D
        and c @ n_k <= min_i p_i @ n_k + D
        
        So for each k: (max_i p_i @ n_k - D) <= c @ n_k <= (min_i p_i @ n_k + D)
        
        This is a 2D LP in (c_x, c_y, D). But we can also just compute it directly.
        """
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        n_pts = len(hull)
        n_constraints = 12 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)
        
        for k in range(6):
            nx, ny = normals[k]
            projs = hull[:, 0] * nx + hull[:, 1] * ny
            start = 2 * k * n_pts
            A_ub[start:start+n_pts, 0] = -nx
            A_ub[start:start+n_pts, 1] = -ny
            A_ub[start:start+n_pts, 2] = -1.0
            b_ub[start:start+n_pts] = -projs
            A_ub[start+n_pts:start+2*n_pts, 0] = nx
            A_ub[start+n_pts:start+2*n_pts, 1] = ny
            A_ub[start+n_pts:start+2*n_pts, 2] = -1.0
            b_ub[start+n_pts:start+2*n_pts] = projs
        
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    # Unit hexagon vertices (side length 1, centered at origin)
    hex_angles = np.pi / 6 + np.arange(6) * np.pi / 3
    hex_vertices = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])

    # Generate candidate configurations
    candidates = []
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(1,3),(2,3),(0,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)])

    # Evaluate all candidates using Minkowski sum approach
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts = axial_to_cartesian(axial)
        
        # Compute the Minkowski sum convex hull
        hull = compute_convex_hull_with_hexagon_offsets(pts, hex_vertices)
        
        # Stage 1: Grid search over beta in [0, pi/6] using support function
        n_grid = 97
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        for i in range(n_grid):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_grid - 1)
            S, center, D = min_enclosing_hexagon_optimal(hull, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        if best_center is None:
            continue
        
        # Stage 2: Scalar refinement of beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: min_enclosing_hexagon_optimal(hull, b)[0],
                                     bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 500, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = min_enclosing_hexagon_optimal(hull, best_beta)
        except Exception:
            pass
        
        # Stage 3: Fine grid around best_beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                n_fine = 51
                for i in range(n_fine):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    beta_fine = best_beta - 0.02 + i * 0.04 / (n_fine - 1)
                    if beta_fine < 0 or beta_fine > math.pi / 6:
                        continue
                    S_fine, center_fine, D_fine = min_enclosing_hexagon_optimal(hull, beta_fine)
                    if S_fine < best_S:
                        best_S = S_fine
                        best_beta = beta_fine
                        best_center = center_fine
        except Exception:
            pass
        
        if best_S < best_global_S:
            best_global_S = best_S
            best_global_pts = pts
            best_global_beta = best_beta
            best_global_center = best_center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback
        best_global_pts = axial_to_cartesian(candidates[0])
        hull = compute_convex_hull_with_hexagon_offsets(best_global_pts, hex_vertices)
        best_global_beta = 0.0
        S, center, D = min_enclosing_hexagon_optimal(hull, best_global_beta)
        best_global_S = S
        best_global_center = center

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Final: compute side length with tiny safety scale
    SCALE = 1.0 + 1e-12
    pts_scaled = best_global_pts.copy()
    for i in range(len(pts_scaled)):
        pts_scaled[i, 0] = cx + (pts_scaled[i, 0] - cx) * SCALE
        pts_scaled[i, 1] = cy + (pts_scaled[i, 1] - cy) * SCALE

    # Recompute hull with scaled points
    hull_scaled = compute_convex_hull_with_hexagon_offsets(pts_scaled, hex_vertices)
    
    # Compute final S
    angles = best_global_beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    projs = hull_scaled @ normals.T
    D_scaled = np.max(np.abs(projs))
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
