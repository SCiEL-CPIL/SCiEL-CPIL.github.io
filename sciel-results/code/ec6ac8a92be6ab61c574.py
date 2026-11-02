import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate diverse candidate topologies
    candidates = []

    # Ring + center (7 hexes)
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2),(1,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(1,-2),(-1,3),(0,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1),(0,2)])

    # 3x3 grid variants
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,1),(-1,1)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3),(3,3)])

    # 4x3 grid
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])

    # 5x2
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(1,1),(2,1),(3,1),(4,1),(5,1)])

    # L-shapes
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1),(1,2),(2,2)])

    # T-shapes
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])

    # Diamond / hexagonal
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,1),(-2,2),(0,2)])
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])

    # More creative
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(4,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(4,1),(0,2),(1,2),(2,2),(3,2),(4,2),(1,3),(2,3)])

    # Compact clusters
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(0,2),(-1,2),(2,-1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1),(0,2),(0,-2)])

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def compute_S(pts, beta, cx, cy):
        """Compute outer hexagon side length given inner points, rotation, and center."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        centered = pts - np.array([cx, cy])
        projs = np.abs(centered @ normals.T)
        max_proj = np.max(projs)
        S = (max_proj + APOTHEM) / APOTHEM
        return S

    def optimize_shape(pts):
        """Optimize center and rotation for a given set of points."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = (0.0, 0.0)

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, (0.0, 0.0)

        # Coarse grid search
        n_grid = 36
        for i in range(n_grid):
            beta = i * (math.pi / 6) / (n_grid - 1)
            # For each beta, find optimal center via LP-like approach
            # The optimal center minimizes max projection, which is the center of the minimal enclosing hexagon
            # For a given orientation, this is a linear programming problem
            # We can solve it by noting that the optimal center equalizes the max projections on opposite faces

            angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
            normals = np.column_stack([np.cos(angles), np.sin(angles)])

            # For each pair of opposite normals, the center should be at the midpoint
            # of the projections. We find the center that minimizes the maximum deviation.
            # This is equivalent to finding the center of the minimal enclosing hexagon.

            # Simple approach: iterate center as weighted average, then refine
            # The optimal center for a given orientation is where the max projection is minimized
            # This can be found by solving a small LP or by iterative method

            # Use the fact that at optimum, at least 3 pairs of opposite faces are tight
            # For now, use a simple heuristic: centroid, then refine
            cx0 = float(np.mean(pts[:, 0]))
            cy0 = float(np.mean(pts[:, 1]))

            # Refine with a few iterations
            cx, cy = cx0, cy0
            for _ in range(20):
                angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                normals = np.column_stack([np.cos(angles), np.sin(angles)])
                centered = pts - np.array([cx, cy])
                projs = centered @ normals.T  # shape (n_pts, 6)

                # For each direction k, the max projection is projs[:, k].max()
                # The min projection is projs[:, k].min()
                # The "width" in direction k is max - min
                # The center should be adjusted to equalize the extreme projections
                max_projs = projs.max(axis=0)
                min_projs = projs.min(axis=0)

                # The constraint is: for each k, -d_k <= proj_k <= d_k where d_k is the half-width
                # At optimum, we want to minimize max_k max(|max_projs[k]|, |min_projs[k]|)
                # This is a linear programming problem in (cx, cy, d)

                # Simple gradient step: move center toward the direction that has the largest imbalance
                # For each pair of opposite faces (k, k+3), the imbalance is (max_projs[k] + min_projs[k+3])/2
                # Actually, opposite faces: face k has normal n_k, face k+3 has normal -n_k
                # The projection on face k is n_k . (p - c), on face k+3 is -n_k . (p - c)
                # So the constraint is: n_k . (p - c) <= d and -n_k . (p - c) <= d
                # i.e., |n_k . (p - c)| <= d

                # The optimal d is max_k max_p |n_k . (p - c)|
                # To minimize d, we adjust c

                # Compute the gradient of d with respect to c
                # d = max_k max_p |n_k . (p - c)|
                # The active constraints are those where |n_k . (p - c)| = d

                # Simple approach: use subgradient
                # The subgradient of d w.r.t. c is the average of the normals of the active constraints
                active_normals = []
                for k in range(6):
                    n_k = normals[k]
                    proj_k = centered @ n_k
                    max_idx = np.argmax(np.abs(proj_k))
                    if abs(proj_k[max_idx]) > d - 1e-10:
                        if proj_k[max_idx] > 0:
                            active_normals.append(n_k)
                        else:
                            active_normals.append(-n_k)

                if active_normals:
                    avg_normal = np.mean(active_normals, axis=0)
                    cx += 0.1 * avg_normal[0]
                    cy += 0.1 * avg_normal[1]

            S = compute_S(pts, beta, cx, cy)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = (cx, cy)

        # Refine with scipy minimize
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def objective(params):
                cx, cy, beta = params
                beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                return compute_S(pts, beta, cx, cy)

            # Multiple starting points
            starts = [
                [best_center[0], best_center[1], best_beta],
                [best_center[0], best_center[1], best_beta + 0.05],
                [best_center[0], best_center[1], best_beta - 0.05],
                [float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])), 0.0],
                [float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])), math.pi/12],
            ]

            for init in starts:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                try:
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                except Exception:
                    pass

        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Sort by bounding box area (smaller is likely better)
    def heuristic(axial):
        pts = axial_to_cartesian(axial)
        return (np.max(pts[:, 0]) - np.min(pts[:, 0])) * (np.max(pts[:, 1]) - np.min(pts[:, 1]))

    sorted_candidates = sorted(candidates, key=heuristic)

    for axial in sorted_candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        best_global_S = compute_S(best_global_pts, best_global_beta, best_global_center[0], best_global_center[1])

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Scale down slightly to ensure strict containment
    SCALE = 1.0 - 1e-10
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))

    pts_scaled = np.array(pts_scaled)

    # Recompute S for the scaled points
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
