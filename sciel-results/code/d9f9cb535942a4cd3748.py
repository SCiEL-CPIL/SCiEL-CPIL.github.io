import math
import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 140.0
SAFETY_MARGIN = 10.0

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

    def compute_S_and_center(pts, beta):
        """Compute exact S and center using LP."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None

        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))

        n_pts = len(pts)
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []

        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy)
            else:
                return float('inf'), None
        except Exception:
            return float('inf'), None

    def fast_S(pts, beta):
        """Fast computation without LP - just compute max projection."""
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))

        # Center points
        center = np.mean(pts, axis=0)
        pts_c = pts - center

        max_proj = 0
        for nx, ny in normals:
            projs = pts_c[:, 0] * nx + pts_c[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))

        S = (max_proj + APOTHEM) / APOTHEM
        return S

    def optimize_beta(pts, n_coarse=360, n_fine=100):
        """Optimize beta for given points."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0

        best_S = float('inf')
        best_beta = 0.0

        # Coarse search
        for i in range(n_coarse + 1):
            beta = i * (math.pi / 6) / n_coarse
            S = fast_S(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta

        # Fine search around best
        range_fine = math.pi / 6 / n_coarse * 2
        for i in range(n_fine + 1):
            beta = best_beta - range_fine + 2 * range_fine * i / n_fine
            if beta < 0:
                beta += math.pi / 6
            elif beta > math.pi / 6:
                beta -= math.pi / 6
            S = fast_S(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta

        # Very fine polish
        range_vf = 0.001
        for i in range(200 + 1):
            beta = best_beta - range_vf + 2 * range_vf * i / 200
            if beta < 0:
                beta += math.pi / 6
            elif beta > math.pi / 6:
                beta -= math.pi / 6
            S = fast_S(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta

        return best_S, best_beta

    # Generate diverse candidate configurations
    candidates = []

    # Pattern 1: Cross/plus shape
    axial_cross = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_cross)

    # Pattern 2: 3x3 block minus corners
    axial_3x3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3x3)

    # Pattern 3: Hexagonal ring + center
    axial_ring = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_ring)

    # Pattern 4: Linear chain with branches
    axial_chain = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_chain)

    # Pattern 5: Compact cluster
    axial_compact = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_compact)

    # Pattern 6: Diagonal arrangement
    axial_diag = [
        (0, 0), (1, 1), (2, 2), (3, 3),
        (0, 1), (1, 2), (2, 3),
        (0, 2), (1, 3),
        (0, 3)
    ]
    candidates.append(axial_diag)

    # Pattern 7: Star pattern
    axial_star = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3)
    ]
    candidates.append(axial_star)

    # Pattern 8: Two rows offset
    axial_two_rows = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_two_rows)

    # Pattern 9: Triangular
    axial_tri = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_tri)

    # Pattern 10: L-shape
    axial_L = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 1), (1, 2), (1, 3)
    ]
    candidates.append(axial_L)

    # Pattern 11: T-shape
    axial_T = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (2, 1), (2, 2), (2, 3),
        (1, 1), (3, 1)
    ]
    candidates.append(axial_T)

    # Pattern 12: Compact with gaps
    axial_gaps = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_gaps)

    # Pattern 13: Wide flat
    axial_wide = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_wide)

    # Pattern 14: Narrow tall
    axial_tall = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4),
        (0, 5)
    ]
    candidates.append(axial_tall)

    # Pattern 15: Mixed
    axial_mixed = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_mixed)

    # Pattern 16: Another compact
    axial_compact2 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_compact2)

    # Pattern 17: Asymmetric
    axial_asym = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_asym)

    # Pattern 18: Dense center
    axial_dense = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2)
    ]
    candidates.append(axial_dense)

    # Pattern 19: Extended cross
    axial_xcross = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2)
    ]
    candidates.append(axial_xcross)

    # Pattern 20: Another variant
    axial_var20 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_var20)

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Evaluate all candidates
    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

        pts = axial_to_cartesian(axial)

        # Optimize beta
        S_est, beta_best = optimize_beta(pts, n_coarse=180, n_fine=80)

        if S_est < best_global_S:
            # Verify with exact LP
            S_exact, center_exact = compute_S_and_center(pts, beta_best)
            if S_exact < best_global_S:
                best_global_S = S_exact
                best_global_pts = pts
                best_global_beta = beta_best
                best_global_center = center_exact

        # Also try with slight perturbations of beta
        for dbeta in [0.005, -0.005, 0.01, -0.01]:
            beta_try = beta_best + dbeta
            if beta_try < 0:
                beta_try += math.pi / 6
            elif beta_try > math.pi / 6:
                beta_try -= math.pi / 6

            S_try, center_try = compute_S_and_center(pts, beta_try)
            if S_try < best_global_S:
                best_global_S = S_try
                best_global_pts = pts
                best_global_beta = beta_try
                best_global_center = center_try

    # Fallback if nothing found
    if best_global_center is None or best_global_S == float('inf'):
        pts = axial_to_cartesian(candidates[0])
        best_global_pts = pts
        best_global_beta = 0.0
        center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_global_center = center
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Scale points very slightly to avoid numerical issues
    SCALE = 1.0 + 1e-10
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))

    pts_scaled = np.array(pts_scaled)

    # Recompute S with scaled points
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
