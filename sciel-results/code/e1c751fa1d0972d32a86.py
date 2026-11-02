import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Precompute the 6 fixed unit normals for the unit hexagon (angle 0)
    # Used for the Minkowski-sum overlap constraint (inner hexagons are axis-aligned to each other)
    hex_normals = np.array([
        [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
        for k in range(6)
    ])

    # 11 distinct initial candidate configurations in axial coordinates (used as seeds for random restarts)
    candidates = [
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 1), (-1, 2), (-2, 1), (2, -1)],
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (0, 2), (0, -2)],
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (-1, 1), (3, 1)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3)],
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3)],
        [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (2, 0), (-2, 0), (0, 2), (0, -2)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 1), (-2, 2), (-1, 3), (1, -2)],
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (1, 3), (2, 3)],
        [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 2)],
    ]

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def check_overlap(pts):
        """Check if any two hexagons overlap (axis-aligned to each other)."""
        n_pts = len(pts)
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                d = pts[i] - pts[j]
                projs = np.abs(d @ hex_normals.T)
                if np.min(projs) < math.sqrt(3) - 1e-9:
                    return False
        return True

    def exact_S(pts, cx, cy, beta):
        """Exact enclosing hexagon side length S."""
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        D = np.max(all_projs)
        return (D + APOTHEM) / APOTHEM

    def smooth_obj(params, temperature=0.05):
        """Smooth surrogate objective: S_smooth + overlap_penalty."""
        n_pts = N
        pts = params[:2 * n_pts].reshape(n_pts, 2)
        cx = params[2 * n_pts]
        cy = params[2 * n_pts + 1]
        beta = params[2 * n_pts + 2]

        # Normalize beta to [0, pi/6)
        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)

        # 1. Smooth enclosing hexagon radius via log-sum-exp
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])

        max_val = np.max(all_projs)
        if max_val > 100:
            return 1e10

        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S_smooth = (D + APOTHEM) / APOTHEM

        # 2. Overlap penalty (force-directed soft constraint)
        penalty = 0.0
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                d = pts[i] - pts[j]
                projs = np.abs(d @ hex_normals.T)
                min_proj = np.min(projs)
                if min_proj < math.sqrt(3):
                    violation = math.sqrt(3) - min_proj
                    penalty += violation * 100.0

        return S_smooth + penalty

    def optimize_from_start(pts_init, cx_init, cy_init, beta_init, temperature=0.05):
        """Run Nelder-Mead from a given starting point."""
        init_params = np.concatenate([pts_init.ravel(), [cx_init, cy_init, beta_init]])

        def obj(p):
            return smooth_obj(p, temperature)

        try:
            res = minimize(obj, init_params, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})

            # Extract result
            n_pts = N
            pts_opt = res.x[:2 * n_pts].reshape(n_pts, 2)
            cx_opt = res.x[2 * n_pts]
            cy_opt = res.x[2 * n_pts + 1]
            beta_opt = ((res.x[2 * n_pts + 2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)

            # Verify exact S and non-overlap
            if check_overlap(pts_opt):
                S_exact = exact_S(pts_opt, cx_opt, cy_opt, beta_opt)
                return S_exact, pts_opt, cx_opt, cy_opt, beta_opt
            else:
                return float('inf'), None, None, None, None
        except Exception:
            return float('inf'), None, None, None, None

    # Multi-start force-directed search
    best_global_S = float('inf')
    best_global_pts = None
    best_global_center = None
    best_global_beta = 0.0

    # Precompute all candidate cartesian points
    candidate_pts_list = [axial_to_cartesian(c) for c in candidates]

    # Phase 1: Optimize each candidate configuration with its natural center
    for pts_init in candidate_pts_list:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

        cx_init = float(np.mean(pts_init[:, 0]))
        cy_init = float(np.mean(pts_init[:, 1]))
        beta_init = 0.0

        S, pts_opt, cx_opt, cy_opt, beta_opt = optimize_from_start(
            pts_init, cx_init, cy_init, beta_init, temperature=0.05)

        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts_opt
            best_global_center = (cx_opt, cy_opt)
            best_global_beta = beta_opt

    # Phase 2: Random restarts with perturbation around best and other candidates
    rng = np.random.default_rng(42)
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

        # Pick a random candidate as base
        base_idx = rng.integers(0, len(candidate_pts_list))
        pts_init = candidate_pts_list[base_idx].copy()

        # Add random perturbation
        perturbation_scale = 0.3
        pts_init += rng.normal(0, perturbation_scale, size=pts_init.shape)

        cx_init = float(np.mean(pts_init[:, 0]))
        cy_init = float(np.mean(pts_init[:, 1]))
        beta_init = rng.uniform(0, math.pi / 6)

        S, pts_opt, cx_opt, cy_opt, beta_opt = optimize_from_start(
            pts_init, cx_init, cy_init, beta_init, temperature=0.05)

        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts_opt
            best_global_center = (cx_opt, cy_opt)
            best_global_beta = beta_opt

    # Phase 3: Tighten with lower temperature around the best solution
    if best_global_pts is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        S_tight, pts_tight, cx_tight, cy_tight, beta_tight = optimize_from_start(
            best_global_pts, best_global_center[0], best_global_center[1], best_global_beta,
            temperature=0.01)

        if S_tight < best_global_S:
            best_global_S = S_tight
            best_global_pts = pts_tight
            best_global_center = (cx_tight, cy_tight)
            best_global_beta = beta_tight

    # Fallback if all optimizations failed
    if best_global_pts is None or best_global_S == float('inf'):
        best_global_pts = candidate_pts_list[0]
        best_global_center = (float(np.mean(best_global_pts[:, 0])),
                              float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    # Final: compute exact S and prepare output
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Verify and compute final S
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (best_global_pts[:, 0] - cx) * nx + (best_global_pts[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

    D_final = max_proj
    S_final = (D_final + APOTHEM) / APOTHEM

    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)

    return inner, center, side, rotation
