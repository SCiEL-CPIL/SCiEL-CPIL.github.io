import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
SQRT3 = math.sqrt(3.0)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Generate candidate configurations
    candidates = []

    # Classic hexagonal cluster patterns
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
        (0, 1), (1, 1), (2, 1), (3, 1),
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

    # Additional diverse candidates
    axial_29 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_32)

    # New: more compact triangular-ish arrangements
    axial_33 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_33)

    axial_34 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_34)

    axial_35 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_35)

    axial_36 = [
        (0, 0), (1, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4)
    ]
    candidates.append(axial_36)

    axial_37 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_37)

    axial_38 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_38)

    axial_39 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_39)

    axial_40 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_40)

    # Remove duplicates by checking sorted tuples
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    candidates = unique_candidates

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))

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
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def exact_objective(params, pts):
        """Exact objective: positions of n hexagons + center + beta"""
        n = len(pts)
        pos = np.array(params[:2 * n]).reshape(n, 2)
        cx_v = params[2 * n]
        cy_v = params[2 * n + 1]
        beta_v = params[2 * n + 2]
        beta_v = ((beta_v % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)

        normals = np.array([
            (math.cos(beta_v + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta_v + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])

        pts_centered = pos - np.array([cx_v, cy_v])
        projs = np.abs(pts_centered @ normals.T)
        max_proj = np.max(projs)

        S = (max_proj + APOTHEM) / APOTHEM
        return S

    def smooth_objective(params, pts, temperature=0.05):
        """Smooth surrogate using log-sum-exp"""
        n = len(pts)
        pos = np.array(params[:2 * n]).reshape(n, 2)
        cx_v = params[2 * n]
        cy_v = params[2 * n + 1]
        beta_v = params[2 * n + 2]
        beta_v = ((beta_v % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)

        normals = np.array([
            (math.cos(beta_v + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta_v + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])

        pts_centered = pos - np.array([cx_v, cy_v])
        projs = pts_centered @ normals.T  # (n, 6)

        # All absolute projections
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])

        max_val = np.max(all_projs)
        if max_val > 100:
            return 1e10

        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S

    def optimize_candidate(pts):
        """Optimize center, rotation for fixed positions, then free positions"""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None, None

        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        # Phase 1: Coarse grid + refine for beta with fixed positions
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        # Refine beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                windows = [
                    (0, math.pi / 6),
                    (max(0, best_beta - 0.05), min(math.pi / 6, best_beta + 0.05)),
                ]
                for lo, hi in windows:
                    if hi <= lo:
                        continue
                    try:
                        res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                             bounds=(lo, hi), method='bounded',
                                             options={'maxiter': 200, 'xatol': 1e-14})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_beta = res.x
                            _, best_center, _ = calc_S_for_beta(pts, best_beta)
                    except Exception:
                        pass
        except Exception:
            pass

        # Phase 2: Free-position optimization (break rigid constraint)
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                n = len(pts)
                init_params = []
                for x, y in pts:
                    init_params.append(float(x))
                    init_params.append(float(y))
                init_params.append(best_center[0])
                init_params.append(best_center[1])
                init_params.append(best_beta)

                # Multiple perturbation scales
                for pert_scale in [0.05, 0.02, 0.01, 0.005]:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for trial in range(3):
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        np.random.seed(hash((pert_scale, trial)) % 2**32)
                        perturbed = init_params.copy()
                        for i in range(n):
                            perturbed[2 * i] += np.random.uniform(-pert_scale, pert_scale)
                            perturbed[2 * i + 1] += np.random.uniform(-pert_scale, pert_scale)

                        # First use smooth objective for broader search
                        res_smooth = minimize(lambda p: smooth_objective(p, pts, temperature=0.05),
                                            perturbed, method='Nelder-Mead',
                                            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})

                        if res_smooth.fun < 1e10:
                            # Polish with exact objective
                            res_exact = minimize(lambda p: exact_objective(p, pts),
                                               res_smooth.x, method='Nelder-Mead',
                                               options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})

                            if res_exact.fun < best_S:
                                best_S = res_exact.fun
                                for i in range(n):
                                    pts[i, 0] = res_exact.x[2 * i]
                                    pts[i, 1] = res_exact.x[2 * i + 1]
                                best_center = (res_exact.x[2 * n], res_exact.x[2 * n + 1])
                                best_beta = ((res_exact.x[2 * n + 2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                                init_params = res_exact.x.copy()
        except Exception:
            pass

        # Phase 3: Final high-precision polish
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                n = len(pts)
                polish_starts = []
                cx0, cy0 = best_center
                for db in [0.0, 0.002, -0.002, 0.005, -0.005]:
                    for dx, dy in [(0, 0), (0.005, 0), (-0.005, 0), (0, 0.005), (0, -0.005)]:
                        ps = []
                        for i in range(n):
                            ps.append(pts[i, 0] + dx)
                            ps.append(pts[i, 1] + dy)
                        ps.append(cx0 + dx)
                        ps.append(cy0 + dy)
                        ps.append(best_beta + db)
                        polish_starts.append(ps)

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(lambda p: exact_objective(p, pts), init, method='Nelder-Mead',
                                 options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
                    if res.fun < best_S:
                        best_S = res.fun
                        for i in range(n):
                            pts[i, 0] = res.x[2 * i]
                            pts[i, 1] = res.x[2 * i + 1]
                        best_center = (res.x[2 * n], res.x[2 * n + 1])
                        best_beta = ((res.x[2 * n + 2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        except Exception:
            pass

        return best_S, best_beta, best_center, pts

    # Evaluate all candidates
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Sort candidates by initial LP score to prioritize good ones
    candidate_scores = []
    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        # Quick LP estimate at beta=0
        S0, _, _ = calc_S_for_beta(pts, 0.0)
        candidate_scores.append((S0, idx, axial))

    candidate_scores.sort()

    for S0, idx, axial in candidate_scores:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        # Skip if already too far from best
        if S0 > best_global_S + 0.1 and best_global_S < float('inf'):
            continue
        pts = axial_to_cartesian(axial).copy()
        S, beta, center, pts_opt = optimize_candidate(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts_opt
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Final verification and slight scale-up for safety
    SCALE = 1.0 + 1e-10
    pts_scaled = np.array([
        (cx + (x - cx) * SCALE, cy + (y - cy) * SCALE)
        for x, y in best_global_pts
    ])

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
