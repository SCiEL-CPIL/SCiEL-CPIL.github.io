import math
import numpy as np
import time
from scipy.optimize import linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def _time_ok():
    return time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- Candidate point sets (axial coordinates) ---
    candidates = []

    # Hexagonal cluster: center + 6 neighbors + 4 more
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ])
    # Star: center + 6 neighbors + 4 opposite
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ])
    # 3x3 block + 2 extras
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ])
    # 4x3 block + 1 extra
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ])
    # 3x4 block + 1 extra
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ])
    # 3x3 + 2 top
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ])
    # Cross-like
    candidates.append([
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ])
    # 5+6 row
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ])
    # Center + 6 + 4 asymmetric
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ])
    # 3x3 + 2 top (variant)
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ])
    # 2+4+4+1
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ])
    # 4+2+2+2+1
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ])
    # Center + 6 + 4 (variant)
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ])
    # Center + 6 + 4 (variant)
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ])
    # 3x3 with extras
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ])
    # Staircase
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ])
    # Center + 6 + 4 (variant)
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ])
    # 4+3+3
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ])
    # 2+3+2+3+1
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ])
    # Center + 6 + 4 (variant)
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ])
    # 4+4+3
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ])
    # 3+4+3+1
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ])
    # Center + 6 + 4 (variant)
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ])
    # 2+2+3+2+2
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ])
    # 3+2+3+3
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ])
    # 4+3+4+2
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ])
    # Center + 6 + 4 (variant)
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ])
    # 2+3+2+3+3
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4)
    ])
    # 3+3+2+2+1
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (2, 3)
    ])
    # 4+3+2+1+1
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3),
        (3, 3)
    ])
    # 3+3+3+1+1
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ])
    # 2+4+3+1+1
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3),
        (2, 3)
    ])
    # 3+4+2+1+1
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2),
        (0, 3),
        (3, 3)
    ])
    # 2+2+4+2+1
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3),
        (2, 3)
    ])
    # 2+3+3+2+1 (alternate)
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ])
    # 2+3+3+2+1
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4)
    ])
    # 3+3+3+1+1 (variant)
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3),
        (0, 3)
    ])
    # 4+2+2+2+1 (alternate)
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ])
    # 3+2+3+2+1
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4)
    ])
    # 2+3+2+3+2
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4)
    ])
    # 3+3+2+2+1 (variant)
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (2, 3)
    ])
    # 2+4+2+2+1
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (2, 3)
    ])
    # 3+2+3+3 (variant)
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ])
    # 2+3+3+3+0
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ])
    # 3+3+3+2+0
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ])
    # 4+2+3+2
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ])
    # 2+4+4+1 (variant)
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (2, 3)
    ])
    # 3+3+2+3+0
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ])

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def normals_for_beta(beta):
        return np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])

    def calc_S_for_beta(pts, beta):
        if not _time_ok():
            return float('inf'), None, None

        normals = normals_for_beta(beta)

        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []

        for nx, ny in normals:
            for px, py in pts:
                proj = px * nx + py * ny
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-proj)
                A_ub.append([nx, ny, -1.0])
                b_ub.append(proj)

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

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
        normals = normals_for_beta(beta)
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 100:
            return float('inf'), (cx, cy), float('inf')
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D

    def exact_S(pts, beta, cx, cy):
        normals = normals_for_beta(beta)
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        max_proj = np.max(np.abs(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        return S

    def newton_polish_center(pts, beta, cx, cy, max_iter=80, tol=1e-14):
        """High-precision 2D Newton polish of the center at fixed rotation.
        Solves the smallest enclosing hexagon center via the LP's active-set
        structure: minimize max_k |n_k . (p - c)| over c. This is a convex
        piecewise-linear problem; we solve it exactly with linprog (the true
        optimum) then refine with a few Newton steps on the smooth active
        residual to remove any residual solver slack."""
        normals = normals_for_beta(beta)
        # Solve exact LP for the center (this is the global optimum for c at
        # fixed beta). Then do a tiny Newton correction on the residual of the
        # active faces to drive the max-projection to machine precision.
        try:
            S_lp, (cx0, cy0), D_lp = calc_S_for_beta(pts, beta)
        except Exception:
            S_lp, (cx0, cy0), D_lp = float('inf'), (cx, cy), float('inf')
        if S_lp == float('inf'):
            cx0, cy0 = cx, cy
        else:
            cx, cy = cx0, cy0

        # Newton on residual: at optimum, the active faces (those attaining
        # +/-D) should be balanced. We nudge c so the top few projections
        # equalize. Use a smooth softmax-weighted gradient.
        for _ in range(max_iter):
            if not _time_ok():
                break
            pts_centered = pts - np.array([cx, cy])
            projs = pts_centered @ normals.T  # shape (11, 6)
            abs_p = np.abs(projs)
            D = float(np.max(abs_p))
            # softmax weights over the near-active (within eps of D)
            temp = 1e-6
            w = np.exp((abs_p - D) / temp)
            w = w / w.sum()
            # gradient of sum w_k * |n_k.(p-c)| w.r.t. c
            # d/d c of |n.(p-c)| = -sign(n.(p-c)) * n
            signs = np.sign(projs)
            # weighted sum over all points/faces
            grad = -np.sum(w[:, None, :] * signs[:, None, :] * normals[None, :, :], axis=(0, 1))
            # Hessian approx (diagonal of weighted |n|): use identity * scale
            scale = float(np.sum(w)) + 1e-12
            step = grad / scale
            # Damped update
            alpha = 1.0
            for _ in range(8):
                new_cx = cx - alpha * step[0]
                new_cy = cy - alpha * step[1]
                new_D = float(np.max(np.abs((pts - np.array([new_cx, new_cy])) @ normals.T)))
                if new_D <= D + 1e-18:
                    cx, cy = new_cx, new_cy
                    break
                alpha *= 0.5
            D_new = float(np.max(np.abs((pts - np.array([cx, cy])) @ normals.T)))
            if abs(D_new - D) < tol:
                break
        D_final = float(np.max(np.abs((pts - np.array([cx, cy])) @ normals.T)))
        S = (D_final + APOTHEM) / APOTHEM
        return S, (cx, cy), D_final

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if not _time_ok():
            return float('inf'), 0.0, None

        cx0 = float(np.mean(pts[:, 0]))
        cy0 = float(np.mean(pts[:, 1]))

        # Coarse grid over beta using smooth surrogate at centroid
        best_smooth_S = float('inf')
        best_smooth_beta = 0.0
        for i in range(49):
            beta = i * (math.pi / 6) / 48
            S_s, _, _ = smooth_calc_S(pts, beta, cx0, cy0, temperature=0.1)
            if S_s < best_smooth_S:
                best_smooth_S = S_s
                best_smooth_beta = beta

        # Smooth-surrogate local optimization from multiple starts, annealing
        try:
            if _time_ok():
                temperatures = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
                start_betas = [best_smooth_beta, 0.0, math.pi / 12, math.pi / 6,
                               best_smooth_beta + 0.03, best_smooth_beta - 0.03,
                               best_smooth_beta + 0.06, best_smooth_beta - 0.06]
                for temp in temperatures:
                    if not _time_ok():
                        break
                    for sb in start_betas:
                        if not _time_ok():
                            break

                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                            S, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=t)
                            return S

                        init = [cx0, cy0, sb]
                        res = minimize(smooth_obj, init, method='Nelder-Mead',
                                       options={'maxiter': 4000, 'xatol': 1e-12, 'fatol': 1e-12})
                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                            # Verify + refine with exact LP
                            S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = c_exact
        except Exception:
            pass

        if best_center is None:
            beta_r = best_smooth_beta
            S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
            if S_exact < best_S:
                best_S = S_exact
                best_beta = beta_r
                best_center = c_exact

        # LP-based center refinement at best beta
        try:
            if _time_ok() and best_center is not None:
                S_lp, c_lp, _ = calc_S_for_beta(pts, best_beta)
                if S_lp < best_S:
                    best_S = S_lp
                    best_center = c_lp
        except Exception:
            pass

        # Final exact (non-smooth) polish over (cx, cy, beta)
        try:
            if _time_ok() and best_center is not None:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return exact_S(pts, beta, cx, cy)

                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.01],
                    [best_center[0], best_center[1], best_beta - 0.01],
                    [best_center[0] + 0.01, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.01, best_beta],
                    [best_center[0] - 0.01, best_center[1], best_beta],
                    [best_center[0], best_center[1] - 0.01, best_beta],
                    [best_center[0], best_center[1], best_beta + 0.02],
                    [best_center[0], best_center[1], best_beta - 0.02],
                ]
                for init in polish_starts:
                    if not _time_ok():
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                   options={'maxiter': 80000, 'xatol': 1e-15, 'fatol': 1e-15})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    if not _time_ok():
                        break
                    try:
                        res2 = minimize(objective, [best_center[0], best_center[1], best_beta],
                                        method='SLSQP', options={'maxiter': 20000, 'ftol': 1e-16})
                        if res2.fun < best_S:
                            best_S = res2.fun
                            best_center = (res2.x[0], res2.x[1])
                            best_beta = ((res2.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        # High-precision Newton polish of the center at the final rotation
        try:
            if _time_ok() and best_center is not None:
                S_n, c_n, _ = newton_polish_center(pts, best_beta, best_center[0], best_center[1])
                if S_n < best_S:
                    best_S = S_n
                    best_center = c_n
        except Exception:
            pass

        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    def quick_estimate(axial_list):
        pts = axial_to_cartesian(axial_list)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, float(np.max(projs)), -float(np.min(projs)))
        return (max_proj + APOTHEM) / APOTHEM

    estimated_scores = [(quick_estimate(ax), ax) for ax in candidates]
    estimated_scores.sort(key=lambda x: x[0])

    for rank, (_, axial) in enumerate(estimated_scores):
        if not _time_ok():
            break
        if rank > 10 and estimated_scores[rank][0] > best_global_S + 0.05:
            continue
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
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, float(np.max(projs)), -float(np.min(projs)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Tiny safety margin to guard against numerical overlap
    SCALE = 1.0 + 1e-10
    pts_scaled = np.array([
        (cx + (x - cx) * SCALE, cy + (y - cy) * SCALE) for x, y in best_global_pts
    ])

    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), -float(np.min(projs)))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
