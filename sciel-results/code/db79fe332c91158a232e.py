import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Polyhex topologies (axial coordinates)
    candidates = [
        # Center + 6 neighbors + 4 more
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        # Compact 3x3-ish
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        # Row-based
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        # Star
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)],
        # Diagonal strip
        [(0,0),(1,0),(2,0),(3,0),(4,0),(1,1),(2,1),(3,1),(2,2),(3,2),(4,2)],
        # T-shape
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(1,3),(1,4),(0,2),(2,2)],
        # Compact diamond
        [(0,0),(1,0),(0,1),(1,1),(-1,1),(0,2),(1,2),(-1,2),(2,0),(-2,0),(0,-1)],
        # L-shape
        [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4),(1,4)],
        # Zigzag
        [(0,0),(1,0),(1,1),(2,1),(2,2),(3,2),(0,1),(0,2),(3,0),(3,1),(4,1)],
        # Compact 2-row
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(4,1),(1,2),(2,2)],
        # Cross
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)],
        # Compact
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,2),(1,3),(2,3)],
        # Wide
        [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(1,1),(2,1),(3,1),(2,2),(3,2)],
        # Compact hexagon-ish
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)],
        # Another compact
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        # Symmetric
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,1),(-2,1),(1,-1),(-1,-1)],
    ]

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def exact_S(pts, cx, cy, beta):
        """Compute exact outer hexagon side length."""
        max_proj = 0.0
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        return (max_proj + APOTHEM) / APOTHEM

    def smooth_S(pts, cx, cy, beta, temp=0.05):
        """Smooth approximation using log-sum-exp."""
        all_projs = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            all_projs.append(projs)
            all_projs.append(-projs)
        all_projs = np.concatenate(all_projs)
        m = np.max(all_projs)
        if m > 1000:
            return float('inf')
        shifted = (all_projs - m) / temp
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted)))
        D = m + temp * log_sum
        return (D + APOTHEM) / APOTHEM

    def optimize_topology(pts):
        """Optimize (cx, cy, beta) for a given point set."""
        best_S = float('inf')
        best_params = None

        # Initial center estimate
        cx0 = float(np.mean(pts[:, 0]))
        cy0 = float(np.mean(pts[:, 1]))

        # Coarse grid over beta (0 to pi/6)
        for i in range(73):
            beta = i * (math.pi / 6.0) / 72.0
            S = exact_S(pts, cx0, cy0, beta)
            if S < best_S:
                best_S = S
                best_params = (cx0, cy0, beta)

        if best_params is None:
            return float('inf'), (cx0, cy0, 0.0)

        # Local optimization with smooth surrogate, decreasing temperature
        temperatures = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        current_params = best_params[:]

        for temp in temperatures:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break

            def smooth_obj(params):
                cx, cy, beta = params
                beta = ((beta % (math.pi / 6.0)) + math.pi / 6.0) % (math.pi / 6.0)
                return smooth_S(pts, cx, cy, beta, temp)

            # Multi-start
            starts = [
                current_params,
                [current_params[0] + 0.1, current_params[1], current_params[2]],
                [current_params[0] - 0.1, current_params[1], current_params[2]],
                [current_params[0], current_params[1] + 0.1, current_params[2]],
                [current_params[0], current_params[1] - 0.1, current_params[2]],
                [current_params[0], current_params[1], current_params[2] + 0.05],
                [current_params[0], current_params[1], current_params[2] - 0.05],
            ]

            for init in starts:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                try:
                    res = minimize(smooth_obj, init, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
                    if res.fun < best_S * 1.001:
                        cx_r, cy_r, beta_r = res.x
                        beta_r = ((beta_r % (math.pi / 6.0)) + math.pi / 6.0) % (math.pi / 6.0)
                        S_exact = exact_S(pts, cx_r, cy_r, beta_r)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_params = (cx_r, cy_r, beta_r)
                            current_params = [cx_r, cy_r, beta_r]
                except Exception:
                    pass

        # Final exact polish
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def exact_obj(params):
                cx, cy, beta = params
                beta = ((beta % (math.pi / 6.0)) + math.pi / 6.0) % (math.pi / 6.0)
                return exact_S(pts, cx, cy, beta)

            polish_starts = [
                list(best_params),
                [best_params[0] + 0.01, best_params[1], best_params[2]],
                [best_params[0] - 0.01, best_params[1], best_params[2]],
                [best_params[0], best_params[1] + 0.01, best_params[2]],
                [best_params[0], best_params[1] - 0.01, best_params[2]],
                [best_params[0], best_params[1], best_params[2] + 0.01],
                [best_params[0], best_params[1], best_params[2] - 0.01],
            ]

            for init in polish_starts:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                try:
                    res = minimize(exact_obj, init, method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
                    if res.fun < best_S:
                        cx_r, cy_r, beta_r = res.x
                        beta_r = ((beta_r % (math.pi / 6.0)) + math.pi / 6.0) % (math.pi / 6.0)
                        S_exact = exact_S(pts, cx_r, cy_r, beta_r)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_params = (cx_r, cy_r, beta_r)
                except Exception:
                    pass

        return best_S, best_params

    # Optimize all topologies
    best_global_S = float('inf')
    best_global_params = None
    best_global_pts = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, params = optimize_topology(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_params = params
            best_global_pts = pts

    if best_global_params is None or best_global_S == float('inf'):
        pts = axial_to_cartesian(candidates[0])
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        best_global_S = exact_S(pts, cx, cy, 0.0)
        best_global_params = (cx, cy, 0.0)
        best_global_pts = pts

    cx, cy, beta = best_global_params
    angle_deg = math.degrees(beta)

    # Compute final S
    S_final = exact_S(best_global_pts, cx, cy, beta)

    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)

    return inner, center, side, rotation
