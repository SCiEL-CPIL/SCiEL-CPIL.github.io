import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates)
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

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_params(pts, beta, scale):
        """
        Given base points (in axial-derived cartesian), rotation beta, and scale factor,
        find the minimum enclosing regular hexagon side length.
        The inner hexagons are at scaled positions, each with side 1.
        For disjointness, we need scale >= 1 (at scale=1 they're exactly touching).
        We use scale slightly > 1 for safety.
        
        Returns (S, cx, cy, D) where S is the outer hexagon side length.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None, None
        
        # Scale the points
        pts_scaled = pts * scale
        
        # Normals of the enclosing hexagon
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # LP: minimize D subject to all points fitting in hexagon centered at (cx, cy)
        # For each normal n and point p: (p - c)·n <= D and -(p - c)·n <= D
        # i.e., p·n - c·n <= D and -p·n + c·n <= D
        # Variables: cx, cy, D
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts_scaled:
                # (px-cx)*nx + (py-cy)*ny <= D  =>  -cx*nx - cy*ny - D <= -px*nx - py*ny
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                
                # -((px-cx)*nx + (py-cy)*ny) <= D  =>  cx*nx + cy*ny - D <= px*nx + py*ny
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
                return float('inf'), None, None, None
        except Exception:
            return float('inf'), None, None, None

    def smooth_objective(params, pts, temperature=0.1):
        """
        Smooth surrogate for the objective.
        params = [cx, cy, beta, scale]
        """
        cx, cy, beta, scale = params
        if scale < 1.0:
            return 1e10
        
        pts_scaled = pts * scale
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        projs = []
        for nx, ny in normals:
            for px, py in pts_scaled:
                proj = (px - cx) * nx + (py - cy) * ny
                projs.append(proj)
                projs.append(-proj)
        
        projs = np.array(projs)
        max_val = np.max(projs)
        if max_val > 1000:
            return 1e10
        
        log_sum = np.log(np.sum(np.exp((projs - max_val) / temperature))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S

    def optimize_shape(pts):
        best_S = float('inf')
        best_params = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None
        
        # Stage 1: Coarse grid search over beta and scale
        scale_min = 1.0 + 1e-6
        scale_max = 1.0 + 0.05
        n_scale = 10
        n_beta = 24
        
        for i in range(n_scale):
            scale = scale_min + i * (scale_max - scale_min) / (n_scale - 1)
            for j in range(n_beta):
                beta = j * (math.pi / 6) / n_beta
                S, center, D = calc_S_for_params(pts, beta, scale)
                if S < best_S:
                    best_S = S
                    best_params = (center[0], center[1], beta, scale)
        
        if best_params is None:
            return float('inf'), None
        
        # Stage 2: Multi-temperature smooth optimization
        temperatures = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        current_params = list(best_params)
        
        for temp in temperatures:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            def obj_smooth(params, t=temp):
                return smooth_objective(params, pts, temperature=t)
            
            try:
                res = minimize(obj_smooth, current_params, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_S:
                    # Verify with exact LP
                    cx, cy, beta, scale = res.x
                    if scale >= 1.0:
                        S_exact, center, D = calc_S_for_params(pts, beta, scale)
                        if S_exact < best_S:
                            best_S = S_exact
                            current_params = [cx, cy, beta, scale]
                            best_params = (cx, cy, beta, scale)
            except Exception:
                pass
        
        # Stage 3: Final Nelder-Mead with exact objective
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def obj_exact(params):
                cx, cy, beta, scale = params
                if scale < 1.0:
                    return 1e10
                S, center, D = calc_S_for_params(pts, beta, scale)
                return S
            
            try:
                res = minimize(obj_exact, current_params, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_S:
                    cx, cy, beta, scale = res.x
                    if scale >= 1.0:
                        S_final, center, D = calc_S_for_params(pts, beta, scale)
                        if S_final < best_S:
                            best_S = S_final
                            best_params = (cx, cy, beta, scale)
            except Exception:
                pass
        
        return best_S, best_params

    best_global_S = float('inf')
    best_global_params = None
    best_global_pts = None

    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts = axial_to_cartesian(axial)
        S, params = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_params = params
            best_global_pts = pts

    if best_global_params is None or best_global_S == float('inf'):
        # Fallback: use first candidate with no scaling
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_params = (0.0, 0.0, 0.0, 1.0 + 1e-6)
        S, center, D = calc_S_for_params(best_global_pts, 0.0, 1.0 + 1e-6)
        if center is not None:
            best_global_params = (center[0], center[1], 0.0, 1.0 + 1e-6)
            best_global_S = S
        else:
            # Very basic fallback
            max_proj = 0
            for k in range(6):
                ang = math.pi / 6.0 + k * math.pi / 3.0
                nx, ny = math.cos(ang), math.sin(ang)
                projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
                max_proj = max(max_proj, np.max(projs), -np.min(projs))
            best_global_S = (max_proj + APOTHEM) / APOTHEM
            best_global_params = (0.0, 0.0, 0.0, 1.0 + 1e-6)

    # Prepare output
    cx, cy, beta, scale = best_global_params
    angle_deg = math.degrees(beta)
    
    pts_scaled = best_global_pts * scale
    
    # Recalculate S for the final configuration
    max_proj = 0
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D_final = max_proj
    S_final = (D_final + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
