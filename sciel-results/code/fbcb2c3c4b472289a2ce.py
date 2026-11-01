import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def hex_support(alpha, direction_angle):
    """
    Compute the support function of a unit regular hexagon with rotation alpha
    in the given direction.
    For a regular hexagon of side 1 (circumradius 1), vertices are at angles
    alpha + k*pi/3 for k=0..5.
    Support in direction theta = max_k cos(theta - alpha - k*pi/3)
    """
    diff = (direction_angle - alpha) % (math.pi / 3.0)
    d = min(diff, math.pi / 3.0 - diff)
    return math.cos(d)

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate candidate axial coordinate configurations for 11 unit hexagons
    candidates = []

    # Flower shape: center + 6 neighbors + 4 from next ring
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    # Alternative: center + 6 neighbors + 4 different from next ring
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    # Compact 3x3 plus 2
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    # 4x3 minus 1
    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    # 3x4 minus 1
    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    # 3x4 minus 2
    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    # Cross shape
    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    # 5+6 row
    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    # Modified flower
    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    # 3x3 + 2
    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    # 2+3+4+2
    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    # Diagonal strip
    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    # Flower variant
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

    def axial_to_cartesian(axial_list, scale=1.0 + 1e-6):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0) * scale
            y = 1.5 * r * scale
            pts.append((x, y))
        return np.array(pts)

    def solve_lp_for_shape(pts, beta):
        """
        Solve LP to find minimal enclosing hexagon for given center points and outer rotation.
        Returns (cx, cy, S) or (None, None, inf)
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return None, None, float('inf')
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        c_obj = np.array([0.0, 0.0, 1.0])  # Minimize D
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                # (px - cx)*nx + (py - cy)*ny <= D
                # -nx*cx - ny*cy - D <= -(px*nx + py*ny)
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                # -(px - cx)*nx - (py - cy)*ny <= D
                # nx*cx + ny*cy - D <= px*nx + py*ny
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                cx, cy, D = res.x
                S = (D + APOTHEM) / APOTHEM
                return cx, cy, S
            else:
                return None, None, float('inf')
        except Exception:
            return None, None, float('inf')

    def optimize_shape(pts):
        """
        Optimize outer rotation beta for a given set of center points.
        Returns (best_S, best_beta, best_cx, best_cy)
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None, None
        
        best_S = float('inf')
        best_beta = 0.0
        best_cx = None
        best_cy = None
        
        # Coarse grid search
        n_coarse = 37
        for i in range(n_coarse):
            beta = i * (math.pi / 6) / (n_coarse - 1)
            cx, cy, S = solve_lp_for_shape(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_cx = cx
                best_cy = cy
        
        # Bounded refinement
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def obj_beta(b):
                    _, _, S = solve_lp_for_shape(pts, b)
                    return S
                
                res = minimize_scalar(obj_beta, bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    best_cx, best_cy, _ = solve_lp_for_shape(pts, best_beta)
        except Exception:
            pass
        
        # Local polish around best
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                local_lo = max(0, best_beta - 0.02)
                local_hi = min(math.pi / 6, best_beta + 0.02)
                
                def obj_beta_local(b):
                    _, _, S = solve_lp_for_shape(pts, b)
                    return S
                
                res2 = minimize_scalar(obj_beta_local, bounds=(local_lo, local_hi), method='bounded',
                                      options={'maxiter': 500, 'xatol': 1e-14})
                if res2.fun < best_S:
                    best_S = res2.fun
                    best_beta = res2.x
                    best_cx, best_cy, _ = solve_lp_for_shape(pts, best_beta)
        except Exception:
            pass
        
        return best_S, best_beta, best_cx, best_cy

    # Try all candidate shapes
    best_overall_S = float('inf')
    best_overall_beta = 0.0
    best_overall_cx = None
    best_overall_cy = None
    best_overall_pts = None
    best_overall_axial = None

    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts = axial_to_cartesian(axial)
        S, beta, cx, cy = optimize_shape(pts)
        
        if S < best_overall_S:
            best_overall_S = S
            best_overall_beta = beta
            best_overall_cx = cx
            best_overall_cy = cy
            best_overall_pts = pts
            best_overall_axial = axial

    # Fallback if all failed
    if best_overall_S == float('inf') or best_overall_pts is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # High-precision polish using smooth surrogate
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        pts = best_overall_pts
        n_pts = len(pts)
        
        def smooth_objective(x):
            cx, cy = x[0], x[1]
            beta = x[2]
            
            normals = []
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            constraints = []
            for nx, ny in normals:
                for px, py in pts:
                    v1 = (px - cx) * nx + (py - cy) * ny
                    v2 = -(px - cx) * nx - (py - cy) * ny
                    constraints.append(v1)
                    constraints.append(v2)
            
            constraints = np.array(constraints)
            max_c = np.max(constraints)
            
            temp = 100.0
            shifted = constraints - max_c
            shifted = np.clip(shifted, -50, 50)
            smooth_max = max_c + (1.0 / temp) * math.log(np.sum(np.exp(temp * shifted)))
            
            S = (smooth_max + APOTHEM) / APOTHEM
            return S
        
        x0 = np.array([best_overall_cx, best_overall_cy, best_overall_beta])
        
        try:
            res = minimize(smooth_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            
            if res.fun < best_overall_S:
                cx_smooth, cy_smooth, beta_smooth = res.x
                # Verify with exact LP
                cx_exact, cy_exact, S_exact = solve_lp_for_shape(pts, beta_smooth)
                if S_exact < best_overall_S:
                    best_overall_S = S_exact
                    best_overall_cx = cx_exact
                    best_overall_cy = cy_exact
                    best_overall_beta = beta_smooth
        except Exception:
            pass

    # Final verification and output
    if best_overall_S == float('inf'):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    inner = []
    for i, (x, y) in enumerate(best_overall_pts):
        angle_deg = 0.0  # All inner hexagons aligned (no rotation needed for lattice packing)
        inner.append([float(x), float(y), angle_deg])
    
    center = [float(best_overall_cx), float(best_overall_cy)]
    side = float(best_overall_S)
    rotation = math.degrees(best_overall_beta)
    
    return inner, center, side, rotation
