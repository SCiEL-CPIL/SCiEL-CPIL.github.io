import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    # Check if we are already close to the time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate 11 points on a triangular lattice
    # Axial coordinates (q, r)
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    # Use a very small expansion to ensure strict disjointness (no touching)
    # while keeping the size minimal.
    SCALE = 1.0 + 1e-6
    
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0) * SCALE
        y = 1.5 * r * SCALE
        pts.append((x, y))
        
    pts = np.array(pts)
    n_pts = len(pts)
    
    def solve_for_beta(beta):
        """
        Solves the LP for a given outer rotation beta.
        Returns (cx, cy, S) or (None, None, inf)
        """
        # Check time limit
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return None, None, float('inf')
        
        # Normals of the outer hexagon
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Build constraint matrix
        # Variables: cx, cy, D
        # For each normal n_k and point p_i:
        #   (p_i - c) . n_k <= D  =>  -n_kx*cx - n_ky*cy - D <= -(p_i . n_k)
        #   -(p_i - c) . n_k <= D =>  n_kx*cx + n_ky*cy - D <= p_i . n_k
        
        A_ub_list = []
        b_ub_list = []
        
        for nx, ny in normals:
            for px, py in pts:
                dot = px * nx + py * ny
                A_ub_list.append([-nx, -ny, -1.0])
                b_ub_list.append(-dot)
                A_ub_list.append([nx, ny, -1.0])
                b_ub_list.append(dot)
        
        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
        
        c = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                cx = res.x[0]
                cy = res.x[1]
                D = res.x[2]
                S = (D + APOTHEM) / APOTHEM
                return cx, cy, S
            else:
                return None, None, float('inf')
        except Exception:
            return None, None, float('inf')

    # Objective function for minimize_scalar
    def obj(beta):
        _, _, S = solve_for_beta(beta)
        return S

    # Initial guess: try several starting points
    best_beta = 0.0
    best_cx, best_cy, best_S = float('inf'), 0.0, 0.0
    
    # Try a few initial betas to get a good starting point
    for beta_init in [0.0, math.pi/12, math.pi/6, math.pi/18, math.pi/36]:
        cx, cy, S = solve_for_beta(beta_init)
        if S < best_S:
            best_S = S
            best_cx, best_cy, best_beta = cx, cy, beta_init
    
    # If initial solve failed, return fallback
    if best_S == float('inf'):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Minimize over beta in [0, pi/6]
    # The function is convex, so bounded minimization should be robust.
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            # First pass: coarse optimization
            res = minimize_scalar(obj, bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 100, 'xatol': 1e-6})
            if res.fun < best_S:
                best_beta = res.x
                best_cx, best_cy, best_S = solve_for_beta(best_beta)
                
            # Second pass: fine optimization around the best found
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                lo = max(0.0, best_beta - 0.01)
                hi = min(math.pi / 6, best_beta + 0.01)
                if lo < hi:
                    res2 = minimize_scalar(obj, bounds=(lo, hi), method='bounded',
                                         options={'maxiter': 100, 'xatol': 1e-9})
                    if res2.fun < best_S:
                        best_beta = res2.x
                        best_cx, best_cy, best_S = solve_for_beta(best_beta)
                        
    except Exception:
        pass

    # Final high-precision polish: do a few more LP solves around the best beta
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        for delta_beta in [-1e-4, -5e-5, -1e-5, 0.0, 1e-5, 5e-5, 1e-4]:
            beta_test = best_beta + delta_beta
            if 0 <= beta_test <= math.pi / 6:
                cx, cy, S = solve_for_beta(beta_test)
                if S < best_S:
                    best_S = S
                    best_cx, best_cy, best_beta = cx, cy, beta_test

    # Final check
    if best_S == float('inf'):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Prepare output
    inner = []
    angle_deg = math.degrees(best_beta)
    for x, y in pts:
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = angle_deg
    
    return inner, center, side, rotation
