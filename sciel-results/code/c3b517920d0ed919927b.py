import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog, minimize

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
    # This is a compact "flower" shape: center + 6 neighbors + 4 from the next ring
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
        
        # Variables: cx, cy, D
        # D is the max projection of the *centers* of the inner hexagons
        # relative to the outer center, in the normal directions.
        # Constraint: (p_i - c) . n_k <= D  AND  -(p_i - c) . n_k <= D
        # Which means | (p_i - c) . n_k | <= D
        
        # The outer hexagon must contain the inner hexagons.
        # Inner hexagon support in direction n_k is APOTHEM (since aligned).
        # So, (p_i - c) . n_k + APOTHEM <= S * APOTHEM
        # (p_i - c) . n_k <= S * APOTHEM - APOTHEM
        # Let D = S * APOTHEM - APOTHEM.
        # Then | (p_i - c) . n_k | <= D.
        # We minimize D.
        # S = (D + APOTHEM) / APOTHEM
        
        c = np.array([0.0, 0.0, 1.0]) # Minimize D
        
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                # (px - cx)nx + (py - cy)ny <= D
                # px nx - cx nx + py ny - cy ny - D <= 0
                # -nx cx - ny cy - D <= -(px nx + py ny)
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                
                # -((px - cx)nx + (py - cy)ny) <= D
                # (cx - px)nx + (cy - py)ny - D <= 0
                # nx cx + ny cy - D <= px nx + py ny
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds
        # cx, cy are unconstrained in theory, but bounded by the spread of pts
        # D >= 0
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

    def smooth_surrogate(beta, cx, cy, D, temperature=0.01):
        """
        Smooth approximation of the min/max objective using log-sum-exp.
        Returns a smooth value approximating S = (D + APOTHEM) / APOTHEM
        where D = max over all (i,k) of |(p_i - c).n_k|
        """
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Compute all projections
        projections = []
        for nx, ny in normals:
            for px, py in pts:
                proj = (px - cx) * nx + (py - cy) * ny
                projections.append(proj)
                projections.append(-proj)
        
        projections = np.array(projections)
        
        # Log-sum-exp for smooth max
        max_proj = np.max(projections)
        if max_proj > 0:
            log_sum_exp = max_proj + math.log(np.sum(np.exp(projections - max_proj)))
        else:
            log_sum_exp = 0.0
        
        # Smooth D approximation
        D_smooth = log_sum_exp
        
        S_smooth = (D_smooth + APOTHEM) / APOTHEM
        return S_smooth

    # Objective function for minimize_scalar
    def obj(beta):
        _, _, S = solve_for_beta(beta)
        return S

    # Initial guess: beta = 0
    best_beta = 0.0
    best_cx, best_cy, best_S = solve_for_beta(best_beta)
    
    # If initial solve failed, try a small offset or return fallback
    if best_S == float('inf'):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Minimize over beta in [0, pi/6]
    # The function is convex, so bounded minimization should be robust.
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            # Use tighter tolerance and more iterations for better precision
            res = minimize_scalar(obj, bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 500, 'xatol': 1e-10})
            if res.fun < best_S:
                best_beta = res.x
                best_cx, best_cy, best_S = solve_for_beta(best_beta)
                
                # Second pass with even tighter tolerance around the found minimum
                if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                    local_bounds = (max(0, best_beta - 0.01), min(math.pi / 6, best_beta + 0.01))
                    res2 = minimize_scalar(obj, bounds=local_bounds, method='bounded',
                                          options={'maxiter': 1000, 'xatol': 1e-12})
                    if res2.fun < best_S:
                        best_beta = res2.x
                        best_cx, best_cy, best_S = solve_for_beta(best_beta)
    except Exception:
        pass

    # Nelder-Mead polish on (beta, cx, cy, D) using smooth surrogate
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        def nm_objective(vars):
            beta, cx, cy, D = vars
            # Use smooth surrogate for speed
            return smooth_surrogate(beta, cx, cy, D, temperature=0.01)
        
        x0 = np.array([best_beta, best_cx, best_cy, best_S * APOTHEM - APOTHEM])
        try:
            nm_res = minimize(nm_objective, x0, method='Nelder-Mead',
                             options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500, 'maxfev': 500})
            if nm_res.fun < best_S:
                # Verify with exact LP
                beta_nm = nm_res.x[0]
                cx_nm = nm_res.x[1]
                cy_nm = nm_res.x[2]
                D_nm = nm_res.x[3]
                
                # Solve exact LP at this beta to get precise center
                _, _, S_exact = solve_for_beta(beta_nm)
                if S_exact < best_S:
                    best_beta = beta_nm
                    best_cx, best_cy, best_S = solve_for_beta(best_beta)
        except Exception:
            pass

    # Final check
    if best_S == float('inf'):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # High-precision polish: refine beta with very tight tolerance
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        try:
            # Try multiple starting points near the current best
            for offset in [-0.005, -0.002, 0.0, 0.002, 0.005]:
                beta_try = best_beta + offset
                if beta_try < 0 or beta_try > math.pi / 6:
                    continue
                _, _, S_try = solve_for_beta(beta_try)
                if S_try < best_S:
                    best_S = S_try
                    best_beta = beta_try
                    best_cx, best_cy, _ = solve_for_beta(best_beta)
            
            # Final tight minimization
            local_bounds = (max(0, best_beta - 0.005), min(math.pi / 6, best_beta + 0.005))
            res_final = minimize_scalar(obj, bounds=local_bounds, method='bounded',
                                       options={'maxiter': 2000, 'xatol': 1e-14})
            if res_final.fun < best_S:
                best_beta = res_final.x
                best_cx, best_cy, best_S = solve_for_beta(best_beta)
        except Exception:
            pass

    # Prepare output
    inner = []
    angle_deg = math.degrees(best_beta)
    for x, y in pts:
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = angle_deg
    
    return inner, center, side, rotation
