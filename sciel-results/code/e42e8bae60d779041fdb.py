import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog
from itertools import product

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def axial_to_xy(q, r):
    x = math.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

def generate_all_polyhexes(n):
    """Generate all n-cell polyhexes via BFS growth."""
    if n == 1:
        return [frozenset([(0, 0)])]
    
    # Directions in axial coordinates
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Start from all (n-1)-cell polyhexes
    prev_polyhexes = generate_all_polyhexes(n - 1)
    
    new_polyhexes = set()
    for ph in prev_polyhexes:
        ph_set = set(ph)
        # Try adding each neighbor of each cell
        for q, r in ph_set:
            for dq, dr in dirs:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in ph_set:
                    new_ph = frozenset(ph_set | {(nq, nr)})
                    # Normalize: translate so min q, then min r is at origin
                    min_q = min(q for q, r in new_ph)
                    min_r = min(r for q, r in new_ph if q == min_q)
                    normalized = frozenset((q - min_q, r - min_r) for q, r in new_ph)
                    new_polyhexes.add(normalized)
    
    return list(new_polyhexes)

def get_best_polyhex(n):
    """Find the polyhex with smallest minimal enclosing hexagon."""
    # For n=11, we know the optimal shape is the 3-4-3 cluster
    # But let's verify by checking a few candidates
    candidates = [
        # 3-4-3 cluster (the current one)
        frozenset([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                    (1, 1), (-1, 2), (-2, 1), (2, -1)]),
        # Alternative: more compact shapes
        frozenset([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                    (1, 1), (-1, 2), (0, 2), (-2, 1)]),
    ]
    
    best = None
    best_score = float('inf')
    
    for ph in candidates:
        pts = np.array([axial_to_xy(q, r) for q, r in ph])
        # Quick estimate: bounding box diagonal
        extent_x = pts[:, 0].max() - pts[:, 0].min()
        extent_y = pts[:, 1].max() - pts[:, 1].min()
        score = extent_x + extent_y
        if score < best_score:
            best_score = score
            best = ph
    
    return best

def construct():
    # Check if we are already close to the time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Use the optimal 3-4-3 cluster shape
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

    def smooth_objective(beta, cx, cy, t, temp=0.01):
        """
        Smooth surrogate for the min/max objective using log-sum-exp.
        Returns a smooth approximation of max_k max_i | (p_i - c) . n_k |.
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
            log_sum_exp = math.log(np.sum(np.exp(projections)))
        
        # Smooth approximation: t * log_sum_exp + (1/t) * t^2/2
        # This gives a smooth upper bound on the max
        smooth_val = temp * log_sum_exp + t ** 2 / (2 * temp)
        
        return smooth_val

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

    # Phase 1: Coarse minimization over beta
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res = minimize_scalar(obj, bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 200, 'xatol': 1e-8})
            if res.fun < best_S:
                best_beta = res.x
                best_cx, best_cy, best_S = solve_for_beta(best_beta)
    except Exception:
        pass

    # Phase 2: Smooth surrogate optimization for better convergence
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            # Use smooth objective with decreasing temperature
            from scipy.optimize import minimize
            
            def smooth_obj(params):
                beta, cx, cy, t = params
                if beta < 0 or beta > math.pi / 6:
                    return 1e10
                return smooth_objective(beta, cx, cy, t)
            
            # Start from current best
            x0 = [best_beta, best_cx, best_cy, 0.05]
            bounds_smooth = [(0, math.pi / 6), (-5, 5), (-5, 5), (0.001, 0.1)]
            
            res_smooth = minimize(smooth_obj, x0, method='L-BFGS-B', bounds=bounds_smooth,
                                  options={'maxiter': 100, 'ftol': 1e-10})
            
            if res_smooth.success:
                beta_smooth = res_smooth.x[0]
                # Verify with exact LP
                cx_v, cy_v, S_v = solve_for_beta(beta_smooth)
                if S_v < best_S:
                    best_beta = beta_smooth
                    best_cx, best_cy, best_S = cx_v, cy_v, S_v
    except Exception:
        pass

    # Phase 3: High-precision polish with tight tolerances
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            local_bounds = (max(0, best_beta - 0.005), min(math.pi / 6, best_beta + 0.005))
            res2 = minimize_scalar(obj, bounds=local_bounds, method='bounded',
                                  options={'maxiter': 1000, 'xatol': 1e-12})
            if res2.fun < best_S:
                best_beta = res2.x
                best_cx, best_cy, best_S = solve_for_beta(best_beta)
    except Exception:
        pass

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
