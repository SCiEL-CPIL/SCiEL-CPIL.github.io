import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: return a valid but large packing
        # 11 hexes in a 3x3 grid plus 2
        # Just return a known valid config if time runs out
        # For now, return a trivial one to avoid crash
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate candidates
    # Axial coordinates
    # We want compact shapes.
    # Let's use the ones from the previous program but ensure they are good.
    # And add a few more compact ones.
    
    candidates = []
    
    # A compact 11-hex shape:
    # Center (0,0)
    # 6 neighbors
    # 4 more from the next ring, chosen to be compact.
    # E.g., (2,0), (1,1), (0,2), (-1,2) ?
    # Let's just use a set of known good polyhexes.
    
    # Shape 1: "Diamond"
    c1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    candidates.append(c1)
    
    # Shape 2: "Hexagon with 2 corners"
    c2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(c2)
    
    # Shape 3: "Bar"
    c3 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(c3)
    
    # Shape 4: "L-shape"
    c4 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(c4)
    
    # Shape 5: "T-shape"
    c5 = [
        (0, 0), (1, 0), (-1, 0),
        (0, 1), (1, 1), (-1, 1),
        (0, 2), (1, 2), (-1, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(c5)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        """
        Calculate minimal S for fixed beta.
        Returns S, center, D.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                # -nx*cx - ny*cy - D <= -n.p
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                # nx*cx + ny*cy - D <= n.p
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

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
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
            return float('inf'), (cx, cy), float('inf')
        
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # 1. Coarse grid search over beta
        for i in range(61):
            beta = i * (math.pi / 6) / 60
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # 2. Refine beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 1000, 'xatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # 3. Smooth surrogate optimization for (cx, cy, beta)
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.5, 0.2, 0.1, 0.05]
                start_betas = [best_beta, best_beta + 0.1, best_beta - 0.1, 0.0, math.pi/12]
                
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        
                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=t)
                            return S
                        
                        init = [best_center[0], best_center[1], sb]
                        res = minimize(smooth_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
                        
                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = c_exact
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Evaluate candidates
    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback
        pts = axial_to_cartesian(candidates[0])
        best_global_center = (0.0, 0.0)
        best_global_beta = 0.0
        # Calculate S manually
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Scale slightly to ensure validity? No, we want min S.
    # The LP ensures D is minimal.
    # We return the points and the outer hex params.
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
