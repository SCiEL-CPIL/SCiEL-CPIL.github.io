import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate diverse candidate configurations
    candidates = []
    
    # Ring + 4 outer
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
        (1,1),(-1,2),(-2,1),(2,-1)
    ])
    # Ring + 4 axial
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
        (2,0),(-2,0),(0,2),(0,-2)
    ])
    # 3x3 + 2
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,1),(3,1)
    ])
    # 4x2 + 3
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)
    ])
    # 2x3 + 3
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)
    ])
    # 3x3 + 2 bottom
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)
    ])
    # Cross
    candidates.append([
        (0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)
    ])
    # 5+6
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)
    ])
    # Ring + 4 diagonal
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)
    ])
    # 3x3 + 2 top
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)
    ])
    # 2x4 + 1
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)
    ])
    # 4x1 + 2+2+1
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)
    ])
    # Ring + 4 mixed
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)
    ])
    # Ring + 4 alt
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)
    ])
    # 3x2 + 3x1
    candidates.append([
        (0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)
    ])
    # 2x2 + 4x1
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)
    ])
    # Ring + 4 top
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)
    ])
    # 4x1 + 3x1 + 3x1
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)
    ])
    # 2x2 + 4x1 + 1
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)
    ])
    # Ring + 4
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)
    ])
    # 4x2 + 3x1
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)
    ])
    # 3x2 + 4x1 + 1
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)
    ])
    # Ring + 4
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)
    ])
    # 2x2 + 3x1 + 2x1
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)
    ])
    # 3x2 + 3x1
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)
    ])
    # 4x2 + 2x1
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)
    ])
    # Ring + 4
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)
    ])
    # 2x2 + 3x1 + 2x1
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)
    ])
    # Full hex ring of 6 + center + 4 more = 11 (different outer)
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
        (2,0),(0,2),(-2,2),(2,-2)
    ])
    # 3x3 block + 2 offset
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (1,3),(2,3)
    ])
    # 2x3 + 2x2
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,0),(3,1)
    ])
    # Elongated 5+4
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2)
    ])
    # 3x3 + 2 diagonal
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,0),(3,2)
    ])
    # Compact 3x3 + 2 adjacent
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (0,-1),(1,-1)
    ])
    # Ring + 4 in corners
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
        (2,0),(-2,0),(0,2),(0,-2)
    ])
    # 4x2 + 3 staggered
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
        (1,2),(2,2),(3,2)
    ])
    # 3x3 + 2 L-shape
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,0),(-1,1)
    ])

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def ensure_no_overlap(pts, min_dist=math.sqrt(3) * 1.001):
        """Separate points that are too close to avoid overlap."""
        pts = pts.copy()
        for _ in range(100):
            moved = False
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < min_dist and d > 1e-12:
                        diff = (pts[i] - pts[j]) / d
                        pts[i] += diff * (min_dist - d) / 2.0
                        pts[j] -= diff * (min_dist - d) / 2.0
                        moved = True
                    elif d <= 1e-12:
                        pts[i] += [0.001, 0.001]
                        moved = True
            if not moved:
                break
        return pts

    def exact_S_for_beta(pts, beta):
        """Fast exact computation: find optimal center and D for given beta."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        # projections: (6, n)
        projs = normals @ pts.T
        
        max_projs = projs.max(axis=1)  # (6,)
        min_projs = projs.min(axis=1)  # (6,)
        
        # LP: minimize D
        # s.t. for each k: -nx_k*cx - ny_k*cy - D <= -max_projs[k]
        #      for each k: nx_k*cx + ny_k*cy - D <= min_projs[k]
        #      D >= 0
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        for k in range(6):
            A_ub.append([-normals[k,0], -normals[k,1], -1.0])
            b_ub.append(-max_projs[k])
            A_ub.append([normals[k,0], normals[k,1], -1.0])
            b_ub.append(min_projs[k])
        
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

    def exact_S_direct(pts, beta, cx, cy):
        """Direct evaluation without LP."""
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        centered = pts - np.array([cx, cy])
        projs = normals @ centered.T  # (6, n)
        max_proj = np.max(np.abs(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        return S

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T  # (n, 6)
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 100:
            return float('inf')
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta (72 intervals for better resolution)
        for i in range(73):
            beta = i * (math.pi / 6) / 72
            S, center, D = exact_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                # Search in neighborhood of best coarse beta
                lo = max(0.0, best_beta - math.pi/6/72)
                hi = min(math.pi/6, best_beta + math.pi/6/72)
                res = minimize_scalar(lambda b: exact_S_for_beta(pts, b)[0], 
                                     bounds=(lo, hi), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = exact_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Multi-start Nelder-Mead on exact objective (cx, cy, beta)
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    return exact_S_direct(pts, beta, cx, cy)
                
                # Generate diverse starting points
                centroid = np.mean(pts, axis=0)
                starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.03],
                    [best_center[0], best_center[1], best_beta - 0.03],
                    [best_center[0] + 0.2, best_center[1], best_beta],
                    [best_center[0] - 0.2, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.2, best_beta],
                    [best_center[0], best_center[1] - 0.2, best_beta],
                    [centroid[0], centroid[1], 0.0],
                    [centroid[0], centroid[1], math.pi/12],
                    [centroid[0], centroid[1], math.pi/6],
                ]
                
                for init in starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
        
        # Smooth surrogate refinement with adaptive temperature
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.3, 0.15, 0.08, 0.04, 0.02]
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    def smooth_obj(params, t=temp):
                        cx, cy, beta = params
                        beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                        return smooth_calc_S(pts, beta, cx, cy, temperature=t)
                    
                    res = minimize(smooth_obj, [best_center[0], best_center[1], best_beta],
                                 method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12})
                    if res.fun < 1e10:
                        cx_r, cy_r, beta_r = res.x
                        beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                        S_exact = exact_S_direct(pts, beta_r, cx_r, cy_r)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_center = (cx_r, cy_r)
                            best_beta = beta_r
        except Exception:
            pass
        
        # Final high-precision polish
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    return exact_S_direct(pts, beta, cx, cy)
                
                for init in [[best_center[0], best_center[1], best_beta],
                             [best_center[0], best_center[1], best_beta + 0.005],
                             [best_center[0], best_center[1], best_beta - 0.005]]:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-15})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        pts = ensure_no_overlap(pts)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = ensure_no_overlap(axial_to_cartesian(candidates[0]))
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Use the original points (no scaling to avoid inflation)
    pts_final = best_global_pts
    
    # Compute the exact required side length
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_final[:, 0] - cx) * nx + (pts_final[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
    
    D_final = max_proj
    S_final = (D_final + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_final:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
