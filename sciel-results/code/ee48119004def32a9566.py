import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
SQRT3 = math.sqrt(3)

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Known optimal or near-optimal configurations for 11 hexagons
    # Each is a list of (x, y) positions in a unit hexagonal grid
    candidates = []
    
    # Configuration 1: Central hexagon with 6 around it, plus 4 more
    config1 = [
        (0, 0),
        (SQRT3, 0), (0, 1.5), (-SQRT3/2, 1.5), (-SQRT3, 0), (0, -1.5), (SQRT3/2, -1.5),
        (2*SQRT3, 0), (0, 3), (-SQRT3, 3), (SQRT3, -3)
    ]
    candidates.append(config1)
    
    # Configuration 2: 3-4-3-1 pattern
    config2 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5), (7*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3),
        (SQRT3/2, 4.5)
    ]
    candidates.append(config2)
    
    # Configuration 3: 4-3-4 pattern
    config3 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0), (3*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3), (3*SQRT3, 3)
    ]
    candidates.append(config3)
    
    # Configuration 4: 2-3-3-3 pattern
    config4 = [
        (0, 0), (SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3),
        (SQRT3/2, 4.5), (3*SQRT3/2, 4.5), (5*SQRT3/2, 4.5)
    ]
    candidates.append(config4)
    
    # Configuration 5: Compact diamond
    config5 = [
        (0, 0),
        (SQRT3, 0), (-SQRT3, 0), (0, 1.5), (0, -1.5),
        (2*SQRT3, 0), (-2*SQRT3, 0),
        (SQRT3/2, 3), (-SQRT3/2, 3),
        (SQRT3/2, -3), (-SQRT3/2, -3)
    ]
    candidates.append(config5)
    
    # Configuration 6: 3-5-3 pattern
    config6 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5), (7*SQRT3/2, 1.5), (9*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3)
    ]
    candidates.append(config6)
    
    # Configuration 7: 4-4-3 pattern
    config7 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0), (3*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5), (7*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3)
    ]
    candidates.append(config7)
    
    # Configuration 8: 5-3-3 pattern
    config8 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0), (3*SQRT3, 0), (4*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3)
    ]
    candidates.append(config8)
    
    # Configuration 9: 3-4-4 pattern
    config9 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5), (7*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3), (3*SQRT3, 3)
    ]
    candidates.append(config9)
    
    # Configuration 10: 2-4-5 pattern
    config10 = [
        (0, 0), (SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5), (7*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3), (3*SQRT3, 3), (4*SQRT3, 3)
    ]
    candidates.append(config10)
    
    # Configuration 11: 1-5-5 pattern
    config11 = [
        (0, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5), (7*SQRT3/2, 1.5), (9*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3), (3*SQRT3, 3), (4*SQRT3, 3)
    ]
    candidates.append(config11)
    
    # Configuration 12: 4-5-2 pattern
    config12 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0), (3*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5), (7*SQRT3/2, 1.5), (9*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3)
    ]
    candidates.append(config12)
    
    # Configuration 13: 3-3-3-2 pattern
    config13 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3),
        (SQRT3/2, 4.5), (3*SQRT3/2, 4.5)
    ]
    candidates.append(config13)
    
    # Configuration 14: 2-3-3-3 pattern (shifted)
    config14 = [
        (0, 0), (SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3),
        (SQRT3/2, 4.5), (3*SQRT3/2, 4.5), (5*SQRT3/2, 4.5)
    ]
    candidates.append(config14)
    
    # Configuration 15: 5-4-2 pattern
    config15 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0), (3*SQRT3, 0), (4*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5), (7*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3)
    ]
    candidates.append(config15)
    
    # Configuration 16: 3-4-4 pattern (shifted)
    config16 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5), (7*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3), (3*SQRT3, 3)
    ]
    candidates.append(config16)
    
    # Configuration 17: 4-4-3 pattern (shifted)
    config17 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0), (3*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5), (7*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3)
    ]
    candidates.append(config17)
    
    # Configuration 18: 2-5-4 pattern
    config18 = [
        (0, 0), (SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5), (7*SQRT3/2, 1.5), (9*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3), (3*SQRT3, 3)
    ]
    candidates.append(config18)
    
    # Configuration 19: 1-6-4 pattern
    config19 = [
        (0, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5), (7*SQRT3/2, 1.5), (9*SQRT3/2, 1.5), (11*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3), (3*SQRT3, 3)
    ]
    candidates.append(config19)
    
    # Configuration 20: 3-3-5 pattern
    config20 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3), (3*SQRT3, 3), (4*SQRT3, 3)
    ]
    candidates.append(config20)
    
    # Configuration 21: 4-3-4 pattern
    config21 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0), (3*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3), (3*SQRT3, 3)
    ]
    candidates.append(config21)
    
    # Configuration 22: 5-2-4 pattern
    config22 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0), (3*SQRT3, 0), (4*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3), (3*SQRT3, 3)
    ]
    candidates.append(config22)
    
    # Configuration 23: 2-4-5 pattern (shifted)
    config23 = [
        (0, 0), (SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5), (7*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3), (3*SQRT3, 3), (4*SQRT3, 3)
    ]
    candidates.append(config23)
    
    # Configuration 24: 3-5-3 pattern (shifted)
    config24 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5), (7*SQRT3/2, 1.5), (9*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3)
    ]
    candidates.append(config24)
    
    # Configuration 25: 4-4-3 pattern (different shift)
    config25 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0), (3*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5), (7*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3)
    ]
    candidates.append(config25)
    
    # Configuration 26: 5-3-3 pattern (shifted)
    config26 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0), (3*SQRT3, 0), (4*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3)
    ]
    candidates.append(config26)
    
    # Configuration 27: 3-4-4 pattern (different shift)
    config27 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5), (7*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3), (3*SQRT3, 3)
    ]
    candidates.append(config27)
    
    # Configuration 28: 2-5-4 pattern (shifted)
    config28 = [
        (0, 0), (SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5), (7*SQRT3/2, 1.5), (9*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3), (3*SQRT3, 3)
    ]
    candidates.append(config28)
    
    # Configuration 29: 1-6-4 pattern (shifted)
    config29 = [
        (0, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5), (7*SQRT3/2, 1.5), (9*SQRT3/2, 1.5), (11*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3), (3*SQRT3, 3)
    ]
    candidates.append(config29)
    
    # Configuration 30: 3-3-5 pattern (shifted)
    config30 = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5),
        (0, 3), (SQRT3, 3), (2*SQRT3, 3), (3*SQRT3, 3), (4*SQRT3, 3)
    ]
    candidates.append(config30)
    
    # Precompute normals for overlap checking
    OVERLAP_NORMALS = np.array([
        [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
        for k in range(6)
    ])
    
    def check_overlap(pts):
        n_pts = len(pts)
        if n_pts < 2:
            return True
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        projs = np.abs(diffs @ OVERLAP_NORMALS.T)
        min_projs = np.min(projs, axis=2)
        iu = np.triu_indices(n_pts, k=1)
        return np.all(min_projs[iu] >= SQRT3 - 1e-9)
    
    def calc_S(pts, beta, cx, cy):
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        max_proj = np.max(np.abs(projs))
        D = max_proj
        S = (D + APOTHEM) / APOTHEM
        return S, D
    
    def optimize_config(pts_init):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        best_pts = pts_init
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None, None
        
        # Try different beta values
        for i in range(12):
            beta = i * (math.pi / 6) / 11
            # Center the configuration
            cx = np.mean(pts_init[:, 0])
            cy = np.mean(pts_init[:, 1])
            S, D = calc_S(pts_init, beta, cx, cy)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = (cx, cy)
        
        # Local optimization: adjust positions to minimize S
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def objective(params):
                n_pts = len(pts_init)
                pts = params[:2*n_pts].reshape(n_pts, 2)
                cx = params[2*n_pts]
                cy = params[2*n_pts+1]
                beta = params[2*n_pts+2]
                beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                
                S, D = calc_S(pts, beta, cx, cy)
                
                # Overlap penalty
                penalty = 0.0
                diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                projs = np.abs(diffs @ OVERLAP_NORMALS.T)
                min_projs = np.min(projs, axis=2)
                iu = np.triu_indices(n_pts, k=1)
                violations = np.maximum(0, SQRT3 - min_projs[iu])
                penalty = np.sum(violations) * 100.0
                
                return S + penalty
            
            init_params = np.concatenate([pts_init.ravel(), [best_center[0], best_center[1], best_beta]])
            
            try:
                res = minimize(objective, init_params, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
                
                if res.fun < best_S + 0.1:
                    n_pts = len(pts_init)
                    candidate_pts = res.x[:2*n_pts].reshape(n_pts, 2)
                    candidate_center = (res.x[2*n_pts], res.x[2*n_pts+1])
                    candidate_beta = ((res.x[2*n_pts+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    
                    if check_overlap(candidate_pts):
                        S_exact, _ = calc_S(candidate_pts, candidate_beta, candidate_center[0], candidate_center[1])
                        if S_exact < best_S:
                            best_S = S_exact
                            best_pts = candidate_pts
                            best_center = candidate_center
                            best_beta = candidate_beta
            except Exception:
                pass
        
        return best_S, best_beta, best_center, best_pts
    
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    
    # Sort candidates by quick heuristic
    def quick_score(pts_list):
        pts = np.array(pts_list)
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        return (max_proj + APOTHEM) / APOTHEM
    
    scored_candidates = [(quick_score(c), idx, c) for idx, c in enumerate(candidates)]
    scored_candidates.sort(key=lambda x: x[0])
    sorted_candidates = [c for _, _, c in scored_candidates]
    
    for config in sorted_candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = np.array(config)
        S, beta, center, optimized_pts = optimize_config(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = optimized_pts
            best_global_beta = beta
            best_global_center = center
    
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = np.array(candidates[0])
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
    
    # Scale slightly to ensure no overlap
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
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
