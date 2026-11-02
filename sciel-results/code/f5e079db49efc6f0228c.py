import math
import numpy as np
import time
import itertools
from scipy.optimize import linprog, minimize
from collections import deque

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate all hexagons within radius 2
    # Axial coordinates (q, r)
    all_hexes = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                all_hexes.append((q, r))
    
    # Ensure origin is included
    origin = (0, 0)
    if origin not in all_hexes:
        all_hexes.append(origin)
        
    # Separate origin and others
    others = [h for h in all_hexes if h != origin]
    
    # Generate all combinations of 10 others to form 11 total
    # 19 total hexes, 1 origin, 18 others. Choose 10 from 18.
    # C(18, 10) = 43758. This is small.
    
    candidates = []
    
    # Pre-compute neighbors for connectivity check
    neighbors = {}
    for q, r in all_hexes:
        nbs = []
        for dq, dr in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
            nbs.append((q+dq, r+dr))
        neighbors[(q, r)] = nbs

    def is_connected(hex_set):
        # hex_set is a set of (q, r)
        if not hex_set:
            return False
        start = next(iter(hex_set))
        visited = {start}
        queue = deque([start])
        while queue:
            curr = queue.popleft()
            for nb in neighbors.get(curr, []):
                if nb in hex_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return len(visited) == len(hex_set)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Generate candidates
    combo_count = 0
    for combo in itertools.combinations(others, 10):
        combo_count += 1
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        hex_set = set(list(combo) + [origin])
        
        if not is_connected(hex_set):
            continue
            
        candidates.append(hex_set)
        
        # Optional: Early break if we have enough diverse candidates?
        # No, let's get all connected ones first, then filter.
        # 43k is fast.

    # Filter candidates by a quick score (lattice-aligned bounding box)
    def quick_score(hex_set):
        pts = axial_to_cartesian(list(hex_set))
        # Center at centroid for quick check
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        pts_centered = pts - np.array([cx, cy])
        
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts_centered[:, 0] * nx + pts_centered[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        return S

    scored_candidates = []
    for hex_set in candidates:
        s = quick_score(hex_set)
        scored_candidates.append((s, hex_set))
        
    # Sort by quick score
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top 50 for detailed optimization
    top_candidates = [x[1] for x in scored_candidates[:50]]
    
    if not top_candidates:
        # Fallback: use the first candidate or a default
        top_candidates = [candidates[0]] if candidates else [{origin}]

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
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

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta
        for i in range(12):
            beta = i * (math.pi / 6) / 11
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 100, 'xatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Nelder-Mead polish on (cx, cy, beta)
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    S, _, _ = calc_S_for_beta(pts, beta)
                    return S
                
                init = [best_center[0], best_center[1], best_beta]
                res = minimize(objective, init, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12})
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

    for hex_set in top_candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(list(hex_set))
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback
        pts = axial_to_cartesian(list(top_candidates[0]))
        best_global_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM
        best_global_pts = pts

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Scale to ensure no overlap (safety)
    SCALE = 1.0 + 1e-10
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
