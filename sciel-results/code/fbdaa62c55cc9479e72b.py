import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar, differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    candidates = []
    seen_configs = set()

    def add_candidate(axial_list):
        if len(axial_list) != N:
            return
        min_q = min(q for q, r in axial_list)
        min_r = min(r for q, r in axial_list)
        normalized = frozenset((q - min_q, r - min_r) for q, r in axial_list)
        if normalized not in seen_configs:
            seen_configs.add(normalized)
            candidates.append(list(axial_list))

    # --- Generate a wide variety of 11-ominoes ---
    
    # 1. Standard Honeycomb Clusters (Radius 1 + 5 more)
    # Center (0,0), Ring 1 (6), Ring 2 (12). We need 11.
    # So we take Center + 5 from Ring 1? No, that's disconnected or sparse.
    # Usually, compact clusters are better.
    
    # Let's define neighbors to grow clusters
    def get_neighbors(q, r):
        return [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]
    
    # Generate all connected 11-ominoes via BFS from a seed, limiting to a small bounding box
    # This might be too slow if done naively, so we use a heuristic set + programmatic expansion
    
    # Base shapes
    base_shapes = [
        # 3x3 block + 2
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)],
        # 4x3 minus corners
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        # Staircase
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        # Compact 3x3+2
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        # Star
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)],
        # Elongated
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        # Asymmetric star
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)],
        # Compact with top
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        # Wide base
        [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)],
        # Tall narrow
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)],
        # Star with asymmetric outer
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)],
        # Star with diagonal
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)],
        # Dense cluster
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)],
        # Zigzag
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)],
        # Star with top cluster
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)],
        # Trapezoid
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)],
        # Wide top
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)],
        # Star with inner diagonal
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)],
        # 4x3 variant
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        # Wide with top
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        # Symmetric star
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        # Very tall
        [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)],
        # 3x3+row
        [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        # Dense 4-wide
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)],
        # Star with asymmetric bottom
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)],
        # Tall with top row
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)],
        # Additional
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(3,3)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(1,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(1,3),(2,3)],
    ]
    
    for shape in base_shapes:
        add_candidate(shape)
        
    # Generate more by "mutating" the 3x3 block
    # 3x3 block is 9 hexes. Add 2 more.
    base_3x3 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2)]
    base_set = set(base_3x3)
    
    # Possible additions to 3x3
    additions = [
        [(-1,0), (-1,1)],
        [(-1,1), (-1,2)],
        [(-1,2), (0,3)],
        [(0,3), (1,3)],
        [(1,3), (2,3)],
        [(2,3), (3,2)],
        [(3,2), (3,1)],
        [(3,1), (3,0)],
        [(3,0), (2,-1)],
        [(2,-1), (1,-1)],
        [(1,-1), (0,-1)],
        [(0,-1), (-1,0)],
        [(-1,0), (-2,0)], # disconnected? no, (-1,0) touches (0,0)
        [(-1,1), (-2,1)],
        [(-2,1), (-2,0)],
        [(-2,0), (-1,-1)],
        [(-1,-1), (0,-2)],
        [(0,-2), (1,-2)],
        [(1,-2), (2,-1)],
        [(2,-1), (3,-1)],
        [(3,-1), (3,0)],
        [(3,0), (4,0)],
        [(4,0), (4,1)],
        [(4,1), (3,2)],
        [(3,2), (4,2)],
        [(4,2), (5,1)],
        [(5,1), (5,0)],
        [(5,0), (4,-1)],
        [(4,-1), (3,-1)],
        [(3,-1), (3,0)],
        [(0,3), (-1,3)],
        [(-1,3), (-1,2)],
        [(-1,2), (-2,2)],
        [(-2,2), (-2,1)],
        [(-2,1), (-3,1)],
        [(-3,1), (-3,0)],
        [(-3,0), (-2,-1)],
        [(-2,-1), (-1,-1)],
        [(-1,-1), (0,-1)],
        [(0,-1), (1,-1)],
        [(1,-1), (2,-1)],
        [(2,-1), (3,-1)],
        [(3,-1), (3,0)],
        [(3,0), (3,1)],
        [(3,1), (3,2)],
        [(3,2), (2,3)],
        [(2,3), (1,3)],
        [(1,3), (0,3)],
        [(0,3), (-1,3)],
        [(-1,3), (-2,3)],
        [(-2,3), (-2,2)],
        [(-2,2), (-3,2)],
        [(-3,2), (-3,1)],
        [(-3,1), (-4,1)],
        [(-4,1), (-4,0)],
        [(-4,0), (-3,-1)],
        [(-3,-1), (-2,-1)],
        [(-2,-1), (-1,-1)],
        [(-1,-1), (0,-1)],
        [(0,-1), (1,-1)],
        [(1,-1), (2,-1)],
        [(2,-1), (3,-1)],
    ]
    
    for add1, add2 in additions:
        shape = list(base_3x3) + [add1, add2]
        # Check connectivity
        shape_set = set(shape)
        if len(shape_set) == 11:
            # Simple connectivity check: BFS from first node
            visited = set()
            queue = [shape[0]]
            visited.add(shape[0])
            while queue:
                cur = queue.pop(0)
                for nb in get_neighbors(*cur):
                    if nb in shape_set and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            if len(visited) == 11:
                add_candidate(shape)

    # Optimization Functions
    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])

        n_pts = len(pts)
        n_constraints = 12 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)

        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            start = 2 * k * n_pts
            A_ub[start:start+n_pts, 0] = -nx
            A_ub[start:start+n_pts, 1] = -ny
            A_ub[start:start+n_pts, 2] = -1.0
            b_ub[start:start+n_pts] = -projs
            A_ub[start+n_pts:start+2*n_pts, 0] = nx
            A_ub[start+n_pts:start+2*n_pts, 1] = ny
            A_ub[start+n_pts:start+2*n_pts, 2] = -1.0
            b_ub[start+n_pts:start+2*n_pts] = projs

        c_obj = np.array([0.0, 0.0, 1.0])
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
        
        # Coarse grid search over beta (0 to pi/6)
        # Use more points for better initial guess
        n_grid = 100
        for i in range(n_grid):
            beta = i * (math.pi / 6) / (n_grid - 1)
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                lo = max(0, best_beta - math.pi/60)
                hi = min(math.pi/6, best_beta + math.pi/60)
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(lo, hi), method='bounded',
                                     options={'maxiter': 1000, 'xatol': 1e-15})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # High-precision Newton polish for center at optimal rotation
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                normals = []
                for k in range(6):
                    ang = best_beta + math.pi / 6.0 + k * math.pi / 3.0
                    normals.append(np.array([math.cos(ang), math.sin(ang)]))
                
                def compute_max_and_active(cx, cy):
                    max_proj = 0.0
                    active = []
                    for i, (px, py) in enumerate(pts):
                        for j, n in enumerate(normals):
                            proj = (px - cx) * n[0] + (py - cy) * n[1]
                            if proj > max_proj:
                                max_proj = proj
                                active = [(i, j, 1.0)]
                            elif proj < -max_proj:
                                max_proj = -proj
                                active = [(i, j, -1.0)]
                            elif abs(proj) > max_proj - 1e-12:
                                if proj > 0:
                                    active.append((i, j, 1.0))
                                else:
                                    active.append((i, j, -1.0))
                    return max_proj, active
                
                cx, cy = best_center
                for it in range(100):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    D, active = compute_max_and_active(cx, cy)
                    if len(active) == 0:
                        break
                    
                    gx = 0.0
                    gy = 0.0
                    for (i, j, sign) in active:
                        gx += sign * normals[j][0]
                        gy += sign * normals[j][1]
                    gx /= len(active)
                    gy /= len(active)
                    
                    norm_g = math.sqrt(gx*gx + gy*gy)
                    if norm_g < 1e-14:
                        break
                    
                    alpha = 1.0
                    for _ in range(50):
                        new_cx = cx - alpha * gx
                        new_cy = cy - alpha * gy
                        new_D, _ = compute_max_and_active(new_cx, new_cy)
                        if new_D < D - 1e-15:
                            cx, cy = new_cx, new_cy
                            break
                        alpha *= 0.5
                    else:
                        break
                
                final_D, _ = compute_max_and_active(cx, cy)
                S_newton = (final_D + APOTHEM) / APOTHEM
                if S_newton < best_S:
                    best_S = S_newton
                    best_center = (cx, cy)
        except Exception:
            pass
        
        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Limit number of candidates to evaluate if we have too many
    # We have ~100 candidates now.
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
        best_global_pts = axial_to_cartesian(candidates[0])
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
