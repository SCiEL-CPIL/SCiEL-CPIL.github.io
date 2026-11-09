import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_verts(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _sat_overlap_depth(h1, h2):
    """Minimum penetration depth between two convex polygons using SAT."""
    min_depth = float('inf')
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
    
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
    
    return min_depth

def _compute_penalty(params, outer_side, outer_center, outer_angle):
    """
    params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
    """
    n = len(params) // 3
    inner_verts_list = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Containment penalty
    cont_pen = 0.0
    for iv in inner_verts_list:
        for i in range(6):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % 6]
            edge = v2 - v1
            elen = np.linalg.norm(edge)
            if elen < 1e-12: continue
            for pt in iv:
                rel = pt - v1
                cross = edge[0] * rel[1] - edge[1] * rel[0]
                if cross < 0:
                    cont_pen += (-cross) / elen
    
    # Disjointness penalty
    disj_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                disj_pen += depth
    
    return cont_pen + disj_pen

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial configuration: 3x4 grid
    def get_grid_config():
        positions = []
        # 3 rows, 4 cols
        for r in range(3):
            for c in range(4):
                x = c * np.sqrt(3) + (np.sqrt(3)/2 if r % 2 == 1 else 0)
                y = r * 1.5
                positions.append((x, y))
        
        # Center the grid
        arr = np.array(positions)
        centroid = arr.mean(axis=0)
        arr = arr - centroid
        return arr.tolist()

    configs = []
    grid = get_grid_config()
    configs.append(grid)
    
    # Also try 2x6
    def get_grid_2x6():
        positions = []
        for r in range(2):
            for c in range(6):
                x = c * np.sqrt(3) + (np.sqrt(3)/2 if r % 2 == 1 else 0)
                y = r * 1.5
                positions.append((x, y))
        arr = np.array(positions)
        centroid = arr.mean(axis=0)
        arr = arr - centroid
        return arr.tolist()
    configs.append(get_grid_2x6())
    
    best_score = float('inf')
    best_result = None
    
    outer_angles = [0.0, 30.0]
    
    for outer_angle in outer_angles:
        if time.time() > deadline - 30.0:
            break
        outer_center = (0.0, 0.0)
        
        # Binary search on outer side
        low = 3.4
        high = 5.0
        
        # First, find a valid high
        for config in configs:
            init_params = []
            for (x, y) in config:
                init_params.extend([float(x), float(y), 0.0])
            init_params = np.array(init_params)
            
            # Try to fit in a large hex
            success, opt_params, opt_val = _local_optimize(init_params, high, outer_center, outer_angle)
            if success and opt_val < 1e-6:
                # Found valid config at high
                break
        
        if best_result is None and time.time() > deadline - 20.0:
            # Fallback to a known valid config if optimization fails
            # Return a grid in a large hex
            inner = []
            for (x, y) in grid:
                inner.append([float(x), float(y), 0.0])
            return inner, (0.0, 0.0), 5.0, 0.0

        # Now binary search
        for _ in range(10):
            if time.time() > deadline - 10.0:
                break
            mid = (low + high) / 2
            
            found_valid = False
            for config in configs:
                init_params = []
                for (x, y) in config:
                    init_params.extend([float(x), float(y), 0.0])
                init_params = np.array(init_params)
                
                success, opt_params, opt_val = _local_optimize(init_params, mid, outer_center, outer_angle)
                
                if success and opt_val < 1e-6:
                    high = mid
                    best_result = (opt_params, outer_center, mid, outer_angle)
                    found_valid = True
                    break
            
            if not found_valid:
                low = mid
    
    if best_result is None:
        inner = []
        for (x, y) in grid:
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.0, 0.0
    
    opt_params, outer_center, outer_side, outer_angle = best_result
    
    inner = []
    for i in range(N):
        x = float(opt_params[3*i])
        y = float(opt_params[3*i+1])
        ang = float(opt_params[3*i+2])
        inner.append([x, y, ang])
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)

def _local_optimize(init_params, outer_side, outer_center, outer_angle, max_iter=100):
    def objective(p):
        return _compute_penalty(p, outer_side, outer_center, outer_angle)
    
    try:
        result = minimize(
            objective,
            init_params,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        return True, result.x, result.fun
    except Exception:
        return False, None, float('inf')
