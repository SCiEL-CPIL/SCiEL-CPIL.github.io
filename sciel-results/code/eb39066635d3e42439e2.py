"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
from scipy.optimize import dual_annealing, minimize

N = 11
H = np.sqrt(3) / 2

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)
_n_tris = len(_idx_i)

# Precompute point masks: which triangles involve each point
_point_masks = []
for p in range(N):
    mask = ((_idx_i == p) | (_idx_j == p) | (_idx_k == p))
    _point_masks.append(mask)


def _min_area_fast(points):
    """Compute minimum triangle area among all C(N,3) triples."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    cross = np.abs(dx1 * dy2 - dy1 * dx2)
    return 0.5 * np.min(cross)


def _min_area_and_idx(points):
    """Return (min_area, index_of_worst_triangle)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    cross = np.abs(dx1 * dy2 - dy1 * dx2)
    areas = 0.5 * cross
    idx = int(np.argmin(areas))
    return areas[idx], idx


def _is_inside(points, tol=1e-8):
    """Check if all points are inside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    return (np.all(y >= -tol) and
            np.all(y <= np.sqrt(3) * x + tol) and
            np.all(y <= np.sqrt(3) * (1.0 - x) + tol))


def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle."""
    points = points.copy()
    for i in range(N):
        x, y = points[i]
        y = max(0.0, min(H, y))
        x_min = y / np.sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        if x_min > x_max:
            x, y = 0.5, 0.0
        else:
            x = max(x_min, min(x_max, x))
        points[i] = [x, y]
    return points


def _init_symmetric():
    """Generate a good symmetric starting configuration."""
    points = np.zeros((N, 2))
    row_ys = [0.08 * H, 0.30 * H, 0.55 * H, 0.78 * H]
    row_ns = [4, 3, 3, 1]
    
    pi = 0
    for y, n_pts in zip(row_ys, row_ns):
        x_left = 0.5 * y / H
        x_right = 1.0 - 0.5 * y / H
        width = x_right - x_left
        
        if n_pts == 1:
            points[pi, 0] = 0.5
            points[pi, 1] = y
            pi += 1
        else:
            margin_frac = 0.15
            usable = width * (1 - 2 * margin_frac)
            start = x_left + width * margin_frac
            xs = start + np.linspace(0, usable, n_pts)
            for x in xs:
                points[pi, 0] = x
                points[pi, 1] = y
                pi += 1
    
    return points


def _objective(params):
    """
    Objective function for global optimization.
    Returns negative min area + penalty for boundary violations.
    """
    points = params.reshape(N, 2)
    x = points[:, 0]
    y = points[:, 1]
    
    # Boundary penalties (smooth)
    # y >= 0
    viol1 = np.maximum(0.0, -y)
    # y <= sqrt(3)*x
    viol2 = np.maximum(0.0, y - np.sqrt(3) * x)
    # y <= sqrt(3)*(1-x)
    viol3 = np.maximum(0.0, y - np.sqrt(3) * (1.0 - x))
    
    pen = np.sum(viol1**2) + np.sum(viol2**2) + np.sum(viol3**2)
    
    if pen > 1e-6:
        # Heavy penalty if outside
        return -_min_area_fast(points) + 100.0 * pen
    
    return -_min_area_fast(points)


def _local_refine(points, time_limit):
    """
    Local refinement using worst-triangle strategy.
    Iteratively finds the smallest triangle and moves its vertices to increase the global min area.
    """
    start = time.time()
    best_points = points.copy()
    best_score = _min_area_fast(points)
    
    # We run this loop until time runs out or we stagnate significantly
    # We use Nelder-Mead on the 3 points of the worst triangle
    # We also occasionally try random pairs to escape local minima
    
    for iteration in range(1000):
        if time.time() > time_limit:
            break
        
        min_area, worst_idx = _min_area_and_idx(best_points)
        
        if min_area > best_score:
            best_score = min_area
            
        # If we haven't improved in a while, maybe do a random perturbation?
        # For now, stick to the deterministic worst-triangle approach as it's robust.
        
        i, j, k = _idx_i[worst_idx], _idx_j[worst_idx], _idx_k[worst_idx]
        tri_points = best_points[[i, j, k]].copy()
        combined_mask = _point_masks[i] | _point_masks[j] | _point_masks[k]
        
        def _tri_obj(tri_params):
            tp = tri_params.reshape(3, 2)
            x, y = tp[:, 0], tp[:, 1]
            # Hard check for boundary in local opt (faster than soft penalty for NM)
            if np.any(y < -1e-8) or np.any(y > np.sqrt(3) * x + 1e-8) or np.any(y > np.sqrt(3) * (1 - x) + 1e-8):
                return 1000.0
            
            full = best_points.copy()
            full[i], full[j], full[k] = tp[0], tp[1], tp[2]
            
            # Only compute areas for triangles involving i, j, or k?
            # No, moving i, j, k affects all triangles containing any of them.
            # The mask combined_mask contains all triangles involving i, j, or k.
            # However, the GLOBAL min might be a triangle NOT involving i, j, k.
            # If we only maximize the min of the affected triangles, we might break the global min
            # if the global min was elsewhere and is smaller than the new local min.
            # So we must compute the global min.
            
            p1 = full[_idx_i]
            p2 = full[_idx_j]
            p3 = full[_idx_k]
            dx1 = p2[:, 0] - p1[:, 0]
            dy1 = p2[:, 1] - p1[:, 1]
            dx2 = p3[:, 0] - p1[:, 0]
            dy2 = p3[:, 1] - p1[:, 1]
            cross = np.abs(dx1 * dy2 - dy1 * dx2)
            areas = 0.5 * cross
            return -np.min(areas)
        
        try:
            res = minimize(_tri_obj, tri_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            
            # Check if the result is valid and better
            if res.fun < -min_area + 1e-12:
                new_tri = res.x.reshape(3, 2)
                if _is_inside(new_tri, tol=1e-8):
                    best_points[i], best_points[j], best_points[k] = new_tri[0], new_tri[1], new_tri[2]
                    new_score = _min_area_fast(best_points)
                    if new_score > best_score:
                        best_score = new_score
        except Exception:
            pass
        
        # Every few iterations, try a random pair to diversify
        if iteration % 5 == 0:
            rng = np.random.default_rng(iteration)
            pair = rng.choice(N, size=2, replace=False)
            a, b = int(pair[0]), int(pair[1])
            pair_points = best_points[[a, b]].copy()
            
            def _pair_obj(pair_params):
                pp = pair_params.reshape(2, 2)
                x, y = pp[:, 0], pp[:, 1]
                if np.any(y < -1e-8) or np.any(y > np.sqrt(3) * x + 1e-8) or np.any(y > np.sqrt(3) * (1 - x) + 1e-8):
                    return 1000.0
                full = best_points.copy()
                full[a], full[b] = pp[0], pp[1]
                p1 = full[_idx_i]
                p2 = full[_idx_j]
                p3 = full[_idx_k]
                dx1 = p2[:, 0] - p1[:, 0]
                dy1 = p2[:, 1] - p1[:, 1]
                dx2 = p3[:, 0] - p1[:, 0]
                dy2 = p3[:, 1] - p1[:, 1]
                cross = np.abs(dx1 * dy2 - dy1 * dx2)
                areas = 0.5 * cross
                return -np.min(areas)
            
            try:
                res = minimize(_pair_obj, pair_points.flatten(), method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < -best_score + 1e-12:
                    new_pair = res.x.reshape(2, 2)
                    if _is_inside(new_pair, tol=1e-8):
                        best_points[a], best_points[b] = new_pair[0], new_pair[1]
                        new_score = _min_area_fast(best_points)
                        if new_score > best_score:
                            best_score = new_score
            except Exception:
                pass
                
    return best_points, best_score


def construct():
    start_time = time.time()
    # The evaluator kills at 150s. We aim to return by 145s.
    deadline = start_time + 145
    
    # Phase 1: Global Optimization using Dual Annealing
    # Dual Annealing is robust for this problem.
    # We allocate ~70% of the time to global search.
    global_time_limit = start_time + 90 
    
    bounds = [(0, 1), (0, H)] * N
    
    best_points = _init_symmetric()
    best_score = _min_area_fast(best_points)
    
    # Run dual_annealing with a few different seeds to ensure we find the global basin
    # We can run it in a loop if time permits, but dual_annealing itself does internal restarts.
    # Let's run it once with a high number of iterations, or multiple times with different seeds.
    
    # Strategy: Run DA with seed 42 for a fixed time, then check time.
    # If time remains, run DA with seed 123, etc.
    
    seeds = [42, 123, 456, 789]
    
    for seed in seeds:
        if time.time() > global_time_limit:
            break
            
        try:
            result = dual_annealing(
                _objective,
                bounds,
                maxiter=100, # DA controls its own steps, this is a soft limit
                seed=seed,
                restarts=10, # Number of restarts within DA
                maxfun=5000, # Limit function evaluations to prevent hanging
                xatol=1e-6,
                fatol=1e-8
            )
            
            da_points = result.x.reshape(N, 2)
            da_points = _clamp_to_triangle(da_points)
            da_score = _min_area_fast(da_points)
            
            if da_score > best_score:
                best_score = da_score
                best_points = da_points.copy()
                
        except Exception as e:
            # If DA fails, continue to next seed
            pass
            
    # Phase 2: Local Refinement
    # Use the remaining time for local polish.
    if time.time() < deadline - 5:
        local_time_limit = deadline - 2
        refined_pts, refined_score = _local_refine(best_points, local_time_limit)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts
    
    if best_points is None:
        best_points = _init_symmetric()
        
    return _clamp_to_triangle(best_points)
