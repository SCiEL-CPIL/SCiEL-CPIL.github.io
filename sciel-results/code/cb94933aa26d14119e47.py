"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

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


def _init_lattice(rng):
    """Generate a lattice-based initialization with random perturbation."""
    points = []
    spacing = 0.25
    for row in range(8):
        y = row * spacing * np.sin(np.pi / 3)
        if y > H:
            break
        x_left = 0.5 * y / H
        x_right = 1.0 - 0.5 * y / H
        if row % 2 == 0:
            x_start = x_left + 0.1 * spacing
        else:
            x_start = x_left + 0.1 * spacing + spacing / 2
        x = x_start
        while x <= x_right - 0.05:
            if 0 <= x <= 1 and 0 <= y <= H:
                points.append([x, y])
            x += spacing
    
    if len(points) > N:
        idx = rng.choice(len(points), size=N, replace=False)
        points = [points[i] for i in idx]
    elif len(points) < N:
        while len(points) < N:
            u = rng.random()
            v = rng.random()
            if u + v <= 1:
                px = v + 0.5 * (1 - u - v)
                py = H * (1 - u - v)
                points.append([px, py])
    
    pts = np.array(points[:N])
    pts += rng.normal(0, 0.02, size=pts.shape)
    return _clamp_to_triangle(pts)


def _coarse_to_fine(points, rng, time_limit):
    """Iteratively refine: find worst triangle, do local search, repeat."""
    start = time.time()
    best_points = points.copy()
    best_score = _min_area_fast(points)
    
    for iteration in range(500):
        if time.time() > time_limit:
            break
        
        min_area, worst_idx = _min_area_and_idx(best_points)
        
        if min_area > best_score:
            best_score = min_area
        
        i, j, k = _idx_i[worst_idx], _idx_j[worst_idx], _idx_k[worst_idx]
        tri_points = best_points[[i, j, k]].copy()
        combined_mask = _point_masks[i] | _point_masks[j] | _point_masks[k]
        
        def _tri_obj(tri_params):
            tp = tri_params.reshape(3, 2)
            x, y = tp[:, 0], tp[:, 1]
            if np.any(y < -1e-6) or np.any(y > np.sqrt(3) * x + 1e-6) or np.any(y > np.sqrt(3) * (1 - x) + 1e-6):
                return 1000.0
            full = best_points.copy()
            full[i], full[j], full[k] = tp[0], tp[1], tp[2]
            p1 = full[_idx_i]
            p2 = full[_idx_j]
            p3 = full[_idx_k]
            dx1 = p2[:, 0] - p1[:, 0]
            dy1 = p2[:, 1] - p1[:, 1]
            dx2 = p3[:, 0] - p1[:, 0]
            dy2 = p3[:, 1] - p1[:, 1]
            cross = np.abs(dx1 * dy2 - dy1 * dx2)
            areas = 0.5 * cross
            return -np.min(areas[combined_mask])
        
        try:
            res = minimize(_tri_obj, tri_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < -min_area + 1e-10:
                new_tri = res.x.reshape(3, 2)
                if _is_inside(new_tri, tol=1e-8):
                    best_points[i], best_points[j], best_points[k] = new_tri[0], new_tri[1], new_tri[2]
                    new_score = _min_area_fast(best_points)
                    if new_score > best_score:
                        best_score = new_score
        except Exception:
            pass
        
        if iteration % 3 == 0:
            pair = rng.choice(N, size=2, replace=False)
            a, b = int(pair[0]), int(pair[1])
            pair_points = best_points[[a, b]].copy()
            pair_mask = _point_masks[a] | _point_masks[b]
            
            def _pair_obj(pair_params):
                pp = pair_params.reshape(2, 2)
                x, y = pp[:, 0], pp[:, 1]
                if np.any(y < -1e-6) or np.any(y > np.sqrt(3) * x + 1e-6) or np.any(y > np.sqrt(3) * (1 - x) + 1e-6):
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
                return -np.min(areas[pair_mask])
            
            try:
                res = minimize(_pair_obj, pair_points.flatten(), method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < -_min_area_fast(best_points) + 1e-10:
                    new_pair = res.x.reshape(2, 2)
                    if _is_inside(new_pair, tol=1e-8):
                        best_points[a], best_points[b] = new_pair[0], new_pair[1]
                        new_score = _min_area_fast(best_points)
                        if new_score > best_score:
                            best_score = new_score
            except Exception:
                pass
    
    return best_points, best_score


def _final_polish(points, time_limit):
    """High-precision final polish using tight Nelder-Mead on all coordinates."""
    start = time.time()
    best_points = points.copy()
    best_score = _min_area_fast(points)
    
    # Run multiple rounds of tight local optimization
    for round_num in range(5):
        if time.time() > time_limit:
            break
        
        # Find worst triangle and optimize those 3 points with very tight tolerance
        min_area, worst_idx = _min_area_and_idx(best_points)
        i, j, k = _idx_i[worst_idx], _idx_j[worst_idx], _idx_k[worst_idx]
        tri_points = best_points[[i, j, k]].copy()
        combined_mask = _point_masks[i] | _point_masks[j] | _point_masks[k]
        
        def _tri_obj(tri_params):
            tp = tri_params.reshape(3, 2)
            x, y = tp[:, 0], tp[:, 1]
            if np.any(y < -1e-6) or np.any(y > np.sqrt(3) * x + 1e-6) or np.any(y > np.sqrt(3) * (1 - x) + 1e-6):
                return 1000.0
            full = best_points.copy()
            full[i], full[j], full[k] = tp[0], tp[1], tp[2]
            p1 = full[_idx_i]
            p2 = full[_idx_j]
            p3 = full[_idx_k]
            dx1 = p2[:, 0] - p1[:, 0]
            dy1 = p2[:, 1] - p1[:, 1]
            dx2 = p3[:, 0] - p1[:, 0]
            dy2 = p3[:, 1] - p1[:, 1]
            cross = np.abs(dx1 * dy2 - dy1 * dx2)
            areas = 0.5 * cross
            return -np.min(areas[combined_mask])
        
        try:
            res = minimize(_tri_obj, tri_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < -min_area + 1e-12:
                new_tri = res.x.reshape(3, 2)
                if _is_inside(new_tri, tol=1e-10):
                    best_points[i], best_points[j], best_points[k] = new_tri[0], new_tri[1], new_tri[2]
                    new_score = _min_area_fast(best_points)
                    if new_score > best_score:
                        best_score = new_score
        except Exception:
            pass
        
        # Also try optimizing random pairs with tight tolerance
        for _ in range(3):
            if time.time() > time_limit:
                break
            pair = np.random.choice(N, size=2, replace=False)
            a, b = int(pair[0]), int(pair[1])
            pair_points = best_points[[a, b]].copy()
            pair_mask = _point_masks[a] | _point_masks[b]
            
            def _pair_obj(pair_params):
                pp = pair_params.reshape(2, 2)
                x, y = pp[:, 0], pp[:, 1]
                if np.any(y < -1e-6) or np.any(y > np.sqrt(3) * x + 1e-6) or np.any(y > np.sqrt(3) * (1 - x) + 1e-6):
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
                return -np.min(areas[pair_mask])
            
            try:
                res = minimize(_pair_obj, pair_points.flatten(), method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
                if res.fun < -_min_area_fast(best_points) + 1e-12:
                    new_pair = res.x.reshape(2, 2)
                    if _is_inside(new_pair, tol=1e-10):
                        best_points[a], best_points[b] = new_pair[0], new_pair[1]
                        new_score = _min_area_fast(best_points)
                        if new_score > best_score:
                            best_score = new_score
            except Exception:
                pass
    
    return best_points, best_score


def _objective(params):
    """Objective: negative min area + boundary penalty."""
    points = params.reshape(N, 2)
    x = points[:, 0]
    y = points[:, 1]
    viol1 = np.maximum(0.0, -y + 0.005)
    viol2 = np.maximum(0.0, y - np.sqrt(3) * x + 0.005)
    viol3 = np.maximum(0.0, y - np.sqrt(3) * (1.0 - x) + 0.005)
    pen = np.sum(viol1**2) + np.sum(viol2**2) + np.sum(viol3**2)
    if pen > 1e-4:
        return -_min_area_fast(points) + 1000.0 * pen
    return -_min_area_fast(points)


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Try multiple initializations
    inits = [
        _init_symmetric(),
        _init_lattice(rng),
    ]
    
    for init_pts in inits:
        init_score = _min_area_fast(init_pts)
        if init_score > best_score:
            best_score = init_score
            best_points = init_pts.copy()
    
    # Phase 2: Coarse-to-fine refinement
    t_limit = time.time() + 30
    refined_pts, refined_score = _coarse_to_fine(best_points, rng, t_limit)
    if refined_score > best_score:
        best_points = refined_pts
        best_score = refined_score
    
    # Phase 3: Global optimization with differential evolution (if time permits)
    if time.time() < deadline - 40:
        bounds = [(0, 1), (0, H)] * N
        
        try:
            result = differential_evolution(
                _objective,
                bounds,
                maxiter=30,
                popsize=20,
                tol=1e-7,
                seed=42,
                polish=False,
                init='latinhypercube',
                mutation=(0.4, 0.9),
                recombination=0.8
            )
            de_points = result.x.reshape(N, 2)
            de_points = _clamp_to_triangle(de_points)
            de_score = _min_area_fast(de_points)
            
            if de_score > best_score:
                best_score = de_score
                best_points = de_points.copy()
        except Exception:
            pass
        
        # Quick local refine on the DE result
        if time.time() < deadline - 20:
            try:
                res = minimize(_objective, best_points.flatten(), method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
                nm_points = res.x.reshape(N, 2)
                nm_points = _clamp_to_triangle(nm_points)
                nm_score = _min_area_fast(nm_points)
                if nm_score > best_score:
                    best_score = nm_score
                    best_points = nm_points.copy()
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 2:
        final_pts, final_score = _final_polish(best_points, deadline - 1)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts
    
    if best_points is None:
        best_points = _init_symmetric()
    
    return _clamp_to_triangle(best_points)
