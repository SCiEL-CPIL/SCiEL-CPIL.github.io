"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize, dual_annealing
from scipy.spatial import Delaunay

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

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


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * np.min(cross)


def _min_area_fast(points):
    """Faster min area using vectorized computation with early exit potential."""
    # Compute all cross products
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
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    areas = 0.5 * cross
    idx = int(np.argmin(areas))
    return areas[idx], idx


def _inside_penalty(points, margin=0.005):
    """Smooth penalty for points outside the triangle. Returns 0 if all inside (with margin)."""
    x = points[:, 0]
    y = points[:, 1]
    # Triangle constraints: y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
    # Use smooth max(0, violation) penalty
    viol1 = np.maximum(0.0, -y + margin)
    viol2 = np.maximum(0.0, y - np.sqrt(3) * x + margin)
    viol3 = np.maximum(0.0, y - np.sqrt(3) * (1.0 - x) + margin)
    return np.sum(viol1**2) + np.sum(viol2**2) + np.sum(viol3**2)


def _objective(params):
    """Objective: negative min area + boundary penalty. params: 22 coords."""
    points = params.reshape(N, 2)
    pen = _inside_penalty(points)
    if pen > 1e-4:
        return -_min_area_fast(points) + 1000.0 * pen
    return -_min_area_fast(points)


def _objective_with_constraint(params, margin=0.001):
    """Objective with hard constraint checking for dual_annealing."""
    points = params.reshape(N, 2)
    x, y = points[:, 0], points[:, 1]
    # Check triangle boundaries
    if np.any(y < -margin) or np.any(y > np.sqrt(3) * x + margin) or np.any(y > np.sqrt(3) * (1 - x) + margin):
        return 1000.0
    return -_min_area_fast(points)


def _init_symmetric():
    """Generate a good symmetric starting configuration."""
    points = np.zeros((N, 2))
    # Place points in a roughly hexagonal/symmetric pattern
    # 3 rows: top has fewer points
    # Row 0 (top): 1 point near center top
    # Row 1: 3 points
    # Row 2: 3 points
    # Row 3: 4 points (bottom)
    
    # Use known good pattern: place points to maximize spacing
    # Approximate optimal for 11 in equilateral triangle
    # Place in a grid clipped to triangle
    
    # Center of triangle
    cx, cy = 1.0/3.0, H/3.0
    
    # Try a structured placement
    # 4-3-3-1 pattern from bottom to top
    # Bottom row (y small): 4 points
    # Second row: 3 points
    # Third row: 3 points
    # Top: 1 point
    
    # Let's use a more systematic approach: golden ratio spacing
    # Actually, let's use a well-known heuristic: place points on concentric "levels"
    
    # Level structure for 11 points in triangle:
    # 4 on bottom edge area, 3 on next level, 3 on next, 1 on top
    levels = [4, 3, 3, 1]
    total = sum(levels)
    
    # y-coordinates for each level (from bottom)
    # Triangle height is H, width at y is (1 - y/H)
    # Place levels at y = 0, H/4, H/2, 3H/4 roughly, but adjust
    
    # Better: place at y fractions that give equal "space"
    # Use y = H * (1 - (k+0.5)/n_levels) for level k, but this is rough
    
    # Let me use a cleaner approach:
    # 1 point at center-top
    # 3 points in a triangle below
    # 3 points below that
    # 4 points on the bottom
    
    # Actually, for 11 points, a known good arrangement is based on
    # a slightly perturbed hexagonal lattice clipped to the triangle.
    
    # Let me just use a parameterized symmetric pattern:
    # 3x3 grid + 2 extra points
    
    # Simple structured init:
    # Place points in 4 rows
    row_ys = [0.08 * H, 0.30 * H, 0.55 * H, 0.78 * H]
    row_ns = [4, 3, 3, 1]
    
    pi = 0
    for y_frac, n_pts in zip(row_ys, row_ns):
        y = y_frac
        # Width of triangle at height y: from x_left to x_right
        # x_left = y / sqrt(3) = y * H / 0.866... 
        # Actually: left edge goes from (0,0) to (0.5, H), so x = 0.5 * y / H
        # Right edge goes from (1,0) to (0.5, H), so x = 1 - 0.5 * y / H
        x_left = 0.5 * y / H
        x_right = 1.0 - 0.5 * y / H
        width = x_right - x_left
        
        if n_pts == 1:
            points[pi, 0] = 0.5
            points[pi, 1] = y
            pi += 1
        else:
            # Place n_pts equally spaced with margin
            margin_frac = 0.15
            usable = width * (1 - 2 * margin_frac)
            start = x_left + width * margin_frac
            if n_pts > 1:
                xs = start + np.linspace(0, usable, n_pts)
            else:
                xs = np.array([0.5])
            for x in xs:
                points[pi, 0] = x
                points[pi, 1] = y
                pi += 1
    
    return points


def _init_random(rng):
    """Generate random points inside the triangle."""
    points = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1:
                points[i, 0] = v + 0.5 * (1 - u - v)
                points[i, 1] = H * (1 - u - v)
                break
    return points


def _coarse_to_fine(points, rng, time_limit):
    """Iteratively refine: find worst triangle, do local search, repeat."""
    start = time.time()
    best_points = points.copy()
    best_score = _min_area_fast(points)
    
    # Precompute mask for each point: which triangles involve it
    point_masks = []
    for p in range(N):
        mask = ((_idx_i == p) | (_idx_j == p) | (_idx_k == p))
        point_masks.append(mask)
    
    for iteration in range(300):
        if time.time() > time_limit:
            break
        
        # Find the worst (smallest area) triangle
        min_area, worst_idx = _min_area_and_idx(best_points)
        
        if min_area > best_score:
            best_score = min_area
        
        # Get the three points forming the worst triangle
        i, j, k = _idx_i[worst_idx], _idx_j[worst_idx], _idx_k[worst_idx]
        tri_points = best_points[[i, j, k]].copy()
        
        # Combine masks for these 3 points
        combined_mask = point_masks[i] | point_masks[j] | point_masks[k]
        
        # Try to improve by moving these points
        def _tri_obj(tri_params):
            tp = tri_params.reshape(3, 2)
            # Check they're inside
            x, y = tp[:, 0], tp[:, 1]
            if np.any(y < -1e-6) or np.any(y > np.sqrt(3)*x + 1e-6) or np.any(y > np.sqrt(3)*(1-x) + 1e-6):
                return 1000.0
            # Reconstruct full points
            full = best_points.copy()
            full[i], full[j], full[k] = tp[0], tp[1], tp[2]
            # Compute min area of triangles that include at least one of i,j,k
            p1 = full[_idx_i]
            p2 = full[_idx_j]
            p3 = full[_idx_k]
            dx1 = p2[:, 0] - p1[:, 0]
            dy1 = p2[:, 1] - p1[:, 1]
            dx2 = p3[:, 0] - p1[:, 0]
            dy2 = p3[:, 1] - p1[:, 1]
            cross = np.abs(dx1 * dy2 - dy1 * dx2)
            areas = 0.5 * cross
            if np.any(combined_mask):
                return -np.min(areas[combined_mask])
            return -np.min(areas)
        
        # Local search on the 3 points
        try:
            res = minimize(_tri_obj, tri_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11})
            if res.fun < -min_area + 1e-10:
                # Check if new points are inside
                new_tri = res.x.reshape(3, 2)
                x, y = new_tri[:, 0], new_tri[:, 1]
                if np.all(y >= -1e-8) and np.all(y <= np.sqrt(3)*x + 1e-8) and np.all(y <= np.sqrt(3)*(1-x) + 1e-8):
                    best_points[i], best_points[j], best_points[k] = new_tri[0], new_tri[1], new_tri[2]
                    new_score = _min_area_fast(best_points)
                    if new_score > best_score:
                        best_score = new_score
        except Exception:
            pass
        
        # Occasionally try moving a random pair of points
        if iteration % 5 == 0:
            pair = rng.choice(N, size=2, replace=False)
            a, b = int(pair[0]), int(pair[1])
            pair_points = best_points[[a, b]].copy()
            pair_mask = point_masks[a] | point_masks[b]
            
            def _pair_obj(pair_params):
                pp = pair_params.reshape(2, 2)
                x, y = pp[:, 0], pp[:, 1]
                if np.any(y < -1e-6) or np.any(y > np.sqrt(3)*x + 1e-6) or np.any(y > np.sqrt(3)*(1-x) + 1e-6):
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
                if np.any(pair_mask):
                    return -np.min(areas[pair_mask])
                return -np.min(areas)
            
            try:
                res = minimize(_pair_obj, pair_points.flatten(), method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11})
                if res.fun < -_min_area_fast(best_points) + 1e-10:
                    new_pair = res.x.reshape(2, 2)
                    x, y = new_pair[:, 0], new_pair[:, 1]
                    if np.all(y >= -1e-8) and np.all(y <= np.sqrt(3)*x + 1e-8) and np.all(y <= np.sqrt(3)*(1-x) + 1e-8):
                        best_points[a], best_points[b] = new_pair[0], new_pair[1]
                        new_score = _min_area_fast(best_points)
                        if new_score > best_score:
                            best_score = new_score
            except Exception:
                pass
    
    return best_points, best_score


def _refine_all_points(points, time_limit):
    """Do a global refinement using Nelder-Mead on all 22 coordinates."""
    start = time.time()
    
    # Run Nelder-Mead from the current point
    try:
        res = minimize(_objective, points.flatten(), method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11,
                               'maxfev': 10000})
        if res.fun < -_min_area(points):
            new_points = res.x.reshape(N, 2)
            # Verify inside
            x, y = new_points[:, 0], new_points[:, 1]
            if np.all(y >= -1e-8) and np.all(y <= np.sqrt(3)*x + 1e-8) and np.all(y <= np.sqrt(3)*(1-x) + 1e-8):
                return new_points
    except Exception:
        pass
    
    # Try COBYLA which handles constraints differently
    try:
        def _cobyla_obj(params):
            pts = params.reshape(N, 2)
            return -_min_area(pts)
        
        def _con1(params):
            pts = params.reshape(N, 2)
            return pts[:, 1] + 1e-6
        
        def _con2(params):
            pts = params.reshape(N, 2)
            return np.sqrt(3) * pts[:, 0] - pts[:, 1] + 1e-6
        
        def _con3(params):
            pts = params.reshape(N, 2)
            return np.sqrt(3) * (1 - pts[:, 0]) - pts[:, 1] + 1e-6
        
        cons = []
        for i in range(N):
            def _c1(i=i):
                return _con1(np.r_[points[i,0], points[i,1]])  # placeholder
            # COBYLA constraints are tricky with vector; skip for now
        pass
    except Exception:
        pass
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Get a good initial configuration
    init_pts = _init_symmetric()
    best_points = init_pts.copy()
    best_score = _min_area_fast(init_pts)
    
    # Phase 2: Refine with coarse-to-fine
    t_limit = time.time() + 25
    refined_pts, refined_score = _coarse_to_fine(init_pts, rng, t_limit)
    if refined_score > best_score:
        best_points = refined_pts
        best_score = refined_score
    
    if time.time() > deadline - 60:
        return _clamp_to_triangle(best_points)
    
    # Phase 3: Global optimization with differential evolution
    bounds = [(0, 1), (0, H)] * N
    
    # Run DE with moderate settings
    n_de = 0
    while time.time() < deadline - 35:
        n_de += 1
        pop_size = 25 if n_de == 1 else 20
        max_iter = 40 if n_de == 1 else 25
        
        try:
            result = differential_evolution(
                _objective,
                bounds,
                maxiter=max_iter,
                popsize=pop_size,
                tol=1e-7,
                seed=n_de * 137,
                polish=False,
                init='latinhypercube',
                mutation=(0.4, 0.9),
                recombination=0.8
            )
        except Exception:
            continue
        
        de_points = result.x.reshape(N, 2)
        de_points = _clamp_to_triangle(de_points)
        de_score = _min_area_fast(de_points)
        
        if de_score > best_score:
            best_score = de_score
            best_points = de_points.copy()
        
        # Quick local refine on the DE result
        if time.time() < deadline - 15:
            try:
                res = minimize(_objective, de_points.flatten(), method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11})
                nm_points = res.x.reshape(N, 2)
                nm_points = _clamp_to_triangle(nm_points)
                nm_score = _min_area_fast(nm_points)
                if nm_score > best_score:
                    best_score = nm_score
                    best_points = nm_points.copy()
            except Exception:
                pass
    
    # Phase 4: Try dual_annealing if time permits
    if time.time() < deadline - 25:
        try:
            da_result = dual_annealing(
                _objective_with_constraint,
                bounds=bounds,
                seed=42,
                maxiter=100,
                restarts=5,
                initial_temp=5230.0,
                visit=[1.1, 1.3],
                clip=[-1, 1]
            )
            da_points = da_result.x.reshape(N, 2)
            da_points = _clamp_to_triangle(da_points)
            da_score = _min_area_fast(da_points)
            if da_score > best_score:
                best_score = da_score
                best_points = da_points.copy()
        except Exception:
            pass
    
    # Phase 5: Final refinement from the best point
    if time.time() < deadline - 3:
        final_pts, final_score = _coarse_to_fine(best_points, rng, deadline - 1)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts
    
    if best_points is None:
        best_points = _init_symmetric()
    
    return _clamp_to_triangle(best_points)


def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle."""
    points = points.copy()
    for i in range(N):
        x, y = points[i]
        # Clamp y to [0, H]
        y = max(0.0, min(H, y))
        # Clamp based on triangle boundaries
        # Left boundary: y <= sqrt(3)*x  =>  x >= y/sqrt(3)
        # Right boundary: y <= sqrt(3)*(1-x)  =>  x <= 1 - y/sqrt(3)
        x_min = y / np.sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        if x_min > x_max:
            # Degenerate, put at center
            x, y = 0.5, 0.0
        else:
            x = max(x_min, min(x_max, x))
        points[i] = [x, y]
    return points
