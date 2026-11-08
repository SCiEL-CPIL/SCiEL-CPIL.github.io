"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

def _all_triangle_areas(points):
    """Compute all triangle areas vectorized. Returns array of shape (C(n,3),)."""
    n = len(points)
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    areas = _all_triangle_areas(points)
    min_tri = np.min(areas)
    return min_tri / hull_area

def _negative_score(points):
    """Negative score for minimization."""
    return -_score(points)

def _two_ring_config(r_inner, angle_offset, n_outer=7, n_inner=7):
    """Generate two-ring configuration with given parameters."""
    outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + angle_offset
    
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_pts = r_inner * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    return np.vstack([outer_pts, inner_pts])

def _three_ring_config(r1, r2, a1, a2, n1=5, n2=5, n3=4):
    """Generate three-ring configuration."""
    angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + a1
    angles3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + a2
    
    pts1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    pts2 = r1 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    pts3 = r2 * np.column_stack([np.cos(angles3), np.sin(angles3)])
    
    return np.vstack([pts1, pts2, pts3])

def _regular_polygon(n, radius=1.0):
    """Generate points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _optimize_full_local(points, deadline):
    """Local optimization on all 28 coordinates using Nelder-Mead."""
    def objective(x):
        pts = x.reshape(-1, 2)
        return _negative_score(pts)
    
    x0 = points.flatten()
    
    if time.time() > deadline - 5:
        return points, _score(points)
    
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 3000,
                'xatol': 1e-9,
                'fatol': 1e-11,
                'adaptive': True
            }
        )
        best_points = result.x.reshape(-1, 2)
        best_score = _score(best_points)
        return best_points, best_score
    except Exception:
        return points, _score(points)

def _optimize_parameterized(params_bounds, param_to_points, deadline, maxiter=500):
    """Global optimization over parameterized family using differential evolution."""
    def objective(params):
        pts = param_to_points(*params)
        if pts is None or len(pts) != N:
            return 1e10
        return _negative_score(pts)
    
    if time.time() > deadline - 10:
        return None, -1
    
    try:
        result = differential_evolution(
            objective,
            bounds=params_bounds,
            maxiter=maxiter,
            tol=1e-10,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        best_params = result.x
        best_pts = param_to_points(*best_params)
        if best_pts is None:
            return None, -1
        best_score = _score(best_pts)
        return best_pts, best_score
    except Exception:
        return None, -1

def _fine_tune(points, deadline, max_iters=10000, step_size=0.002):
    """Fine-tuning with small random steps."""
    points = points.copy()
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    rng = np.random.RandomState(777)
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline - 2:
            break
        
        idx = rng.randint(len(points))
        delta = rng.normal(0, step_size, 2)
        new_points = points.copy()
        new_points[idx] += delta
        
        new_score = _score(new_points)
        if new_score > current_score:
            points = new_points
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            no_improve_count += 1
            if no_improve_count % 500 == 0:
                step_size *= 0.85
                no_improve_count = 0
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -1
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Phase 1: Try parameterized families with global optimization
    
    # Two-ring: 7 outer + 7 inner
    if time.time() < deadline - 40:
        bounds = [(0.1, 0.9), (0, 2*np.pi)]
        pts, sc = _optimize_parameterized(bounds, _two_ring_config, deadline, maxiter=200)
        if pts is not None and sc > best_score:
            update_best(pts, sc)
    
    # Two-ring with different splits: 6 outer + 8 inner
    if time.time() < deadline - 35:
        def _two_ring_68(r_inner, angle_offset):
            n_outer, n_inner = 6, 8
            outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
            inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + angle_offset
            outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
            inner_pts = r_inner * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
            return np.vstack([outer_pts, inner_pts])
        
        bounds = [(0.1, 0.9), (0, 2*np.pi)]
        pts, sc = _optimize_parameterized(bounds, _two_ring_68, deadline, maxiter=150)
        if pts is not None and sc > best_score:
            update_best(pts, sc)
    
    # Two-ring: 8 outer + 6 inner
    if time.time() < deadline - 30:
        def _two_ring_86(r_inner, angle_offset):
            n_outer, n_inner = 8, 6
            outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
            inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + angle_offset
            outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
            inner_pts = r_inner * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
            return np.vstack([outer_pts, inner_pts])
        
        bounds = [(0.1, 0.9), (0, 2*np.pi)]
        pts, sc = _optimize_parameterized(bounds, _two_ring_86, deadline, maxiter=150)
        if pts is not None and sc > best_score:
            update_best(pts, sc)
    
    # Three-ring: 5 + 5 + 4
    if time.time() < deadline - 25:
        bounds = [(0.2, 0.8), (0.3, 0.95), (0, 2*np.pi), (0, 2*np.pi)]
        pts, sc = _optimize_parameterized(bounds, _three_ring_config, deadline, maxiter=150)
        if pts is not None and sc > best_score:
            update_best(pts, sc)
    
    # Regular 14-gon as baseline
    if time.time() < deadline - 20:
        reg = _regular_polygon(N)
        sc = _score(reg)
        if sc > best_score:
            update_best(reg, sc)
    
    # Phase 2: Local optimization on best candidate
    if best_points is not None and time.time() < deadline - 15:
        pts, sc = _optimize_full_local(best_points, deadline)
        if sc > best_score:
            update_best(pts, sc)
    
    # Phase 3: Fine-tuning
    if best_points is not None and time.time() < deadline - 8:
        pts, sc = _fine_tune(best_points, deadline, max_iters=8000, step_size=0.003)
        if sc > best_score:
            update_best(pts, sc)
    
    # Phase 4: Another round of parameterized search with different initializations
    if time.time() < deadline - 10:
        # Try two-ring with 7+7 but different DE settings
        bounds = [(0.2, 0.8), (0, 2*np.pi)]
        
        def objective_78(params):
            pts = _two_ring_config(params[0], params[1])
            return _negative_score(pts)
        
        try:
            result = differential_evolution(
                objective_78,
                bounds=bounds,
                maxiter=300,
                tol=1e-12,
                seed=123,
                polish=True,
                init='latinhypercube'
            )
            best_params = result.x
            best_pts = _two_ring_config(*best_params)
            best_score_new = _score(best_pts)
            if best_score_new > best_score:
                update_best(best_pts, best_score_new)
        except Exception:
            pass
    
    # Phase 5: Final fine-tuning
    if best_points is not None and time.time() < deadline - 3:
        pts, sc = _fine_tune(best_points, deadline, max_iters=5000, step_size=0.001)
        if sc > best_score:
            update_best(pts, sc)
    
    # Phase 6: Final local polish
    if best_points is not None and time.time() < deadline - 1:
        def objective(x):
            pts = x.reshape(-1, 2)
            return _negative_score(pts)
        
        x0 = best_points.flatten()
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            polish_points = result.x.reshape(-1, 2)
            polish_score = _score(polish_points)
            if polish_score > best_score:
                update_best(polish_points, polish_score)
        except Exception:
            pass
    
    return best_points if best_points is not None else _regular_polygon(N)
# EVOLVE-BLOCK-END
