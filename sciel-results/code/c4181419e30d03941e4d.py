import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _polar_to_cartesian(angles, radii):
    """Convert polar parameters to Cartesian points."""
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _polar_objective(params):
    """
    Objective for polar parameterization.
    params: 26-dim array [angles_13, radii_13]
    Angles are sorted, radii are positive.
    """
    angles = params[:N]
    radii = params[N:]
    
    # Ensure angles are sorted (they should be by construction in DE bounds)
    # Ensure radii are positive
    if np.any(radii <= 0):
        return 1e10
    
    points = _polar_to_cartesian(angles, radii)
    
    # Check if points form a convex polygon (all cross products same sign)
    # For a convex polygon with sorted angles, this should hold if radii are reasonable
    # But let's verify by checking the hull has all N points
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < N:
            # Not all points are on the hull - penalty
            penalty = (N - len(hull.vertices)) * 0.1
            score = _score(points)
            return -(score - penalty)
    except Exception:
        return 1e10
    
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _run_polar_de(deadline):
    """Run Differential Evolution on polar parameterization."""
    # Bounds: angles in [0, 2*pi] sorted, radii in [0.1, 2.0]
    # For sorted angles, we can use cumulative approach or just bound each angle
    # Simpler: each angle in [0, 2*pi], then sort after decoding
    # But DE works better with proper bounds. Let's use angle differences.
    
    # Alternative: parameterize as angle gaps that sum to 2*pi
    # gaps_i > 0, sum(gaps) = 2*pi
    # Use softmax-like or just normalize
    
    # Actually, let's keep it simple: angles in [0, 2*pi], sort them
    # DE will handle the search space
    
    angle_bounds = [(0, 2 * np.pi)] * N
    radius_bounds = [(0.05, 2.5)] * N
    bounds = angle_bounds + radius_bounds
    
    # Initial population
    rng = np.random.default_rng(42)
    
    def _de_callback(xk, convergence):
        if time.time() > deadline - 10:
            raise Exception("Time budget exceeded")
    
    try:
        from scipy.optimize import differential_evolution
        
        def _de_obj(params):
            angles = np.sort(params[:N])
            radii = np.maximum(params[N:], 0.01)
            points = _polar_to_cartesian(angles, radii)
            s = _score(points)
            if s <= 0:
                return 1e10
            return -s
        
        res = differential_evolution(
            _de_obj,
            bounds,
            maxiter=100,
            popsize=20,
            tol=0.01,
            seed=42,
            polish=True,
            callback=_de_callback,
            updating='deferred'
        )
        
        angles = np.sort(res.x[:N])
        radii = np.maximum(res.x[N:], 0.01)
        points = _polar_to_cartesian(angles, radii)
        
        return points, _score(points)
    except Exception as e:
        warnings.warn(f"Polar DE failed: {e}")
        return None, 0.0

def _optimize_from_polar_start(angles, radii, deadline):
    """Local optimization from a polar parameterization."""
    points = _polar_to_cartesian(angles, radii)
    best_x = points.flatten()
    best_s = _score(points)
    
    # Nelder-Mead on Cartesian coordinates
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _score(x.reshape(N, 2)) * -1,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x.reshape(N, 2), best_s

def _smooth_objective(x_flat, p=30):
    """Smooth surrogate using power mean."""
    points = x_flat.reshape(N, 2)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    if np.allclose(points, 0):
        return 1e10
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    inv_areas = 1.0 / areas
    soft_min = (np.mean(inv_areas ** p)) ** (-1.0 / p)
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _objective_full(x_flat):
    """Full objective: minimize negative score."""
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _simulated_annealing(points, deadline, best_score_ref):
    """Run simulated annealing to escape local optima."""
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_s = current_score
    
    t_start = time.time()
    time_budget = deadline - 5.0
    
    total_time = max(1.0, time_budget - t_start)
    est_iters = int(total_time * 300)
    est_iters = max(100, min(est_iters, 20000))
    
    T0 = 0.05
    T_end = 0.001
    alpha = (T_end / T0) ** (1.0 / max(1, est_iters - 1))
    
    rng = np.random.default_rng(42)
    
    for it in range(est_iters):
        if time.time() > time_budget:
            break
        
        T = T0 * (alpha ** it)
        noise = rng.normal(0, T, size=(N, 2))
        candidate = current + noise
        candidate = np.clip(candidate, -5, 5)
        
        cand_score = _score(candidate)
        
        if cand_score > current_score:
            current = candidate
            current_score = cand_score
            if current_score > best_s:
                best_s = current_score
                best = current.copy()
        else:
            delta = current_score - cand_score
            if delta < 1e-12:
                current = candidate
                current_score = cand_score
            else:
                prob = np.exp(-delta / max(T, 1e-10))
                if rng.random() < prob:
                    current = candidate
                    current_score = cand_score
    
    return best, best_s

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # Phase 1: Differential Evolution on polar parameterization
    if time.time() < deadline - 30:
        polar_pts, polar_score = _run_polar_de(deadline - 30)
        if polar_pts is not None:
            _update_best(polar_pts)
            print(f"Polar DE: score={polar_score:.6f}", flush=True)
    
    # Phase 2: Local optimization from polar result
    if best_points is not None and time.time() < deadline - 15:
        # Convert best points back to polar for local opt
        centroid = np.mean(best_points, axis=0)
        centered = best_points - centroid
        angles = np.arctan2(centered[:, 1], centered[:, 0])
        radii = np.sqrt(centered[:, 0]**2 + centered[:, 1]**2)
        angles = np.sort(angles)
        
        polished, pol_score = _optimize_from_polar_start(angles, radii, deadline - 10)
        _update_best(polished)
        print(f"After polar local opt: score={best_score:.6f}", flush=True)
    
    # Phase 3: Additional diverse starts + local optimization
    inits = []
    
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]).flatten())
    
    # Two-ring configs
    for n_out in [6, 7, 8]:
        n_in = N - n_out
        for r_in in [0.3, 0.5, 0.7]:
            angles_outer = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
            angles_inner = np.linspace(np.pi / n_out, 2 * np.pi + np.pi / n_out, n_in, endpoint=False)
            outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
            inner_pts = np.column_stack([r_in * np.cos(angles_inner), r_in * np.sin(angles_inner)])
            pts = np.vstack([outer_pts, inner_pts])
            inits.append(pts.flatten())
    
    # Random starts
    rng = np.random.default_rng(12345)
    for _ in range(10):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts.flatten())
    
    # Optimize from each start
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        try:
            res = minimize(
                _objective_full,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            _update_best(res.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 4: Simulated Annealing
    if best_points is not None and time.time() < deadline - 5:
        try:
            sa_best, sa_score = _simulated_annealing(best_points, deadline, best_score)
            _update_best(sa_best)
        except Exception:
            pass
    
    # Phase 5: Final polish
    if best_points is not None and time.time() < deadline - 1:
        try:
            res_final = minimize(
                _objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            _update_best(res_final.x.reshape(N, 2))
        except Exception:
            pass
    
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
