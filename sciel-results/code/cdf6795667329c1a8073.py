import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

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
    
    # Soft min
    inv_areas = 1.0 / areas
    soft_min = (np.mean(inv_areas ** p)) ** (-1.0 / p)
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _logsumexp_smooth_obj(x_flat, beta=50.0):
    """Log-sum-exp smooth surrogate."""
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
    
    neg_beta_over_areas = -beta / areas
    m = np.max(neg_beta_over_areas)
    lse = m + np.log(np.mean(np.exp(neg_beta_over_areas - m)))
    return lse + np.log(hull_area)

def _objective_full(x_flat):
    """Full objective: minimize negative score."""
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _de_objective(x_flat):
    """Objective for differential evolution."""
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _make_ring_config(n_outer, n_inner, r_inner, angle_offset):
    """Create a two-ring configuration."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_inner, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([outer_pts, inner_pts])

def _coordinate_descent_polish(points, deadline, best_score):
    """Coordinate descent polishing with adaptive step sizes."""
    best_pts = points.copy()
    best_s = _score(points)
    step_size = 0.01
    
    while time.time() < deadline - 2:
        improved = False
        for i in range(N):
            for d in range(2):
                for sign in [1, -1]:
                    new_pts = best_pts.copy()
                    new_pts[i, d] += sign * step_size
                    s = _score(new_pts)
                    if s > best_s:
                        best_s = s
                        best_pts = new_pts.copy()
                        improved = True
                    elif s > best_s * 0.99:
                        # Try smaller step
                        new_pts2 = best_pts.copy()
                        new_pts2[i, d] += sign * step_size * 0.5
                        s2 = _score(new_pts2)
                        if s2 > best_s:
                            best_s = s2
                            best_pts = new_pts2.copy()
                            improved = True
        if not improved:
            step_size *= 0.7
            if step_size < 1e-8:
                break
    
    return best_pts, best_s

def _local_polish(points, deadline):
    """Multi-method local polish."""
    best_pts = points.copy()
    best_s = _score(points)
    
    # Nelder-Mead with adaptive options
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _objective_full,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_pts = res.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    # Coordinate descent
    if time.time() < deadline - 2:
        cd_pts, cd_s = _coordinate_descent_polish(best_pts, deadline, best_s)
        if cd_s > best_s:
            best_s = cd_s
            best_pts = cd_pts.copy()
    
    return best_pts, best_s

def _optimize_from_start(x0, deadline):
    """Run multi-phase optimization from a starting point."""
    best_x = x0.copy()
    best_s = _score(x0.reshape(N, 2))
    
    # Phase 1: Smooth optimization with log-sum-exp
    if time.time() < deadline - 15:
        try:
            res = minimize(
                lambda x: _logsumexp_smooth_obj(x, beta=50.0),
                x0,
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Smooth optimization with power mean
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=30),
                best_x,
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Local polish
    if time.time() < deadline - 3:
        polished_pts, polished_s = _local_polish(best_x.reshape(N, 2), deadline)
        if polished_s > best_s:
            best_s = polished_s
            best_x = polished_pts.flatten().copy()
    
    return best_x, best_s

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
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
    
    # Phase 1: Run Differential Evolution for global search
    de_result = None
    if time.time() < deadline - 30:
        try:
            bounds = [(-2.0, 2.0)] * (2 * N)
            result = differential_evolution(
                _de_objective,
                bounds,
                maxiter=40,
                popsize=15,
                mutation=(0.3, 0.9),
                recombination=0.8,
                tol=1e-8,
                seed=42,
                polish=False,
                init='sobol',
            )
            if result.fun < 1e9:
                de_result = result.x
                _update_best(de_result.reshape(N, 2))
                print(f"DE result: score={best_score:.6f}", flush=True)
        except Exception:
            pass
    
    # Generate diverse initial configurations
    inits = []
    
    # Add DE result as first init if available
    if de_result is not None:
        inits.append(de_result.copy())
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]).flatten())
    
    # 2. Two-ring configurations
    for n_out in [6, 7, 8]:
        n_in = N - n_out
        for r_in in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for offset in [0, np.pi / n_out, np.pi / (2 * n_out)]:
                pts = _make_ring_config(n_out, n_in, r_in, offset)
                inits.append(pts.flatten())
    
    # 3. Three-ring configurations
    for n1, n2, n3 in [(5, 4, 4), (6, 5, 2), (7, 3, 3), (4, 4, 5)]:
        for r2 in [0.5, 0.6, 0.7]:
            for r3 in [0.2, 0.3, 0.4]:
                a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
                a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n1
                a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + np.pi / n2
                pts = np.vstack([
                    np.column_stack([np.cos(a1), np.sin(a1)]),
                    np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
                    np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
                ])
                inits.append(pts.flatten())
    
    # 4. Random starts
    rng = np.random.default_rng(12345)
    for _ in range(12):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts.flatten())
    
    # 5. Perturbed regular polygon
    for noise_scale in [0.02, 0.05, 0.1]:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0 + noise_scale * (rng.random(N) - 0.5)
        angles_perturbed = angles + noise_scale * (rng.random(N) - 0.5)
        pts = np.column_stack([radii * np.cos(angles_perturbed), radii * np.sin(angles_perturbed)])
        inits.append(pts.flatten())
    
    # Shuffle inits
    rng2 = np.random.default_rng(999)
    indices = rng2.permutation(len(inits))
    inits = [inits[i] for i in indices]
    
    # Optimize from each start
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 12:
            break
        
        x_opt, s_opt = _optimize_from_start(x0, deadline)
        _update_best(x_opt.reshape(N, 2))
        
        if i % 5 == 0:
            print(f"Init {i}: score={best_score:.6f}", flush=True)
    
    # Final high-precision polish on best
    if best_points is not None and time.time() < deadline - 3:
        try:
            res_final = minimize(
                _objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            _update_best(res_final.x.reshape(N, 2))
        except Exception:
            pass
        
        # Additional coordinate descent polish
        if time.time() < deadline - 1:
            cd_pts, cd_s = _coordinate_descent_polish(best_points, deadline, best_score)
            if cd_s > best_score:
                best_score = cd_s
                best_points = cd_pts.copy()
    
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
