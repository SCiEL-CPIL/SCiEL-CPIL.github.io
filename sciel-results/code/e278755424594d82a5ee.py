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

def _polar_to_cartesian(polar):
    """Convert (angles, radii) to Cartesian points."""
    angles = polar[:N]
    radii = polar[N:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _polar_objective(polar):
    """Objective for polar-parameterized DE: minimize negative score."""
    points = _polar_to_cartesian(polar)
    points = points - np.mean(points, axis=0)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s

def _smooth_objective(x_flat, p=80):
    """Smooth surrogate using log-sum-exp based soft-min."""
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
    
    # Log-sum-exp based soft-min (numerically stable)
    neg_p_areas = -p * areas
    max_neg = np.max(neg_p_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_p_areas - max_neg)))
    soft_min = -log_sum / p
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _soft_score(points, p=50):
    """Compute soft score directly from points."""
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    if np.allclose(pts, 0):
        return 0.0
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(pts)
    areas = np.maximum(areas, 1e-15)
    neg_p_areas = -p * areas
    max_neg = np.max(neg_p_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_p_areas - max_neg)))
    soft_min = -log_sum / p
    return soft_min / hull_area

def _objective_full(x_flat):
    """Full objective: minimize negative score."""
    points = x_flat.reshape(N, 2)
    points = points - np.mean(points, axis=0)
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

def _coordinate_descent_polish(points, deadline, max_iter=5000):
    """Coordinate descent on the hard objective with adaptive step sizes."""
    best_pts = points.copy()
    best_s = _score(best_pts)
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            for coord in range(2):
                for delta in step_sizes:
                    for sign in [1, -1]:
                        trial = best_pts.copy()
                        trial[i, coord] += sign * delta
                        trial_s = _score(trial)
                        if trial_s > best_s + 1e-14:
                            best_pts = trial
                            best_s = trial_s
                            improved = True
                            break
                    if improved:
                        break
        
        if not improved:
            break
    
    return best_pts, best_s

def _local_polish(points, deadline):
    """Multi-phase local optimization from a starting point."""
    best_x = points.flatten().copy()
    best_s = _score(points)
    
    # Phase 1: Powell on smooth objective (coarse)
    if time.time() < deadline - 15:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=50),
                best_x,
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Powell on smoother objective (fine)
    if time.time() < deadline - 10:
        try:
            res2 = minimize(
                lambda x: _smooth_objective(x, p=100),
                best_x,
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            s2 = _score(res2.x.reshape(N, 2))
            if s2 > best_s:
                best_s = s2
                best_x = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead on full objective
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(
                _objective_full,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 600, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            s_nm = _score(res_nm.x.reshape(N, 2))
            if s_nm > best_s:
                best_s = s_nm
                best_x = res_nm.x.copy()
        except Exception:
            pass
    
    # Phase 4: Coordinate descent polish
    if time.time() < deadline - 2:
        try:
            cd_pts, cd_s = _coordinate_descent_polish(best_x.reshape(N, 2), deadline)
            if cd_s > best_s:
                best_s = cd_s
                best_x = cd_pts.flatten().copy()
        except Exception:
            pass
    
    return best_x.reshape(N, 2), best_s

def _run_differential_evolution(deadline, best_score_ref):
    """Run DE in polar space to find diverse good starting points."""
    # Bounds: angles in [0, 2*pi), radii in [0.1, 2.0]
    bounds = []
    for _ in range(N):
        bounds.append((0, 2 * np.pi))
    for _ in range(N):
        bounds.append((0.1, 2.0))
    
    # Time budget for DE
    de_time_budget = min(40.0, max(10.0, deadline - time.time() - 60.0))
    
    try:
        de_result = differential_evolution(
            _polar_objective,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            init='latinhypercube',
            polish=False,
            callback=lambda xk, convergence: (time.time() > deadline - 60.0)
        )
        
        if de_result.success or de_result.fun < 1e9:
            points = _polar_to_cartesian(de_result.x)
            points = points - np.mean(points, axis=0)
            s = _score(points)
            if s > best_score_ref:
                return points, s
    except Exception:
        pass
    
    return None, best_score_ref

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
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
    
    # Phase 1: Differential Evolution in polar space for global exploration
    de_points, de_score = _run_differential_evolution(deadline, best_score)
    if de_points is not None:
        _update_best(de_points)
    
    # Generate diverse initial configurations
    inits = []
    rng = np.random.default_rng(12345)
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]).flatten())
    
    # 2. Two-ring configurations (more diverse)
    for n_out in [5, 6, 7, 8]:
        n_in = N - n_out
        if n_in < 3:
            continue
        for r_in in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
            for offset in [0, np.pi / n_out, np.pi / (2 * n_out), np.pi / (3 * n_out)]:
                pts = _make_ring_config(n_out, n_in, r_in, offset)
                inits.append(pts.flatten())
    
    # 3. Three-ring configurations (more diverse)
    for n1, n2, n3 in [(4, 4, 5), (5, 4, 4), (6, 5, 2), (7, 3, 3), (5, 5, 3)]:
        for r2 in [0.4, 0.5, 0.6, 0.7]:
            for r3 in [0.15, 0.25, 0.35, 0.45]:
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
    for _ in range(10):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts.flatten())
    
    # 5. Perturbed regular polygon (more scales)
    for noise_scale in [0.01, 0.03, 0.05, 0.08, 0.12]:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0 + noise_scale * (rng.random(N) - 0.5)
        angles_perturbed = angles + noise_scale * (rng.random(N) - 0.5)
        pts = np.column_stack([radii * np.cos(angles_perturbed), radii * np.sin(angles_perturbed)])
        inits.append(pts.flatten())
    
    # 6. Quasi-regular with alternating radii
    for base_r in [0.8, 0.9, 1.0]:
        for alt_r in [0.5, 0.6, 0.7]:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            radii = np.where(np.arange(N) % 2 == 0, base_r, alt_r)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            inits.append(pts.flatten())
    
    # Add DE result as an init if available
    if de_points is not None:
        inits.insert(0, de_points.flatten())
    
    # Shuffle inits
    rng2 = np.random.default_rng(999)
    indices = rng2.permutation(len(inits))
    inits = [inits[i] for i in indices]
    
    # Optimize from each start
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        t_start = time.time()
        pts_opt, s_opt = _local_polish(x0.reshape(N, 2), deadline - 8)
        _update_best(pts_opt)
        
        if i % 3 == 0:
            print(f"Init {i}: score={best_score:.6f}, elapsed={time.time()-start_time:.1f}s", flush=True)
    
    # Final high-precision polish on best
    if best_points is not None and time.time() < deadline - 5:
        try:
            # High-precision Nelder-Mead
            res_final = minimize(
                _objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            _update_best(res_final.x.reshape(N, 2))
        except Exception:
            pass
        
        # Final coordinate descent with very fine steps
        if best_points is not None and time.time() < deadline - 1:
            try:
                # Use finer step sizes for final polish
                fine_steps = [0.001, 0.0005, 0.0002, 0.0001, 0.00005]
                cd_pts = best_points.copy()
                cd_s = _score(cd_pts)
                for _ in range(2000):
                    if time.time() > deadline - 0.5:
                        break
                    improved = False
                    for i in range(N):
                        for coord in range(2):
                            for delta in fine_steps:
                                for sign in [1, -1]:
                                    trial = cd_pts.copy()
                                    trial[i, coord] += sign * delta
                                    trial_s = _score(trial)
                                    if trial_s > cd_s + 1e-14:
                                        cd_pts = trial
                                        cd_s = trial_s
                                        improved = True
                                        break
                                if improved:
                                    break
                        if improved:
                            break
                    if not improved:
                        break
                _update_best(cd_pts)
            except Exception:
                pass
    
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
