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

def _normalize(points):
    """Center and normalize points."""
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    hull_area = _hull_area(pts)
    if hull_area > 1e-12:
        pts = pts / np.sqrt(hull_area)
    return pts

def _objective_full(x_flat):
    """Full objective: minimize negative score."""
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

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

def _de_objective(x_flat):
    """Objective for differential evolution: minimize negative score."""
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

def _make_grid_config():
    """Create a 4x4 grid minus one corner (13 points), centered."""
    # 4x4 grid = 16 points, remove 3 corners to get 13
    # Actually let's use a different approach: 4x4 grid minus 3 points
    # Better: use a structured 13-point grid
    # 4x4 = 16, remove 3 to get 13
    grid = []
    for i in range(4):
        for j in range(4):
            grid.append([i - 1.5, j - 1.5])
    grid = np.array(grid)
    # Remove 3 points (corners) to get 13
    # Remove (0,0), (0,3), (3,0) -> indices 0, 3, 12
    remove_idx = [0, 3, 12]
    keep_idx = [i for i in range(16) if i not in remove_idx]
    grid = grid[keep_idx]
    # Center
    grid = grid - np.mean(grid, axis=0)
    return grid

def _make_grid_config2():
    """Alternative grid: 5x3 grid minus 2 points = 13 points."""
    grid = []
    for i in range(5):
        for j in range(3):
            grid.append([i - 2.0, j - 1.0])
    grid = np.array(grid)
    # Remove 2 points
    remove_idx = [0, 14]  # remove two corners
    keep_idx = [i for i in range(15) if i not in remove_idx]
    grid = grid[keep_idx]
    grid = grid - np.mean(grid, axis=0)
    return grid

def _optimize_from_start(x0, deadline, best_score_ref):
    """Run multi-phase optimization from a starting point."""
    best_x = x0.copy()
    best_s = _score(x0.reshape(N, 2))
    
    # Phase 1: Smooth optimization with Powell
    if time.time() < deadline - 8:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=30),
                x0,
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            s_smooth = _score(res.x.reshape(N, 2))
            if s_smooth > best_s:
                best_s = s_smooth
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 1b: Log-sum-exp smooth optimization
    if time.time() < deadline - 6:
        try:
            res_lse = minimize(
                lambda x: _logsumexp_smooth_obj(x, beta=50.0),
                best_x,
                method='Powell',
                options={'maxiter': 150, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            s_lse = _score(res_lse.x.reshape(N, 2))
            if s_lse > best_s:
                best_s = s_lse
                best_x = res_lse.x.copy()
        except Exception:
            pass
    
    # Phase 2: Hard objective polish with Nelder-Mead
    if time.time() < deadline - 3:
        try:
            res_nm = minimize(
                _objective_full,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            s_nm = _score(res_nm.x.reshape(N, 2))
            if s_nm > best_s:
                best_s = s_nm
                best_x = res_nm.copy()
        except Exception:
            pass
    
    return best_x, best_s

def _simulated_annealing(points, deadline, best_score_ref):
    """Run simulated annealing to escape local optima."""
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_s = current_score
    
    t_start = time.time()
    time_budget = deadline - 3.0
    
    total_time = max(1.0, time_budget - t_start)
    est_iters = int(total_time * 400)
    est_iters = max(200, min(est_iters, 30000))
    
    T0 = 0.08
    T_end = 0.0005
    alpha = (T_end / T0) ** (1.0 / max(1, est_iters - 1))
    
    rng = np.random.default_rng(42)
    
    for it in range(est_iters):
        if time.time() > time_budget:
            break
        
        T = T0 * (alpha ** it)
        
        # Adaptive perturbation: sometimes perturb only a subset of points
        if rng.random() < 0.3:
            # Perturb 3-5 random points more strongly
            n_perturb = rng.integers(3, 6)
            idx = rng.choice(N, size=n_perturb, replace=False)
            noise = np.zeros((N, 2))
            noise[idx] = rng.normal(0, T * 2.0, size=(n_perturb, 2))
        else:
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

def _run_differential_evolution(deadline):
    """Run DE for global search."""
    time_remaining = deadline - time.time()
    if time_remaining < 20:
        return None
    
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    de_params = {
        'maxiter': 40,
        'popsize': 15,
        'mutation': (0.4, 0.9),
        'recombination': 0.8,
        'tol': 1e-6,
        'seed': 42,
        'polish': False,
        'init': 'sobol',
    }
    
    try:
        result = differential_evolution(
            _de_objective,
            bounds,
            **de_params
        )
        if result.fun < 1e9:
            return result.x
    except Exception:
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (150s limit)
    
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
    
    # Phase 1: Differential Evolution for global search
    de_result = _run_differential_evolution(deadline)
    if de_result is not None:
        de_points = de_result.reshape(N, 2)
        _update_best(de_points)
        print(f"DE result: score={best_score:.6f}, time={time.time()-start_time:.1f}s", flush=True)
    
    # Generate diverse initial configurations
    inits = []
    
    # Add DE result if available
    if de_result is not None:
        inits.append(de_result.copy())
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]).flatten())
    
    # 2. Two-ring configurations
    for n_out in [5, 6, 7, 8]:
        n_in = N - n_out
        if n_in < 3:
            continue
        for r_in in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for offset in [0, np.pi / n_out, np.pi / (2 * n_out), 2 * np.pi / (3 * n_out)]:
                pts = _make_ring_config(n_out, n_in, r_in, offset)
                inits.append(pts.flatten())
    
    # 3. Three-ring configurations
    for n1, n2, n3 in [(4, 4, 5), (5, 4, 4), (6, 5, 2), (7, 3, 3), (5, 5, 3), (6, 4, 3)]:
        for r2 in [0.4, 0.5, 0.6, 0.7]:
            for r3 in [0.15, 0.25, 0.35, 0.45]:
                a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
                a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n1
                a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + np.pi / (2 * n2)
                pts = np.vstack([
                    np.column_stack([np.cos(a1), np.sin(a1)]),
                    np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
                    np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
                ])
                inits.append(pts.flatten())
    
    # 4. Grid-based configurations (affine transform strategy)
    grid1 = _make_grid_config()
    grid2 = _make_grid_config2()
    # Apply various affine transforms to grid
    for grid in [grid1, grid2]:
        for sx in [0.8, 1.0, 1.2, 1.5]:
            for sy in [0.8, 1.0, 1.2, 1.5]:
                for shear in [-0.3, -0.1, 0.0, 0.1, 0.3]:
                    for rot in [0, np.pi/8, np.pi/4, 3*np.pi/8]:
                        # Affine: scale, shear, rotate
                        c, s = np.cos(rot), np.sin(rot)
                        # Rotation matrix
                        R = np.array([[c, -s], [s, c]])
                        # Shear matrix
                        S = np.array([[1, shear], [0, 1]])
                        # Scale matrix
                        Sc = np.array([[sx, 0], [0, sy]])
                        # Combined: R @ S @ Sc
                        A = R @ S @ Sc
                        transformed = (grid @ A.T)
                        inits.append(transformed.flatten())
    
    # 5. Random starts
    rng = np.random.default_rng(12345)
    for _ in range(20):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts.flatten())
    
    # 6. Perturbed regular polygon
    for perturb_scale in [0.05, 0.1, 0.2]:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        base = np.column_stack([np.cos(angles), np.sin(angles)])
        for _ in range(3):
            perturbed = base + rng.normal(0, perturb_scale, size=(N, 2))
            inits.append(perturbed.flatten())
    
    # Shuffle inits
    rng2 = np.random.default_rng(999)
    indices = rng2.permutation(len(inits))
    inits = [inits[i] for i in indices]
    
    print(f"Total initial configs: {len(inits)}, time={time.time()-start_time:.1f}s", flush=True)
    
    # Optimize from each start
    n_optimized = 0
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        t_start = time.time()
        x_opt, s_opt = _optimize_from_start(x0, deadline, best_score)
        _update_best(x_opt.reshape(N, 2))
        n_optimized += 1
        
        if i % 10 == 0:
            elapsed = time.time() - t_start
            print(f"Init {i}/{len(inits)}: score={best_score:.6f}, elapsed={elapsed:.1f}s", flush=True)
    
    print(f"Optimized {n_optimized} configs, score={best_score:.6f}, time={time.time()-start_time:.1f}s", flush=True)
    
    # Final polish on best with local optimization
    if best_points is not None and time.time() < deadline - 15:
        try:
            res_final = minimize(
                _objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            _update_best(res_final.x.reshape(N, 2))
        except Exception:
            pass
    
    # Simulated Annealing to escape local optima
    if best_points is not None and time.time() < deadline - 8:
        try:
            sa_best, sa_score = _simulated_annealing(best_points, deadline, best_score)
            _update_best(sa_best)
        except Exception:
            pass
    
    # One more local polish after SA
    if best_points is not None and time.time() < deadline - 3:
        try:
            res_post = minimize(
                _objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            _update_best(res_post.x.reshape(N, 2))
        except Exception:
            pass
    
    # Final SA pass if time remains
    if best_points is not None and time.time() < deadline - 2:
        try:
            sa_best2, sa_score2 = _simulated_annealing(best_points, deadline, best_score)
            _update_best(sa_best2)
        except Exception:
            pass
    
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    print(f"Final score: {best_score:.6f}, time={time.time()-start_time:.1f}s", flush=True)
    
    return best_points
