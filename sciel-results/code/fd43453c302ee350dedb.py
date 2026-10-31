import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

# Precompute triple indices for vectorized triangle area computation
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(13)
    for j in range(i + 1, 13)
    for k in range(j + 1, 13)
], dtype=np.int32)

N = 13

def _affine_transform(points, a, b, c, d):
    """Apply affine transform: [x'] = [a b; c d] * [x; y]"""
    xs = points[:, 0]
    ys = points[:, 1]
    new_x = a * xs + b * ys
    new_y = c * xs + d * ys
    return np.column_stack([new_x, new_y])

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

def _smooth_objective_adaptive(x_flat, p_start=20, p_end=80, beta=30.0):
    """Adaptive smooth surrogate that increases sharpness over iterations."""
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
    
    # Adaptive power mean: start smooth, get sharper
    p = p_start + (p_end - p_start) * min(1.0, _smooth_iter_counter[0] / 100.0)
    _smooth_iter_counter[0] += 1
    
    inv_areas = 1.0 / areas
    soft_min = (np.mean(inv_areas ** p)) ** (-1.0 / p)
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

_smooth_iter_counter = [0]

def _coordinate_descent_polish(points, max_iters=200, deadline=None):
    """Coordinate descent: optimize each point's x and y separately."""
    pts = points.copy()
    best_s = _score(pts)
    step = 0.01
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 2:
            break
        
        improved = False
        for i in range(N):
            # Optimize x-coordinate of point i
            for dx in [step, -step, step*0.5, -step*0.5]:
                trial = pts.copy()
                trial[i, 0] += dx
                s = _score(trial)
                if s > best_s + 1e-12:
                    pts = trial
                    best_s = s
                    improved = True
                    break
            
            # Optimize y-coordinate of point i
            for dy in [step, -step, step*0.5, -step*0.5]:
                trial = pts.copy()
                trial[i, 1] += dy
                s = _score(trial)
                if s > best_s + 1e-12:
                    pts = trial
                    best_s = s
                    improved = True
                    break
        
        if not improved:
            step *= 0.5
            if step < 1e-10:
                break
    
    return pts, best_s

def _local_polish(points, deadline):
    """Multi-method local polish: Nelder-Mead + coordinate descent."""
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    # Nelder-Mead with tight tolerances
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_pts = res.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    # Coordinate descent polish
    if time.time() < deadline - 3:
        cd_pts, cd_s = _coordinate_descent_polish(best_pts, max_iters=100, deadline=deadline)
        if cd_s > best_s:
            best_s = cd_s
            best_pts = cd_pts
    
    return best_pts, best_s
    """Apply affine transform: [x'] = [a b; c d] * [x; y]"""
    xs = points[:, 0]
    ys = points[:, 1]
    new_x = a * xs + b * ys
    new_y = c * xs + d * ys
    return np.column_stack([new_x, new_y])

def _grid_affine_objective(params):
    """Objective for optimizing affine transform of a grid configuration."""
    a, b, c, d = params
    # Use the best-performing grid base (4x4 minus one point, centered)
    base = _get_best_grid_base()
    pts = _affine_transform(base, a, b, c, d)
    score = _score(pts)
    if score <= 0:
        return 1e10
    return -score

def _get_best_grid_base():
    """Return a centered 4x4 grid minus one corner (13 points)."""
    # 4x4 grid centered at origin
    xs = np.array([-1.5, -0.5, 0.5, 1.5])
    ys = np.array([-1.5, -0.5, 0.5, 1.5])
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    # Remove one corner (top-right) to get 13 points
    pts = pts[:-1]
    # Center it
    pts = pts - np.mean(pts, axis=0)
    return pts

def _objective_full(x_flat):
    """Full objective: minimize negative score."""
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _de_objective(x_flat):
    """Objective for differential evolution: minimize negative score."""
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _run_differential_evolution(deadline, best_score_ref):
    """Run DE to find good starting points."""
    time_remaining = deadline - time.time()
    
    if time_remaining < 20:
        return None
    
    # DE bounds: points in [-2, 2] x [-2, 2]
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    # DE parameters tuned for this problem - more iterations for better global search
    de_params = {
        'maxiter': 80,
        'popsize': 25,
        'mutation': (0.3, 0.9),
        'recombination': 0.8,
        'tol': 1e-10,
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
        if result.success or result.fun < 1e9:
            return result.x
    except Exception:
        pass
    
    return None

def _optimize_affine_grid(deadline):
    """Optimize affine transform of grid configurations."""
    if time.time() > deadline - 8:
        return None
    
    best_score = 0.0
    best_points = None
    
    # Try several grid bases
    grid_bases = []
    
    # 4x4 minus corner
    base1 = _get_best_grid_base()
    grid_bases.append(base1)
    
    # 3x5 grid
    xs = np.array([-1.0, 0.0, 1.0])
    ys = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    gx, gy = np.meshgrid(xs, ys)
    base2 = np.column_stack([gx.ravel(), gy.ravel()])
    base2 = base2 - np.mean(base2, axis=0)
    grid_bases.append(base2)
    
    # 5x3 grid
    xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    ys = np.array([-1.0, 0.0, 1.0])
    gx, gy = np.meshgrid(xs, ys)
    base3 = np.column_stack([gx.ravel(), gy.ravel()])
    base3 = base3 - np.mean(base3, axis=0)
    grid_bases.append(base3)
    
    # Staggered grid (hexagonal-like)
    for offset in [0.0, 0.5]:
        rows = []
        for i in range(4):
            row_x = np.array([-1.5, -0.5, 0.5, 1.5]) + (offset if i % 2 else 0)
            row_y = np.full(4, i * 1.0 - 1.5)
            rows.append(np.column_stack([row_x, row_y]))
        base4 = np.vstack(rows)[:13]
        base4 = base4 - np.mean(base4, axis=0)
        grid_bases.append(base4)
    
    for base in grid_bases:
        if time.time() > deadline - 4:
            break
        
        # Start with identity transform
        x0 = np.array([1.0, 0.0, 0.0, 1.0])
        
        try:
            res = minimize(
                _grid_affine_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_points = _affine_transform(base, *res.x).copy()
        except Exception:
            pass
        
        # Also try with shear
        for shear in [0.2, -0.2, 0.3, -0.3, 0.4, -0.4]:
            if time.time() > deadline - 4:
                break
            x0_shear = np.array([1.0, shear, 0.0, 1.0])
            try:
                res = minimize(
                    _grid_affine_objective,
                    x0_shear,
                    method='Nelder-Mead',
                    options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
                s = -res.fun
                if s > best_score:
                    best_score = s
                    best_points = _affine_transform(base, *res.x).copy()
            except Exception:
                pass
    
    return best_points

def _make_ring_config(n_outer, n_inner, r_inner, angle_offset):
    """Create a two-ring configuration."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_inner, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([outer_pts, inner_pts])

def _optimize_from_start(x0, deadline, best_score_ref):
    """Run multi-phase optimization from a starting point."""
    best_x = x0.copy()
    best_s = _score(x0.reshape(N, 2))
    
    # Phase 1: Smooth optimization with adaptive surrogate
    if time.time() < deadline - 12:
        _smooth_iter_counter[0] = 0
        try:
            res = minimize(
                lambda x: _smooth_objective_adaptive(x, p_start=15, p_end=60),
                x0,
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            s_smooth = _score(res.x.reshape(N, 2))
            if s_smooth > best_s:
                best_s = s_smooth
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Hard objective Nelder-Mead polish
    if time.time() < deadline - 6:
        try:
            res_nm = minimize(
                _objective_full,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True}
            )
            s_nm = _score(res_nm.x.reshape(N, 2))
            if s_nm > best_s:
                best_s = s_nm
                best_x = res_nm.x.copy()
        except Exception:
            pass
    
    # Phase 3: Coordinate descent polish
    if time.time() < deadline - 4:
        cd_pts, cd_s = _coordinate_descent_polish(best_x.reshape(N, 2), max_iters=100, deadline=deadline)
        if cd_s > best_s:
            best_s = cd_s
            best_x = cd_pts.flatten()
    
    return best_x, best_s

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
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
    de_result = _run_differential_evolution(deadline, best_score)
    if de_result is not None:
        de_points = de_result.reshape(N, 2)
        _update_best(de_points)
        print(f"DE result: score={best_score:.6f}", flush=True)
    
    # Phase 1b: Optimize affine transforms of grid configurations
    if time.time() < deadline - 25:
        affine_points = _optimize_affine_grid(deadline)
        if affine_points is not None:
            _update_best(affine_points)
            print(f"Affine grid result: score={best_score:.6f}", flush=True)
    
    # Generate diverse initial configurations
    inits = []
    
    # Add DE result as first init if available
    if de_result is not None:
        inits.append(de_result.copy())
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]).flatten())
    
    # 2. Two-ring configurations with various parameters
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
    
    # 4. Random starts (increased count)
    rng = np.random.default_rng(12345)
    for _ in range(20):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts.flatten())
    
    # 5. Perturbed regular polygon (slight radial/angular noise)
    for noise_scale in [0.02, 0.05, 0.1, 0.15]:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0 + noise_scale * (rng.random(N) - 0.5)
        angles_perturbed = angles + noise_scale * (rng.random(N) - 0.5)
        pts = np.column_stack([radii * np.cos(angles_perturbed), radii * np.sin(angles_perturbed)])
        inits.append(pts.flatten())
    
    # 6. Grid-like configurations (approximating optimal packing)
    for grid_type in ['4x4_minus', '3x5']:
        if grid_type == '4x4_minus':
            xs = np.array([0, 1, 2, 3])
            ys = np.array([0, 1, 2, 3])
            gx, gy = np.meshgrid(xs, ys)
            pts_grid = np.column_stack([gx.ravel(), gy.ravel()])
            pts_grid = pts_grid[3:]
            if len(pts_grid) == N:
                pts_grid = pts_grid - np.mean(pts_grid, axis=0)
                inits.append(pts_grid.flatten())
        elif grid_type == '3x5':
            xs = np.array([0, 1, 2])
            ys = np.array([0, 1, 2, 3, 4])
            gx, gy = np.meshgrid(xs, ys)
            pts_grid = np.column_stack([gx.ravel(), gy.ravel()])
            if len(pts_grid) == N:
                pts_grid = pts_grid - np.mean(pts_grid, axis=0)
                inits.append(pts_grid.flatten())
    
    # 7. Golden angle spiral (quasi-crystalline)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for k in range(20):
        r = np.sqrt(np.arange(1, N + 1)) / np.sqrt(N)
        theta = np.arange(N) * golden + k * 0.1
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts.flatten())
    
    # Shuffle inits to diversify
    rng2 = np.random.default_rng(999)
    indices = rng2.permutation(len(inits))
    inits = [inits[i] for i in indices]
    
    # Optimize from each start
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 12:
            break
        
        t_start = time.time()
        x_opt, s_opt = _optimize_from_start(x0, deadline, best_score)
        _update_best(x_opt.reshape(N, 2))
        
        elapsed = time.time() - t_start
        if i % 5 == 0:
            print(f"Init {i}: score={best_score:.6f}, elapsed={elapsed:.1f}s", flush=True)
    
    # Final high-precision polish on best
    if best_points is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                _objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True}
            )
            _update_best(res_final.x.reshape(N, 2))
        except Exception:
            pass
        
        # Coordinate descent final polish
        if best_points is not None and time.time() < deadline - 3:
            cd_pts, cd_s = _coordinate_descent_polish(best_points, max_iters=200, deadline=deadline)
            if cd_s > best_score:
                best_score = cd_s
                best_points = cd_pts
    
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
