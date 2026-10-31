import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13

# Pre-compute triangle indices for N=13
_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_I_IDX = _TRI_INDICES[:, 0]
_J_IDX = _TRI_INDICES[:, 1]
_K_IDX = _TRI_INDICES[:, 2]
_NUM_TRI = len(_TRI_INDICES)

def _get_triangle_areas(points):
    """Vectorized calculation of areas for all triples."""
    pi = points[_I_IDX]
    pj = points[_J_IDX]
    pk = points[_K_IDX]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the raw score: min_triangle_area / hull_area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _softmin_score(points, alpha):
    """Smooth approximation using softmin of triangle areas."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    a_min = np.min(areas)
    exp_terms = np.exp(-alpha * (areas - a_min))
    softmin_val = a_min - np.log(np.sum(exp_terms)) / alpha
    return softmin_val / hull_area

def _neg_softmin_score(params, alpha):
    """Objective for softmin-based optimization."""
    points = params.reshape(N, 2)
    s = _softmin_score(points, alpha)
    return -s

def _neg_score(params):
    """Objective function for minimization (maximizing score)."""
    points = params.reshape(N, 2)
    s = _score(points)
    return -s

def _fourier_to_points(params):
    """
    Convert Fourier parameters to 2D points.
    params: [a0, b0, a1, b1, a2, b2, ..., aK, bK, scale]
    First K harmonics + global scale.
    """
    K = 4  # number of harmonics
    n_params = 2 * (K + 1) + 1
    a0 = params[0]
    b0 = params[1]
    scale = params[-1]
    
    angles = 2 * np.pi * np.arange(N) / N
    x = np.zeros(N)
    y = np.zeros(N)
    
    for k in range(1, K + 1):
        ak = params[2 * k]
        bk = params[2 * k + 1]
        x += ak * np.cos(k * angles)
        y += bk * np.sin(k * angles)
    
    x = a0 + x
    y = b0 + y
    points = np.column_stack([x, y]) * scale
    return points

def _neg_fourier_softmin(params, alpha):
    """Fourier-parameterized softmin objective."""
    points = _fourier_to_points(params)
    s = _softmin_score(points, alpha)
    return -s

def _neg_fourier_hard(params):
    """Fourier-parameterized hard score objective."""
    points = _fourier_to_points(params)
    s = _score(points)
    return -s

def _optimize_fourier(start_params, deadline, rng):
    """Optimize in Fourier parameter space."""
    best_params = start_params.copy()
    best_score = _score(_fourier_to_points(start_params))
    
    # Stage 1: Softmin Nelder-Mead in Fourier space
    alphas = [200, 100, 50, 20, 10, 5, 2, 1]
    for alpha in alphas:
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(
                _neg_fourier_softmin,
                best_params,
                args=(alpha,),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
            )
            cand_params = res.x
            cand_pts = _fourier_to_points(cand_params)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_params = cand_params
        except Exception:
            pass
    
    # Stage 2: L-BFGS-B polish
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _neg_fourier_softmin,
                best_params,
                args=(1.0,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            cand_params = res.x
            cand_pts = _fourier_to_points(cand_params)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_params = cand_params
        except Exception:
            pass
    
    return _fourier_to_points(best_params), best_score

def _optimize_direct(start_pts, deadline, rng):
    """
    Multi-stage direct optimization:
    1. Softmin-based Nelder-Mead with annealing
    2. L-BFGS-B polish
    3. Nelder-Mead on hard score
    """
    best_pts = start_pts.copy()
    best_score = _score(start_pts)
    
    # Stage 1: Softmin optimization with annealing alpha
    alphas = [200, 100, 50, 20, 10, 5, 2, 1]
    for alpha in alphas:
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(
                _neg_softmin_score,
                best_pts.flatten(),
                args=(alpha,),
                method='Nelder-Mead',
                options={'maxiter': 600, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Stage 2: L-BFGS-B polish with softmin
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _neg_softmin_score,
                best_pts.flatten(),
                args=(1.0,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Stage 3: Nelder-Mead on hard score
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    return best_pts, best_score

def _polish_with_local_search(points, deadline):
    """
    Final high-precision polish: adaptive coordinate descent.
    Uses multiple step sizes and evaluates all points per sweep.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for step in step_sizes:
        if time.time() > deadline - 3:
            break
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(N):
                if time.time() > deadline - 3:
                    break
                
                orig = best_pts[i].copy()
                best_local = orig
                best_local_score = best_score
                
                # Try 8 directions + center
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        cand = best_pts.copy()
                        cand[i] = orig + np.array([dx, dy])
                        s = _score(cand)
                        if s > best_local_score + 1e-13:
                            best_local_score = s
                            best_local = orig + np.array([dx, dy])
                
                # Also try diagonal directions
                for diag in [1, -1]:
                    for d in [step * 0.7071]:
                        for sx, sy in [(d, d), (d, -d), (-d, d), (-d, -d)]:
                            cand = best_pts.copy()
                            cand[i] = orig + np.array([sx, sy])
                            s = _score(cand)
                            if s > best_local_score + 1e-13:
                                best_local_score = s
                                best_local = orig + np.array([sx, sy])
                
                if np.any(best_local != orig):
                    best_pts[i] = best_local
                    improved = True
        
        current_score = _score(best_pts)
        if current_score > best_score:
            best_score = current_score
    
    return best_pts, best_score

def _make_initial_configs(rng):
    """Generate diverse initial configurations."""
    configs = []
    
    # Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Regular polygon with slight radial perturbation
    for amp in [0.05, 0.1, 0.15, 0.2]:
        for freq in [2, 3, 5, 7]:
            angles = 2 * np.pi * np.arange(N) / N
            radii = 1.0 + amp * np.sin(freq * angles)
            configs.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # Two-ring structures
    for r1, r2 in [(0.3, 1.0), (0.4, 1.0), (0.5, 1.0), (0.5, 1.1), (0.6, 1.0)]:
        n_inner = N // 2
        n_outer = N - n_inner
        inner_angles = 2 * np.pi * np.arange(n_inner) / n_inner
        outer_angles = 2 * np.pi * np.arange(n_outer) / n_outer + np.pi / n_outer
        inner_pts = np.column_stack([r1 * np.cos(inner_angles), r1 * np.sin(inner_angles)])
        outer_pts = np.column_stack([r2 * np.cos(outer_angles), r2 * np.sin(outer_angles)])
        configs.append(np.vstack([inner_pts, outer_pts]))
    
    # Random configurations
    for _ in range(15):
        configs.append(rng.uniform(-1, 1, (N, 2)))
    
    # Near-regular with random angular jitter
    for jitter in [0.05, 0.1, 0.15]:
        angles = 2 * np.pi * np.arange(N) / N + rng.uniform(-jitter, jitter, N)
        angles = np.sort(angles)
        configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)
    
    # Generate initial configurations
    initial_configs = _make_initial_configs(rng)
    
    # Phase 1: Quick screening of all configs with brief optimization
    # Sort by initial score to prioritize good starts
    scored_configs = []
    for config in initial_configs:
        s = _score(config)
        scored_configs.append((s, config))
    scored_configs.sort(key=lambda x: -x[0])
    
    # Optimize top configs with direct method
    for s, config in scored_configs:
        if time.time() > deadline - 40:
            break
        
        try:
            cand_pts, score = _optimize_direct(config, deadline, rng)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            continue
    
    # Phase 2: Fourier-based optimization from best points
    if best_points is not None and time.time() < deadline - 30:
        # Convert best points to Fourier parameters
        angles = 2 * np.pi * np.arange(N) / N
        K = 4
        fourier_params = np.zeros(2 * (K + 1) + 1)
        
        # Fit Fourier coefficients
        for k in range(0, K + 1):
            if k == 0:
                fourier_params[0] = np.mean(best_points[:, 0])
                fourier_params[1] = np.mean(best_points[:, 1])
            else:
                fourier_params[2*k] = np.mean(best_points[:, 0] * np.cos(k * angles)) * 2 / N
                fourier_params[2*k + 1] = np.mean(best_points[:, 1] * np.sin(k * angles)) * 2 / N
        fourier_params[-1] = 1.0  # scale
        
        try:
            cand_pts, score = _optimize_fourier(fourier_params, deadline, rng)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            pass
    
    # Phase 3: Differential evolution for global search (if time permits)
    if time.time() < deadline - 35:
        try:
            bounds = [(-2, 2)] * (2 * N)
            de_result = differential_evolution(
                _neg_score,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol',
                update_interval=1,
                callback=None
            )
            cand_pts = de_result.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            pass
    
    # Phase 4: Perturbation restarts from best solution
    if best_points is not None and time.time() < deadline - 25:
        for i in range(10):
            if time.time() > deadline - 10:
                break
            
            noise_scale = 0.15 * (1 - i / 10.0) + 0.02
            perturbed = best_points + noise_scale * rng.normal(0, 1, (N, 2))
            
            try:
                cand_pts, score = _optimize_direct(perturbed, deadline, rng)
                if score > best_score:
                    best_score = score
                    best_points = cand_pts
            except Exception:
                continue
    
    # Phase 5: Final high-precision polish
    if best_points is not None and time.time() < deadline - 3:
        try:
            polished_pts, polished_score = _polish_with_local_search(best_points, deadline)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_pts
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
