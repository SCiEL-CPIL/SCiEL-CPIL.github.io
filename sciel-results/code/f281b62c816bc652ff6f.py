import time
import numpy as np
from scipy.optimize import minimize
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
    """
    Vectorized calculation of areas for all triples.
    """
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
    """
    Smooth approximation of the score using softmin of triangle areas.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    
    a_min = np.min(areas)
    exp_terms = np.exp(-alpha * (areas - a_min))
    log_sum = np.log(np.sum(exp_terms))
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

def _polar_to_cartesian(params):
    """
    Convert polar parameters to Cartesian points.
    params: [r0, a1, b1, a2, b2, a3, b3, cx, cy] (9 params)
    """
    r0 = params[0]
    a1, b1 = params[1], params[2]
    a2, b2 = params[3], params[4]
    a3, b3 = params[5], params[6]
    cx, cy = params[7], params[8]
    
    angles = 2 * np.pi * np.arange(N) / N
    radii = (r0 + a1 * np.cos(angles) + b1 * np.sin(angles) + 
             a2 * np.cos(2 * angles) + b2 * np.sin(2 * angles) +
             a3 * np.cos(3 * angles) + b3 * np.sin(3 * angles))
    
    radii = np.maximum(radii, 0.1)
    
    points = np.column_stack([radii * np.cos(angles) + cx, radii * np.sin(angles) + cy])
    return points

def _neg_polar_score(params, alpha):
    """
    Objective for polar-parameterized optimization.
    """
    points = _polar_to_cartesian(params)
    s = _softmin_score(points, alpha)
    return -s

def _neg_polar_score_hard(params):
    """
    Objective for polar-parameterized optimization (hard score).
    """
    points = _polar_to_cartesian(params)
    s = _score(points)
    return -s

def _penalty_t_objective(params, t, mu, alpha):
    """
    Penalty-based objective: maximize t subject to area_i/hull_area >= t for all i.
    Uses smooth penalty: t - mu * sum(smooth_max(0, t - area_i/hull_area)^2)
    """
    points = params[:N*2].reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 1e6
    
    norm_areas = areas / hull_area
    
    # Smooth penalty for violations
    violations = t - norm_areas
    # smooth_max(0, x) ≈ (x + sqrt(x^2 + eps)) / 2
    eps = 1e-6
    smooth_viol = (violations + np.sqrt(violations**2 + eps)) / 2.0
    
    penalty = mu * np.sum(smooth_viol**2)
    
    # Also penalize if t is too large relative to actual min
    return -(t) + penalty

def _penalty_t_optimize(start_pts, deadline, rng):
    """
    Optimize using penalty-based t formulation.
    """
    best_pts = start_pts.copy()
    best_score = _score(start_pts)
    
    # Initial t estimate
    t = best_score * 0.5
    
    # Penalty weight schedule
    mu_schedule = [10, 50, 200, 1000, 5000]
    
    for mu in mu_schedule:
        if time.time() > deadline - 10:
            break
        
        # Optimize over (points, t) jointly
        # params = [x0, y0, x1, y1, ..., x12, y12, t]
        init_params = np.concatenate([best_pts.flatten(), [t]])
        
        try:
            res = minimize(
                _penalty_t_objective,
                init_params,
                args=(t, mu, 10.0),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
            )
            cand_params = res.x
            cand_pts = cand_params[:N*2].reshape(N, 2)
            cand_t = cand_params[N*2]
            
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts
                t = max(t * 0.9, cand_score * 0.5)
        except Exception:
            pass
    
    return best_pts, best_score

def _coordinate_descent_polish(points, deadline, rng):
    """
    High-precision coordinate descent polish.
    For each point, optimize its position while keeping others fixed.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    
    for outer_iter in range(100):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            
            orig = best_pts[i].copy()
            best_local = orig
            best_local_score = best_score
            
            # Adaptive step sizes
            step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            
            for step in step_sizes:
                # Try 8 directions + center
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        cand = best_pts.copy()
                        cand[i] = orig + np.array([dx, dy])
                        s = _score(cand)
                        if s > best_local_score + 1e-12:
                            best_local_score = s
                            best_local = orig + np.array([dx, dy])
            
            # Also try random perturbations
            for _ in range(5):
                dx, dy = rng.normal(0, 0.02, 2)
                cand = best_pts.copy()
                cand[i] = orig + np.array([dx, dy])
                s = _score(cand)
                if s > best_local_score + 1e-12:
                    best_local_score = s
                    best_local = orig + np.array([dx, dy])
            
            if np.any(best_local != orig):
                best_pts[i] = best_local
                improved = True
        
        if not improved:
            break
        
        current_score = _score(best_pts)
        if current_score > best_score:
            best_score = current_score
    
    return best_pts, best_score

def _local_polish(points, deadline, rng):
    """
    Local polish using Nelder-Mead on hard score with tight tolerances.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    
    try:
        res = minimize(
            _neg_score,
            best_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        cand_pts = res.x.reshape(N, 2)
        score = _score(cand_pts)
        if score > best_score:
            best_score = score
            best_pts = cand_pts
    except Exception:
        pass
    
    return best_pts, best_score

def _make_ring_config(n_points, radius, phase_offset=0.0, n_rings=1, ring_radii=None):
    """Generate a ring-based initial configuration."""
    if ring_radii is None:
        if n_rings == 1:
            ring_radii = [radius]
        else:
            ring_radii = np.linspace(0.3, radius, n_rings).tolist()
    
    points = []
    per_ring = n_points // n_rings
    remainder = n_points % n_rings
    
    for ring_idx, r in enumerate(ring_radii):
        count = per_ring + (1 if ring_idx < remainder else 0)
        for i in range(count):
            angle = 2 * np.pi * i / count + phase_offset + ring_idx * 0.3
            points.append([r * np.cos(angle), r * np.sin(angle)])
    
    while len(points) < n_points:
        idx = len(points)
        angle = 2 * np.pi * idx / n_points + phase_offset
        points.append([radius * 0.5 * np.cos(angle), radius * 0.5 * np.sin(angle)])
    
    return np.array(points[:n_points])

def _optimize_polar(deadline, rng):
    """
    Optimize using polar parameterization (reduced dimensionality).
    """
    best_score = 0.0
    best_params = None
    best_points = None
    
    # Initial parameter guesses
    initial_params_list = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Regular polygon
        [1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0],
        [0.95, 0.03, 0.02, 0.02, 0.01, 0.01, 0.005, 0.0, 0.0],
        [1.05, -0.02, 0.03, 0.01, -0.02, 0.015, -0.01, 0.0, 0.0],
    ]
    
    # Add random initializations
    for _ in range(8):
        p = [1.0 + rng.normal(0, 0.1)]
        for _ in range(6):
            p.append(rng.normal(0, 0.05))
        p.extend([rng.normal(0, 0.05), rng.normal(0, 0.05)])
        initial_params_list.append(p)
    
    for init_params in initial_params_list:
        if time.time() > deadline - 15:
            break
        
        init_params = np.array(init_params, dtype=float)
        
        current_params = init_params.copy()
        current_score = _score(_polar_to_cartesian(current_params))
        
        # Multi-stage optimization in polar space
        for alpha in [50, 20, 10, 5, 2]:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    _neg_polar_score,
                    current_params,
                    args=(alpha,),
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                cand_params = res.x
                cand_score = _score(_polar_to_cartesian(cand_params))
                if cand_score > current_score:
                    current_score = cand_score
                    current_params = cand_params
            except Exception:
                pass
        
        # Hard score polish
        if time.time() < deadline - 5:
            try:
                res = minimize(
                    _neg_polar_score_hard,
                    current_params,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
                )
                cand_params = res.x
                cand_score = _score(_polar_to_cartesian(cand_params))
                if cand_score > current_score:
                    current_score = cand_score
                    current_params = cand_params
            except Exception:
                pass
        
        if current_score > best_score:
            best_score = current_score
            best_params = current_params.copy()
            best_points = _polar_to_cartesian(current_params)
    
    return best_points, best_score

def _optimize_from_start(start_pts, deadline, rng):
    """
    Multi-stage optimization from a starting point.
    """
    best_pts = start_pts.copy()
    best_score = _score(start_pts)
    
    # Stage 1: Softmin optimization with annealing alpha
    for alpha in [100, 50, 20, 10, 5, 2]:
        if time.time() > deadline - 10:
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
    
    # Stage 2: Penalty-t optimization
    if time.time() < deadline - 15:
        try:
            cand_pts, score = _penalty_t_optimize(best_pts, deadline, rng)
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s kill
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)
    
    # FIRST: Try polar-parameterized optimization
    if time.time() < deadline - 40:
        try:
            polar_pts, polar_score = _optimize_polar(deadline, rng)
            if polar_pts is not None and polar_score > best_score:
                best_score = polar_score
                best_points = polar_pts
        except Exception:
            pass
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # If we have a good polar result, use it
    if best_points is not None:
        initial_configs.append(best_points.copy())
    
    # Ring configurations
    for radius in [1.0, 1.2, 0.8]:
        for phase in [0.0, np.pi / N, np.pi / (2 * N)]:
            initial_configs.append(_make_ring_config(N, radius, phase_offset=phase))
    
    # Two-ring structures
    for r1, r2 in [(0.5, 1.0), (0.4, 1.1), (0.6, 0.9)]:
        for phase in [0.0, np.pi / N]:
            initial_configs.append(_make_ring_config(N, 1.0, phase_offset=phase, n_rings=2, ring_radii=[r1, r2]))
    
    # Radial perturbation of regular polygon
    for amp in [0.1, 0.2, 0.3]:
        for freq in [2, 3, 5]:
            angles = 2 * np.pi * np.arange(N) / N
            radii = 1.0 + amp * np.sin(freq * angles)
            initial_configs.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # Random configurations
    for _ in range(5):
        initial_configs.append(rng.uniform(-1, 1, (N, 2)))
    
    # Try each initial configuration
    for config in initial_configs:
        if time.time() > deadline - 25:
            break
        
        try:
            cand_pts, score = _optimize_from_start(config, deadline, rng)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            continue
    
    # Perturbation-based restarts from best
    if best_points is not None and time.time() < deadline - 35:
        for i in range(12):
            if time.time() > deadline - 12:
                break
            
            noise_scale = 0.15 * (1 - i / 12.0) + 0.01
            perturbed = best_points + noise_scale * rng.normal(0, 1, (N, 2))
            
            try:
                cand_pts, score = _optimize_from_start(perturbed, deadline, rng)
                if score > best_score:
                    best_score = score
                    best_points = cand_pts
            except Exception:
                continue
    
    # Coordinate descent polish
    if best_points is not None and time.time() < deadline - 10:
        try:
            polished_pts, polished_score = _coordinate_descent_polish(best_points, deadline, rng)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_pts
        except Exception:
            pass
    
    # Final local polish
    if best_points is not None and time.time() < deadline - 5:
        try:
            polished_pts, polished_score = _local_polish(best_points, deadline, rng)
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
