import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLE_IDX)


def _triangle_areas_vectorized(points):
    """Compute all triangle areas using precomputed indices."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        if np.ptp(points, axis=0).max() < 1e-10:
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / hull area."""
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


def _soft_min_objective(x_flat, beta=80.0):
    """Smooth surrogate for -min(triangle_area)/hull_area using log-sum-exp."""
    points = x_flat.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    areas = _triangle_areas_vectorized(points)
    a_min_val = np.min(areas)
    if a_min_val < 1e-15:
        return 1e10
    ratios = areas / hull_area
    r_min = np.min(ratios)
    if r_min < 1e-15:
        return 1e10
    shifted = -beta * (ratios - r_min)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = r_min + log_sum / beta
    return -soft_min_val


def _hard_objective(x_flat):
    """Hard objective: -min_triangle_area / hull_area"""
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _simulated_annealing(x0, objective_fn, n_iter=20000, T0=0.01, T_end=1e-6, 
                          step_scale=0.05, rng=None):
    """Simulated annealing with adaptive step size."""
    if rng is None:
        rng = np.random.default_rng()
    
    x_best = x0.copy()
    f_best = objective_fn(x0)
    x_curr = x0.copy()
    f_curr = f_best
    
    n_dim = len(x0)
    
    for i in range(n_iter):
        # Linear cooling
        progress = i / n_iter
        T = T0 * (T_end / T0) ** progress
        
        # Adaptive step size
        step = step_scale * (0.5 + 0.5 * (1 - progress))
        
        # Generate candidate
        x_new = x_curr + rng.normal(0, step, size=n_dim)
        
        # Evaluate
        f_new = objective_fn(x_new)
        
        # Accept or reject
        if f_new < f_curr:
            x_curr = x_new
            f_curr = f_new
            if f_new < f_best:
                x_best = x_new.copy()
                f_best = f_new
        else:
            # Metropolis criterion
            delta = f_new - f_curr
            if delta < 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta / max(T, 1e-12))
            if rng.random() < accept_prob:
                x_curr = x_new
                f_curr = f_new
    
    return x_best, f_best


def _local_polish(x0, max_iter=2000):
    """Nelder-Mead local polish."""
    try:
        result = minimize(
            _soft_min_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        return result.x
    except Exception:
        return x0


def _generate_diverse_seeds(rng, n_seeds=8):
    """Generate diverse initial configurations."""
    seeds = []
    
    # Regular polygon
    seeds.append(_regular_polygon(N, 1.0).flatten())
    
    # Ring configurations with various parameters
    for i in range(n_seeds - 1):
        r_out = rng.uniform(0.8, 1.2)
        r_in = rng.uniform(0.2, 0.7)
        n_out = rng.integers(5, 10)
        n_in = N - n_out
        
        if n_in > 0:
            outer_angles = np.linspace(0, 2 * np.pi, n_out, endpoint=False) + rng.uniform(0, 2*np.pi)
            inner_angles = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + rng.uniform(0, 2*np.pi)
            
            outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
            inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
            pts = np.vstack([outer_pts, inner_pts])
            # Add small random perturbations
            pts += rng.normal(0, 0.02, size=pts.shape)
            seeds.append(pts.flatten())
        else:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 2*np.pi)
            radii = np.ones(N) * r_out
            radii += rng.normal(0, 0.05, size=N)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            seeds.append(pts.flatten())
    
    return seeds


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin against 150s limit
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # Generate diverse seeds
    rng = np.random.default_rng(42)
    seeds = _generate_diverse_seeds(rng, n_seeds=10)
    
    # Phase 1: Quick evaluation of all seeds
    for seed in seeds:
        pts = seed.reshape(N, 2)
        _update_best(pts)
    
    # Phase 2: Simulated annealing from top seeds
    # First, rank seeds by score
    seed_scores = []
    for seed in seeds:
        pts = seed.reshape(N, 2)
        s = _score(pts)
        seed_scores.append((s, seed))
    seed_scores.sort(key=lambda x: -x[0])
    
    # Take top 5 seeds for SA
    top_seeds = [s[1] for s in seed_scores[:5]]
    
    for idx, seed in enumerate(top_seeds):
        if time.time() > deadline - 30:
            break
        
        # Simulated annealing
        try:
            x_sa, f_sa = _simulated_annealing(
                seed, _soft_min_objective,
                n_iter=15000, T0=0.005, T_end=1e-7,
                step_scale=0.03, rng=np.random.default_rng(idx * 12345 + 6789)
            )
            pts_sa = x_sa.reshape(N, 2)
            _update_best(pts_sa)
        except Exception:
            pass
        
        # Local polish after SA
        if time.time() > deadline - 20:
            break
        try:
            x_polished = _local_polish(x_sa, max_iter=1000)
            pts_polished = x_polished.reshape(N, 2)
            _update_best(pts_polished)
        except Exception:
            pass
    
    # Phase 3: Additional SA runs from best found configuration with perturbations
    if best_points is not None and time.time() < deadline - 25:
        for perturb_idx in range(3):
            if time.time() > deadline - 20:
                break
            
            # Create perturbed version of best
            rng_p = np.random.default_rng(1000 + perturb_idx)
            x_perturbed = best_points.flatten() + rng_p.normal(0, 0.05, size=N * 2)
            
            try:
                x_sa2, f_sa2 = _simulated_annealing(
                    x_perturbed, _soft_min_objective,
                    n_iter=10000, T0=0.003, T_end=1e-7,
                    step_scale=0.02, rng=np.random.default_rng(2000 + perturb_idx)
                )
                pts_sa2 = x_sa2.reshape(N, 2)
                _update_best(pts_sa2)
                
                # Polish
                if time.time() > deadline - 15:
                    break
                x_polished2 = _local_polish(x_sa2, max_iter=800)
                pts_polished2 = x_polished2.reshape(N, 2)
                _update_best(pts_polished2)
            except Exception:
                pass
    
    # Phase 4: Final ultra-fine polish on best configuration
    if best_points is not None and time.time() < deadline - 10:
        try:
            x_final = _local_polish(best_points.flatten(), max_iter=3000)
            pts_final = x_final.reshape(N, 2)
            _update_best(pts_final)
        except Exception:
            pass
        
        # One more polish with tighter tolerance
        if time.time() < deadline - 5:
            try:
                x_final2 = minimize(
                    _soft_min_objective,
                    pts_final.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                pts_final2 = x_final2.x.reshape(N, 2)
                _update_best(pts_final2)
            except Exception:
                pass
    
    # Final fallback
    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
    
    return best_points
