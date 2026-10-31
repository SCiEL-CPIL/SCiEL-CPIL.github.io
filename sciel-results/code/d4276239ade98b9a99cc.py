import numpy as np
import time
import warnings
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys

warnings.filterwarnings('ignore')

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


def _neg_score(x_flat):
    """Negative score for minimization."""
    points = x_flat.reshape(N, 2)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _soft_min_value(ratios, T):
    """
    Compute soft minimum using temperature-based log-sum-exp.
    As T -> 0, this approaches the true minimum.
    soft_min = r_min - T * log(sum(exp(-(r_i - r_min)/T)))
    """
    r_min = np.min(ratios)
    if T < 1e-10:
        return r_min
    shifted = -(ratios - r_min) / T
    # Clip for numerical stability
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return r_min - T * log_sum


def _neg_soft_score(x_flat, T):
    """Negative soft score for minimization."""
    points = x_flat.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 1e10
    ratios = areas / hull_area
    soft_min = _soft_min_value(ratios, T)
    if soft_min < 1e-15:
        return 1e10
    return -soft_min


def _seed_regular(n, radius=1.0):
    """Regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _seed_two_ring(r_outer=1.0, r_inner=0.5, n_outer=7, n_inner=6, 
                   a_outer=0.0, a_inner=np.pi/6):
    """Two concentric rings."""
    outer_angles = np.linspace(a_outer, 2 * np.pi + a_outer, n_outer, endpoint=False)
    inner_angles = np.linspace(a_inner, 2 * np.pi + a_inner, n_inner, endpoint=False)
    outer_pts = np.column_stack([r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles)])
    inner_pts = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    return np.vstack([outer_pts, inner_pts])


def _seed_three_ring(r1=1.0, r2=0.65, r3=0.3, n1=5, n2=4, n3=4,
                     a1=0.0, a2=np.pi/5, a3=np.pi/4):
    """Three concentric rings."""
    a1_angles = np.linspace(a1, 2 * np.pi + a1, n1, endpoint=False)
    a2_angles = np.linspace(a2, 2 * np.pi + a2, n2, endpoint=False)
    a3_angles = np.linspace(a3, 2 * np.pi + a3, n3, endpoint=False)
    p1 = np.column_stack([r1 * np.cos(a1_angles), r1 * np.sin(a1_angles)])
    p2 = np.column_stack([r2 * np.cos(a2_angles), r2 * np.sin(a2_angles)])
    p3 = np.column_stack([r3 * np.cos(a3_angles), r3 * np.sin(a3_angles)])
    return np.vstack([p1, p2, p3])


def _seed_quasi_regular():
    """Quasi-regular: 13 points with near-uniform spacing but slight perturbations."""
    # Start with regular 13-gon, add small radial perturbations
    base = _seed_regular(13, 1.0)
    rng = np.random.default_rng(42)
    radial_perturb = 1.0 + rng.uniform(-0.05, 0.05, 13)
    return base * radial_perturb[:, np.newaxis]


def _annealed_soft_minimize(x0, T_start=1.0, T_end=0.01, n_steps=50, max_iter_per_step=200):
    """
    Annealed soft-minimization: gradually reduce temperature T from T_start to T_end,
    optimizing at each step. This helps escape local minima.
    """
    current_x = x0.copy()
    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        T = T_start * (T_end / T_start) ** frac
        
        def objective(x):
            return _neg_soft_score(x, T)
        
        try:
            result = minimize(
                objective,
                current_x,
                method='Nelder-Mead',
                options={'maxiter': max_iter_per_step, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            current_x = result.x
        except Exception:
            pass
    
    return current_x


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin against 120s limit
    
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
    
    # Generate diverse seed configurations
    seeds = []
    
    # Regular 13-gon
    seeds.append(_seed_regular(13, 1.0))
    
    # Two-ring configurations with different parameters
    seeds.append(_seed_two_ring(1.0, 0.5, 7, 6, 0.0, 0.0))
    seeds.append(_seed_two_ring(1.0, 0.5, 7, 6, 0.0, np.pi/7))
    seeds.append(_seed_two_ring(1.0, 0.6, 7, 6, 0.0, np.pi/6))
    seeds.append(_seed_two_ring(1.0, 0.4, 7, 6, 0.0, np.pi/7))
    seeds.append(_seed_two_ring(1.0, 0.5, 8, 5, 0.0, np.pi/8))
    seeds.append(_seed_two_ring(1.0, 0.5, 8, 5, 0.0, 0.0))
    seeds.append(_seed_two_ring(1.0, 0.5, 6, 7, 0.0, np.pi/6))
    seeds.append(_seed_two_ring(1.0, 0.5, 6, 7, 0.0, 0.0))
    
    # Three-ring configurations
    seeds.append(_seed_three_ring(1.0, 0.65, 0.3, 5, 4, 4, 0.0, np.pi/5, np.pi/4))
    seeds.append(_seed_three_ring(1.0, 0.6, 0.3, 5, 4, 4, 0.0, np.pi/5, 0.0))
    seeds.append(_seed_three_ring(1.0, 0.7, 0.35, 6, 4, 3, 0.0, np.pi/6, np.pi/4))
    seeds.append(_seed_three_ring(1.0, 0.65, 0.3, 4, 5, 4, 0.0, np.pi/4, np.pi/5))
    
    # Quasi-regular
    seeds.append(_seed_quasi_regular())
    
    # Random seeds
    for seed_idx in range(5):
        rng = np.random.default_rng(seed_idx * 12345 + 67890)
        angles = np.sort(rng.uniform(0, 2 * np.pi, 13))
        radii = rng.uniform(0.3, 1.0, 13)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        seeds.append(pts)
    
    # Evaluate all seeds
    for seed_pts in seeds:
        _update_best(seed_pts)
    
    # Phase 1: Annealed soft-minimization from each seed
    for i, seed_pts in enumerate(seeds):
        if time.time() > deadline - 30:
            break
        
        x0 = seed_pts.flatten()
        
        try:
            # Use shorter annealing for more seeds
            n_steps = 30 if i < 8 else 20
            max_iter = 150 if i < 8 else 100
            
            result_x = _annealed_soft_minimize(
                x0, 
                T_start=0.5, 
                T_end=0.01, 
                n_steps=n_steps, 
                max_iter_per_step=max_iter
            )
            opt_points = result_x.reshape(N, 2)
            _update_best(opt_points)
        except Exception:
            pass
    
    # Phase 2: Local refinement with hard objective from best candidates
    if best_points is not None and time.time() < deadline - 20:
        x0 = best_points.flatten()
        
        # Multiple local refinement attempts with different initial perturbations
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            if attempt == 0:
                x_start = x0.copy()
            else:
                rng = np.random.default_rng(attempt * 999 + 1)
                x_start = x0 + rng.normal(0, 0.01, x0.shape)
            
            try:
                result = minimize(
                    _neg_score,
                    x_start,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                opt_points = result.x.reshape(N, 2)
                _update_best(opt_points)
            except Exception:
                pass
    
    # Phase 3: Additional annealed optimization from best point with different T schedules
    if best_points is not None and time.time() < deadline - 15:
        x0 = best_points.flatten()
        
        # Different temperature schedules
        T_schedules = [
            (1.0, 0.005, 40),
            (0.3, 0.005, 30),
            (0.8, 0.01, 35),
        ]
        
        for T_start, T_end, n_steps in T_schedules:
            if time.time() > deadline - 8:
                break
            
            try:
                result_x = _annealed_soft_minimize(
                    x0,
                    T_start=T_start,
                    T_end=T_end,
                    n_steps=n_steps,
                    max_iter_per_step=100
                )
                opt_points = result_x.reshape(N, 2)
                _update_best(opt_points)
            except Exception:
                pass
    
    # Phase 4: Final polish if time permits
    if best_points is not None and time.time() < deadline - 5:
        x0 = best_points.flatten()
        
        try:
            result = minimize(
                _neg_score,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            opt_points = result.x.reshape(N, 2)
            _update_best(opt_points)
        except Exception:
            pass
    
    # Final check
    if best_points is None:
        best_points = _seed_regular(13, 1.0)
    
    return best_points
