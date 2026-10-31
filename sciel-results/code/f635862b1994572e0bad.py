"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

def _precompute_indices():
    """Precompute all (i,j,k) index triples for N points."""
    idxs = list(combinations(range(N), 3))
    return np.array(idxs)

_IDX = _precompute_indices()
_I = _IDX[:, 0]
_J = _IDX[:, 1]
_K = _IDX[:, 2]

def _min_triangle_area_vec(points):
    """
    Compute the minimum area of any triangle formed by three points.
    Vectorized for speed.
    points: (N, 2) array
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    cross = v1x * v2y - v1y * v2x
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _soft_min_triangle_area(points, beta=50.0):
    """
    Soft minimum of triangle areas using log-sum-exp for smoothness.
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    cross = v1x * v2y - v1y * v2x
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp for soft min
    m = np.min(areas)
    log_sum = m + np.log(np.sum(np.exp(areas - m)))
    soft_min = -log_sum / beta + m
    return soft_min

def _clamp_to_triangle_fast(points):
    """Clamp all points to be inside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    h = np.sqrt(3) / 2
    eps = 1e-9
    
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    y = np.clip(y, eps, h - eps)
    
    x_min_left = y / np.sqrt(3)
    x_max_right = 1.0 - y / np.sqrt(3)
    
    x = np.clip(x, x_min_left + eps, x_max_right - eps)
    
    points[:, 0] = x
    points[:, 1] = y
    return points

def _objective_smooth(x_flat, beta=50.0):
    """
    Smooth objective function for Nelder-Mead.
    Returns negative soft-min area (to minimize).
    """
    points = x_flat.reshape(N, 2)
    _clamp_to_triangle_fast(points)
    return -_soft_min_triangle_area(points, beta)

def _objective_hard(x_flat):
    """
    Hard objective: negative min area.
    """
    points = x_flat.reshape(N, 2)
    _clamp_to_triangle_fast(points)
    return -_min_triangle_area_vec(points)

def _nelder_mead_optimize(points, time_limit=5.0, beta=50.0):
    """
    Use Nelder-Mead with smooth surrogate to optimize.
    """
    points = np.array(points, dtype=float)
    _clamp_to_triangle_fast(points)
    
    x0 = points.flatten()
    start_time = time.time()
    
    # Adaptive beta: start lower, increase over time
    def adaptive_objective(x_flat):
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / time_limit)
        current_beta = 20.0 + 80.0 * progress  # 20 -> 100
        return -_soft_min_triangle_area(x_flat.reshape(N, 2), current_beta)
    
    try:
        result = minimize(
            adaptive_objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-7,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
    except Exception:
        return points, _min_triangle_area_vec(points)
    
    opt_points = result.x.reshape(N, 2)
    _clamp_to_triangle_fast(opt_points)
    score = _min_triangle_area_vec(opt_points)
    
    # If NM didn't improve, return original
    orig_score = _min_triangle_area_vec(points)
    if score > orig_score:
        return opt_points, score
    else:
        return points, orig_score

def _differential_evolution_global(time_limit=20.0, pop_size=40, seed=42):
    """
    Global search using differential evolution on the hard objective.
    """
    h = np.sqrt(3) / 2
    rng = np.random.default_rng(seed)
    
    # Bounds: each point must be inside the triangle
    # We'll use a penalty-based approach with simple bounds
    x_bounds = [(0, 1), (0, h)] * N
    
    start_time = time.time()
    
    # Initialize population
    def _random_in_triangle(rng):
        """Generate a random point inside the equilateral triangle."""
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1:
                x = u
                y = v * h
                return x, y
            else:
                u2 = 1 - u
                v2 = 1 - v
                if u2 + v2 <= 1:
                    x = u
                    y = (1 - v) * h
                    return x, y
                else:
                    continue
    
    # Better: use barycentric coordinates
    def _random_in_triangle_bary(rng):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        # Vertices: A=(0,0), B=(1,0), C=(0.5, h)
        x = r1 * 1.0 + r2 * 0.5
        y = r2 * h
        return x, y
    
    pop = np.zeros((pop_size, N, 2))
    for i in range(pop_size):
        for j in range(N):
            x, y = _random_in_triangle_bary(rng)
            pop[i, j, 0] = x
            pop[i, j, 1] = y
    
    # Evaluate initial population
    scores = np.array([-_min_triangle_area_vec(pop[i]) for i in range(pop_size)])
    best_idx = np.argmin(scores)
    best_points = pop[best_idx].copy()
    best_score = -scores[best_idx]
    
    F = 0.8  # scale factor
    CR = 0.9  # crossover rate
    
    gen = 0
    while time.time() - start_time < time_limit:
        gen += 1
        
        # Sort by score
        order = np.argsort(scores)
        pop_sorted = pop[order]
        scores_sorted = scores[order]
        
        for i in range(pop_size):
            if time.time() - start_time > time_limit:
                break
            
            # Select three distinct random individuals
            r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
            
            # Create trial vector
            trial = pop_sorted[r1] + F * (pop_sorted[r2] - pop_sorted[r3])
            _clamp_to_triangle_fast(trial)
            
            # Crossover
            j_rand = rng.integers(N * 2)
            for j in range(N * 2):
                if rng.random() < CR or j == j_rand:
                    pass  # keep trial
                else:
                    trial[j] = pop_sorted[i, j]
            
            trial_points = trial.reshape(N, 2)
            _clamp_to_triangle_fast(trial_points)
            trial_score = -_min_triangle_area_vec(trial_points)
            
            if trial_score <= scores_sorted[i]:
                pop_sorted[i] = trial_points.flatten()
                scores_sorted[i] = trial_score
                if trial_score < scores_sorted[0]:
                    scores_sorted[0] = trial_score
                    best_points = trial_points.copy()
                    best_score = -trial_score
        
        # Re-sort for next generation
        order = np.argsort(scores_sorted)
        pop = pop_sorted[order]
        scores = scores_sorted[order]
    
    return best_points, best_score

def _generate_structured_config():
    """Generate a well-structured initial configuration."""
    h = np.sqrt(3) / 2
    
    # Based on known Heilbronn triangle configurations
    # 3 corners + 3 edge midpoints + 5 interior points in hexagonal pattern
    pts = np.array([
        [0.0, 0.0],           # corner A
        [1.0, 0.0],           # corner B  
        [0.5, h],             # corner C
        [0.5, 0.0],           # midpoint AB
        [0.25, h/2],          # midpoint AC
        [0.75, h/2],          # midpoint BC
        [0.5, h/3],           # interior 1
        [0.333, h/3],         # interior 2
        [0.667, h/3],         # interior 3
        [0.5, 2*h/3],         # interior 4
        [0.5, h/6],           # interior 5
    ])
    _clamp_to_triangle_fast(pts)
    return pts

def _generate_grid_config():
    """Generate a grid-based initial configuration."""
    h = np.sqrt(3) / 2
    
    # Triangular grid pattern
    pts = []
    # Row 0 (bottom): 3 points
    for i in range(3):
        pts.append([i/2, 0.0])
    # Row 1: 2 points
    for i in range(2):
        pts.append([0.25 + i/2, h/3])
    # Row 2: 2 points
    for i in range(2):
        pts.append([0.25 + i/2, 2*h/3])
    # Row 3 (top): 1 point
    pts.append([0.5, h])
    # Add one more interior point
    pts.append([0.5, h/2])
    
    pts = np.array(pts[:N])
    _clamp_to_triangle_fast(pts)
    return pts

def _generate_random_config(seed):
    """Generate a random initial configuration."""
    rng = np.random.default_rng(seed)
    h = np.sqrt(3) / 2
    
    def _random_in_triangle_bary(rng):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        x = r1 * 1.0 + r2 * 0.5
        y = r2 * h
        return x, y
    
    pts = np.zeros((N, 2))
    for j in range(N):
        x, y = _random_in_triangle_bary(rng)
        pts[j, 0] = x
        pts[j, 1] = y
    
    _clamp_to_triangle_fast(pts)
    return pts

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: Global search with differential evolution
    if time.time() < deadline - 15:
        de_time = min(30.0, deadline - time.time() - 15)
        try:
            de_pts, de_score = _differential_evolution_global(
                time_limit=de_time, pop_size=30, seed=12345
            )
            if de_score > best_score:
                best_score = de_score
                best_points = de_pts.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead on best + structured configs
    configs_to_refine = []
    if best_points is not None:
        configs_to_refine.append(best_points)
    configs_to_refine.append(_generate_structured_config())
    configs_to_refine.append(_generate_grid_config())
    configs_to_refine.append(_generate_random_config(42))
    configs_to_refine.append(_generate_random_config(999))
    
    n_refine = len(configs_to_refine)
    for i, config in enumerate(configs_to_refine):
        if time.time() > deadline - 2:
            break
        
        time_left = deadline - time.time() - 2
        nm_time = min(8.0, time_left / max(1, n_refine - i))
        
        if nm_time < 1.0:
            break
        
        try:
            opt_pts, score = _nelder_mead_optimize(config, time_limit=nm_time)
            if score > best_score:
                best_score = score
                best_points = opt_pts.copy()
        except Exception:
            continue
    
    # Phase 3: Final DE polish on the best
    if best_points is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        if remaining > 3.0:
            try:
                # Run DE starting near the best point
                h = np.sqrt(3) / 2
                rng = np.random.default_rng(777)
                pop_size = 20
                F = 0.5
                CR = 0.8
                
                pop = np.zeros((pop_size, N, 2))
                pop[0] = best_points
                for i in range(1, pop_size):
                    # Perturb best point
                    noise = rng.normal(0, 0.05, (N, 2))
                    pop[i] = best_points + noise
                    _clamp_to_triangle_fast(pop[i])
                
                scores = np.array([-_min_triangle_area_vec(pop[i]) for i in range(pop_size)])
                local_best_idx = np.argmin(scores)
                local_best_pts = pop[local_best_idx].copy()
                local_best_score = -scores[local_best_idx]
                
                gen = 0
                while time.time() - start_time < deadline - 3:
                    gen += 1
                    order = np.argsort(scores)
                    pop_s = pop[order]
                    scores_s = scores[order]
                    
                    for i in range(pop_size):
                        if time.time() - start_time > deadline - 3:
                            break
                        r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                        trial = pop_s[r1] + F * (pop_s[r2] - pop_s[r3])
                        _clamp_to_triangle_fast(trial)
                        
                        trial_pts = trial.reshape(N, 2)
                        _clamp_to_triangle_fast(trial_pts)
                        trial_score = -_min_triangle_area_vec(trial_pts)
                        
                        if trial_score <= scores_s[i]:
                            pop_s[i] = trial_pts.flatten()
                            scores_s[i] = trial_score
                            if trial_score < scores_s[0]:
                                scores_s[0] = trial_score
                                local_best_pts = trial_pts.copy()
                                local_best_score = -trial_score
                    
                    order = np.argsort(scores_s)
                    pop = pop_s[order]
                    scores = scores_s[order]
                
                if local_best_score > best_score:
                    best_score = local_best_score
                    best_points = local_best_pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        best_points = _generate_structured_config()
    
    return best_points

# EVOLVE-BLOCK-END
