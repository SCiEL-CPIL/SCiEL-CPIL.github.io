"""Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 13
DEADLINE = 120.0  # seconds, with safety margin
START_TIME = time.time()

# Pre-compute all triangle index combinations
def _precompute_triangles(n):
    tris = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                tris.append((i, j, k))
    return np.array(tris)

TRIANGLES = _precompute_triangles(N)

def _min_triangle_area(points):
    """Compute minimum triangle area for N points using precomputed indices."""
    pi = points[TRIANGLES[:, 0]]
    pj = points[TRIANGLES[:, 1]]
    pk = points[TRIANGLES[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min_triangle_area / hull_area."""
    ma = _min_triangle_area(points)
    ha = _hull_area(points)
    if ha < 1e-10:
        return 0.0
    return ma / ha

def _objective(x):
    """Negative score for minimization."""
    points = x.reshape(N, 2)
    return -_score(points)

def _nelder_mead_optimize(x0, maxiter=2000, time_limit=None, start_time=None):
    """Run Nelder-Mead optimization with time guard."""
    if start_time is None:
        start_time = time.time()
    
    options = {
        'maxiter': maxiter,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'adaptive': True,
    }
    
    def callback(xk):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            raise TimeoutError()
    
    try:
        result = minimize(_objective, x0, method='Nelder-Mead', options=options, callback=callback)
        return result.x
    except TimeoutError:
        return x0
    except Exception:
        return x0

def _coordinate_descent_polish(points, n_iter=200, step_init=0.01, time_limit=None, start_time=None):
    """Polish with coordinate descent on each coordinate."""
    if start_time is None:
        start_time = time.time()
    
    x = points.flatten().copy()
    best_score = _score(x.reshape(N, 2))
    step = step_init
    
    for iteration in range(n_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        improved = False
        for i in range(2 * N):
            if time_limit is not None and (time.time() - start_time) > time_limit:
                break
            
            # Try increasing
            x_try = x.copy()
            x_try[i] += step
            s_try = _score(x_try.reshape(N, 2))
            if s_try > best_score:
                x = x_try
                best_score = s_try
                improved = True
            # Try decreasing
            x_try = x.copy()
            x_try[i] -= step
            s_try = _score(x_try.reshape(N, 2))
            if s_try > best_score:
                x = x_try
                best_score = s_try
                improved = True
        
        if not improved:
            step *= 0.7
            if step < 1e-10:
                break
    
    return x.reshape(N, 2)

def _simulated_annealing(points, n_iter=5000, T_start=0.1, T_end=1e-4, time_limit=None, start_time=None):
    """Simulated annealing to escape local minima."""
    if start_time is None:
        start_time = time.time()
    
    x = points.flatten().copy()
    current_score = _score(x.reshape(N, 2))
    best_x = x.copy()
    best_score = current_score
    
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / max(n_iter, 1))
    
    for i in range(n_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Generate a perturbation
        x_try = x + np.random.normal(0, T, size=x.shape)
        try_score = _score(x_try.reshape(N, 2))
        
        # Accept if better or with probability exp((new-old)/T)
        if try_score > current_score:
            x = x_try
            current_score = try_score
            if try_score > best_score:
                best_score = try_score
                best_x = x.copy()
        else:
            delta = try_score - current_score
            if np.random.random() < np.exp(delta / max(T, 1e-10)):
                x = x_try
                current_score = try_score
        
        T *= cooling
    
    return best_x.reshape(N, 2)

def _make_regular_polygon(n, radius=0.5, center=(0.5, 0.5)):
    """Create a regular n-gon centered at center with given radius."""
    angles = 2 * np.pi * np.arange(n) / n
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])

def _make_hybrid_config(n_hull, n_interior, rng):
    """Create a configuration with n_hull points on a regular polygon and n_interior interior points."""
    # Points on regular polygon
    hull_points = _make_regular_polygon(n_hull, radius=0.5, center=(0.5, 0.5))
    
    # Interior points: use a smaller regular polygon or random placement
    interior_points = []
    if n_interior > 0:
        # Place interior points in a smaller regular polygon pattern
        n_inner_ring = min(n_interior, 5)
        inner_radius = 0.2
        inner_angles = 2 * np.pi * np.arange(n_inner_ring) / n_inner_ring + np.pi / n_inner_ring
        for i in range(n_inner_ring):
            interior_points.append([0.5 + inner_radius * np.cos(inner_angles[i]),
                                   0.5 + inner_radius * np.sin(inner_angles[i])])
        # Remaining interior points near center
        for i in range(n_interior - n_inner_ring):
            angle = 2 * np.pi * i / max(n_interior - n_inner_ring, 1) + 0.3
            r = 0.05
            interior_points.append([0.5 + r * np.cos(angle),
                                   0.5 + r * np.sin(angle)])
    
    points = np.vstack([hull_points, np.array(interior_points)]) if interior_points else hull_points
    
    # Add small random perturbation
    points += rng.normal(0, 0.02, size=points.shape)
    
    return points

def _make_grid_config(rng):
    """Create a grid-based configuration with 13 points."""
    # 4x4 grid minus 3 points
    xs = [0.1, 0.37, 0.63, 0.9]
    ys = [0.1, 0.37, 0.63, 0.9]
    grid = [(x, y) for x in xs for y in ys]
    removed = rng.sample(range(16), 3)
    points = []
    for idx, (x, y) in enumerate(grid):
        if idx not in removed:
            points.append([x + rng.gauss(0, 0.01), y + rng.gauss(0, 0.01)])
    return np.array(points)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # Safety margin
    
    rng = np.random.RandomState(42)
    import random
    rng_py = random.Random(42)
    
    best_points = None
    best_score = -np.inf
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Regular 13-gon (all points on hull)
    initial_configs.append(_make_regular_polygon(N, radius=0.5, center=(0.5, 0.5)))
    
    # 2. 10-gon hull + 3 interior points
    initial_configs.append(_make_hybrid_config(10, 3, rng))
    
    # 3. 12-gon hull + 1 interior point
    initial_configs.append(_make_hybrid_config(12, 1, rng))
    
    # 4. 8-gon hull + 5 interior points
    initial_configs.append(_make_hybrid_config(8, 5, rng))
    
    # 5. Grid configuration
    initial_configs.append(_make_grid_config(rng))
    
    # 6. Random configuration
    initial_configs.append(np.array([[rng_py.random(), rng_py.random()] for _ in range(N)]))
    
    # 7. Another regular polygon with different radius
    initial_configs.append(_make_regular_polygon(N, radius=0.45, center=(0.5, 0.5)))
    
    # 8. Hybrid with 11-gon + 2 interior
    initial_configs.append(_make_hybrid_config(11, 2, rng))
    
    # Optimize each initial configuration
    for init_idx, x0 in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        x0 = x0.flatten()
        
        # First: Nelder-Mead for local refinement
        x_opt = _nelder_mead_optimize(x0, maxiter=300, time_limit=10.0, start_time=start_time)
        
        # Second: Simulated annealing to explore broader landscape
        points = x_opt.reshape(N, 2)
        points_sa = _simulated_annealing(points, n_iter=2000, T_start=0.05, T_end=1e-4,
                                          time_limit=10.0, start_time=start_time)
        
        # Third: Polish with coordinate descent
        points_final = _coordinate_descent_polish(points_sa, n_iter=100, step_init=0.01,
                                                   time_limit=5.0, start_time=start_time)
        
        s = _score(points_final)
        if s > best_score:
            best_score = s
            best_points = points_final.copy()
    
    # If we have time, do additional rounds with perturbed best
    if best_points is not None and time.time() < deadline - 20:
        for trial in range(5):
            if time.time() > deadline:
                break
            
            # Perturb the best solution
            x_pert = best_points.flatten() + rng.normal(0, 0.03, size=2 * N)
            
            x_opt = _nelder_mead_optimize(x_pert, maxiter=200, time_limit=8.0, start_time=start_time)
            points = x_opt.reshape(N, 2)
            points_sa = _simulated_annealing(points, n_iter=1000, T_start=0.03, T_end=1e-4,
                                              time_limit=8.0, start_time=start_time)
            points_final = _coordinate_descent_polish(points_sa, n_iter=50, step_init=0.005,
                                                       time_limit=3.0, start_time=start_time)
            
            s = _score(points_final)
            if s > best_score:
                best_score = s
                best_points = points_final.copy()
    
    if best_points is None:
        # Fallback: random points
        best_points = np.array([[rng_py.random(), rng_py.random()] for _ in range(N)])
    
    return best_points

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
