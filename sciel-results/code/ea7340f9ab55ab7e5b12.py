"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

# Precompute triangle indices
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])

def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles."""
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def _project_to_triangle(pts):
    """Project points to stay inside the equilateral triangle."""
    h = np.sqrt(3) / 2.0
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        # Constraint: y >= 0
        if y < 0:
            y = 0.0
        # Constraint: y <= h * x (left edge)
        if y > h * x:
            y = h * x
        # Constraint: y <= h * (1 - x) (right edge)
        if y > h * (1.0 - x):
            y = h * (1.0 - x)
        # Ensure x is within [0,1]
        if x < 0: x = 0.0
        if x > 1: x = 1.0
        pts[i] = [x, y]
    return pts

def _objective_flat(flat_pts):
    """Negative min triangle area for minimization."""
    pts = flat_pts.reshape(N, 2)
    pts = _project_to_triangle(pts.copy())
    return -_min_triangle_area(pts)

def _local_refine(pts, max_time=10.0):
    """Refine a point configuration using local optimization."""
    start_time = time.time()
    flat_pts = pts.flatten()
    
    # Try SLSQP first, then Powell if it fails or doesn't improve
    best_fun = _objective_flat(flat_pts)
    best_pts = pts.copy()
    
    methods = ['SLSQP', 'Powell']
    for method in methods:
        if time.time() - start_time > max_time:
            break
        try:
            result = minimize(
                _objective_flat,
                flat_pts,
                method=method,
                options={'maxiter': 500, 'ftol': 1e-12} if method == 'SLSQP' else {'maxiter': 500}
            )
            if result.fun < best_fun:
                best_fun = result.fun
                best_pts = result.x.reshape(N, 2)
                best_pts = _project_to_triangle(best_pts.copy())
        except Exception:
            pass
            
    return best_pts, -best_fun

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    h = np.sqrt(3) / 2.0
    
    # Known good starting configuration (symmetric arrangement)
    base_config = np.array([
        [0.5, 0.0],        # bottom center
        [0.0, 0.0],        # bottom left
        [1.0, 0.0],        # bottom right
        [0.25, h * 0.25],  # left mid
        [0.75, h * 0.25],  # right mid
        [0.5, h * 0.5],    # center
        [0.125, h * 0.125],# lower left
        [0.875, h * 0.125],# lower right
        [0.375, h * 0.375],# mid left
        [0.625, h * 0.375],# mid right
        [0.5, h * 0.875],  # top
    ])
    
    # Another symmetric configuration: vertices + midpoints + center
    sym_config = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.25, h/2],
        [0.75, h/2],
        [0.5, h/2],
        [0.5, 0.0],
        [0.25, h/4],
        [0.75, h/4],
        [0.5, 3*h/4],
        [0.5, h/4] # This might be a duplicate or close to center, let's vary
    ])
    # Let's use a random symmetric-like start instead
    rng = np.random.RandomState(42)
    
    best_points = base_config.copy()
    best_area = _min_triangle_area(best_points)
    
    # List of initial guesses
    initial_guesses = [base_config, sym_config]
    # Add a few random guesses
    for _ in range(5):
        # Generate random points inside triangle
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            if u + v > 1.0:
                u = 1.0 - u
                v = 1.0 - v
            pts[i] = [0.5 * u + 0.5 * (1.0 - u - v), 0.5 * np.sqrt(3) * v]
        initial_guesses.append(pts)

    # Strategy:
    # 1. Refine each initial guess locally.
    # 2. Use Differential Evolution for global search if time permits.
    
    for guess in initial_guesses:
        if time.time() > deadline - 20: # Leave 20s for DE
            break
        ref_pts, ref_area = _local_refine(guess, max_time=5.0)
        if ref_area > best_area:
            best_area = ref_area
            best_points = ref_pts.copy()

    # Global search using Differential Evolution
    # Bounds for x: [0, 1], for y: [0, h]
    # We use a rectangular bound and project in the objective.
    bounds = [(0.0, 1.0), (0.0, h)] * N
    
    if time.time() < deadline - 10:
        try:
            # DE is stochastic, so set seed
            # maxiter is number of generations
            # popsize is population size
            de_result = differential_evolution(
                _objective_flat,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                polish=True # This will call minimize at the end
            )
            
            de_pts = de_result.x.reshape(N, 2)
            de_pts = _project_to_triangle(de_pts.copy())
            de_area = _min_triangle_area(de_pts)
            
            if de_area > best_area:
                best_area = de_area
                best_points = de_pts.copy()
                
                # Refine the DE result further
                if time.time() < deadline - 5:
                    ref_pts, ref_area = _local_refine(best_points, max_time=5.0)
                    if ref_area > best_area:
                        best_area = ref_area
                        best_points = ref_pts.copy()
                        
        except Exception:
            pass
            
    return best_points
