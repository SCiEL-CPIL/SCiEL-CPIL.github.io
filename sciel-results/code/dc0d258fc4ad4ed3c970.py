import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import random

N = 11

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_indices = np.array(tri_indices)
I = tri_indices[:, 0]
J = tri_indices[:, 1]
K = tri_indices[:, 2]
NUM_TRIS = len(tri_indices)

def _get_points(x):
    """
    Map 22-dim vector x in [0,1]^22 to 11 points in the unit equilateral triangle.
    x is flattened: [u0, v0, u1, v1, ..., u10, v10]
    """
    x = np.reshape(x, (N, 2))
    u = x[:, 0]
    v = x[:, 1] * (1 - u)
    w = 1 - u - v
    
    # Barycentric to Cartesian
    # A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # P = u*A + v*B + w*C
    px = 0.5 * w + 1.0 * v
    py = (np.sqrt(3) / 2) * w
    
    return np.column_stack((px, py))

def _min_triangle_area(points):
    """
    Calculate the minimum area of triangles formed by the points.
    points: (N, 2) array
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    # Cross product in 2D
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def objective(x):
    """
    Objective function for minimization.
    Returns negative of the minimum triangle area.
    """
    points = _get_points(x)
    min_area = _min_triangle_area(points)
    return -min_area

def _structured_init(seed):
    """
    Generate a structured initial placement using a grid-like pattern
    with some jitter to break symmetry.
    """
    rng = np.random.RandomState(seed)
    
    # Use a triangular grid pattern
    # Place points in rows within the triangle
    points = []
    
    # Row 0: 1 point at top
    # Row 1: 2 points
    # Row 2: 3 points
    # Row 3: 5 points (remaining)
    
    # Actually, let's use a more systematic approach:
    # Place points on a hexagonal lattice clipped to the triangle
    
    # Simpler: use concentric triangular rings
    # Center + ring of 6 + ring of 9 = 16, too many
    # Center + ring of 6 + 4 more = 11
    
    # Let's try: 3x3 grid in barycentric coordinates
    # u, v from {0, 0.5, 1} with u+v <= 1
    # (0,0), (0.5,0), (1,0)
    # (0,0.5), (0.5,0.5)
    # (0,1)
    # That's 6 points. Need 5 more.
    
    # Alternative: use a pattern based on known optimal packings
    # For 11 points, try a modified triangular lattice
    
    # Let's use a simple approach: place on a grid in barycentric coords
    # with random perturbation
    
    # Generate 11 points using stratified sampling in barycentric coords
    # Divide triangle into 11 regions and pick center of each
    
    # Simpler: use a hexagonal close packing pattern
    # Row 0 (top): 1 point at (0.5, sqrt(3)/6) in Cartesian -> barycentric (1/3, 1/3, 1/3)
    # Actually, let's just use random with some structure
    
    # Use a grid in u-v space
    # u from 0 to 1, v from 0 to 1-u
    # Create a 4x3 grid and pick 11 points
    
    # Better: use the fact that optimal configs often have points on a triangular lattice
    # Let's generate points on a triangular lattice and select 11
    
    # Triangular lattice points in barycentric coordinates
    # Points where u, v, w are multiples of 1/3
    # (u, v) with u+v <= 1, u,v >= 0
    
    lattice_points = []
    for i in range(4):  # u = i/3
        for j in range(4 - i):  # v = j/3, u+v <= 1
            u = i / 3.0
            v = j / 3.0
            lattice_points.append((u, v))
    
    # We have 10 points from the 3x3 lattice (including boundary)
    # Need 1 more, so add a perturbed version or use a finer grid
    
    # Let's use a different approach: place points at vertices, edge midpoints, and center
    # Vertices: (0,0), (1,0), (0,1) in barycentric -> 3 points
    # Edge midpoints: (0.5,0), (0,0.5), (0.5,0.5) -> 3 points
    # Center: (1/3, 1/3) -> 1 point
    # That's 7 points. Need 4 more.
    
    # Add points at 1/4 and 3/4 positions on edges
    # (0.25, 0), (0.75, 0), (0, 0.25), (0, 0.75) -> 4 points
    # Total: 11 points
    
    base_points = [
        (0.0, 0.0),   # Vertex A
        (1.0, 0.0),   # Vertex B
        (0.0, 1.0),   # Vertex C
        (0.5, 0.0),   # Mid AB
        (0.0, 0.5),   # Mid AC
        (0.5, 0.5),   # Mid BC
        (1/3, 1/3),   # Center
        (0.25, 0.0),  # 1/4 AB
        (0.75, 0.0),  # 3/4 AB
        (0.0, 0.25),  # 1/4 AC
        (0.0, 0.75),  # 3/4 AC
    ]
    
    # Add small random perturbation to break symmetry
    x = np.array(base_points, dtype=float).flatten()
    jitter = rng.normal(0, 0.02, size=2*N)
    x += jitter
    
    # Clip to valid range
    for i in range(N):
        x[2*i] = np.clip(x[2*i], 0, 1)
        x[2*i+1] = np.clip(x[2*i+1], 0, 1 - x[2*i])
    
    return x

def _simulate_annealing(x0, max_iter=5000, T0=0.05, T_end=0.001, alpha=0.999):
    """
    Simple simulated annealing to escape local optima.
    """
    rng = np.random.RandomState(42)
    x = x0.copy()
    best_x = x.copy()
    best_val = objective(x)
    current_val = best_val
    
    T = T0
    for i in range(max_iter):
        # Generate a perturbation
        x_new = x.copy()
        # Randomly perturb 1-3 coordinates
        n_perturb = rng.randint(1, 4)
        indices = rng.choice(2*N, size=n_perturb, replace=False)
        for idx in indices:
            x_new[idx] += rng.normal(0, 0.05)
        
        # Clip to valid range
        for i_pt in range(N):
            x_new[2*i_pt] = np.clip(x_new[2*i_pt], 0, 1)
            x_new[2*i_pt+1] = np.clip(x_new[2*i_pt+1], 0, 1 - x_new[2*i_pt])
        
        new_val = objective(x_new)
        
        # Accept or reject
        if new_val < current_val:
            x = x_new
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = x_new.copy()
        else:
            # Accept with probability exp((new_val - current_val) / T)
            prob = np.exp((new_val - current_val) / T)
            if rng.random() < prob:
                x = x_new
                current_val = new_val
        
        T *= alpha
    
    return best_x, best_val

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s limit
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_x = None
    best_val = -np.inf
    
    # Strategy 1: Structured initialization + L-BFGS-B from multiple starts
    # This should find good local optima quickly
    
    # Generate multiple structured initial points with different jitter seeds
    init_seeds = [42, 123, 999, 7, 1234]
    
    for seed in init_seeds:
        if time.time() - start_time > timeout - 30.0:
            break
        
        x0 = _structured_init(seed)
        
        # Run L-BFGS-B
        try:
            res = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Simulated annealing from best found so far
    if best_x is not None and time.time() - start_time < timeout - 20.0:
        try:
            sa_x, sa_val = _simulate_annealing(best_x, max_iter=3000)
            if sa_val < best_val:
                best_val = sa_val
                best_x = sa_x.copy()
        except Exception:
            pass
    
    # Strategy 3: Differential Evolution as a broader search
    if time.time() - start_time < timeout - 15.0:
        try:
            res = differential_evolution(
                objective,
                bounds,
                maxiter=2000,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=False,
                init='latinhypercube',
                update_interval=1
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final polish with Nelder-Mead
    if best_x is not None and time.time() - start_time < timeout - 5.0:
        try:
            res_polish = minimize(
                objective,
                best_x,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'maxfev': 1000}
            )
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_x = res_polish.x.copy()
        except Exception:
            pass
    
    # Fallback if all failed
    if best_x is None:
        x = np.random.rand(2 * N)
        best_x = x
        best_val = objective(x)
    
    # Convert best_x to points
    points = _get_points(best_x)
    
    return points
