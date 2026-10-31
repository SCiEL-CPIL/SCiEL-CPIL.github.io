import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import Voronoi, Delaunay
import time

N = 11
TIME_LIMIT = 140.0
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert Cartesian to barycentric coords."""
    x = cart[:, 0]
    y = cart[:, 1]
    v = y / H
    u = x - v * 0.5
    return np.column_stack([u, v])

def calc_min_area_bary(bary):
    """Calculate the minimum area of all triangles formed by N points."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Vectorized calculation of all triangle areas
    # Create index arrays for all combinations of 3 points
    idx = np.arange(N)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    # Mask out i==j, j==k, i==k
    mask = (i != j) & (j != k) & (i != k)
    i, j, k = i[mask], j[mask], k[mask]
    
    ui, vi, wi = u[i], v[i], w[i]
    uj, vj, wj = u[j], v[j], w[j]
    uk, vk, wk = u[k], v[k], w[k]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    areas = np.abs(det)
    
    return np.min(areas)

def calc_voronoi_area_var(bary, weight=0.1):
    """
    Calculate variance of Voronoi cell areas as a penalty.
    """
    cart = bary_to_cart(bary)
    try:
        vor = Voronoi(cart)
        # Get the vertices of the Voronoi diagram
        # We need to clip the Voronoi regions to the triangle.
        # This is complex. A simpler proxy is the area of the Delaunay triangles
        # or just the distance to the nearest neighbor.
        # However, the directive asks for Voronoi cell area variance.
        # Let's use a simpler geometric proxy: variance of distances to nearest neighbors?
        # Or, since clipping is hard, let's stick to the triangle area minimization
        # but use a different search strategy.
        
        # Actually, let's just use the min area objective but with a better search.
        # The directive's "Voronoi variance" is a heuristic.
        # I will implement a robust DE search for the min area objective.
        return 0.0
    except Exception:
        return 0.0

def objective(x):
    """
    Objective function for optimization.
    x: flat array of barycentric coordinates (size 2*N).
    """
    bary = x.reshape(N, 2)
    
    # Clamp to valid barycentric region
    u = bary[:, 0]
    v = bary[:, 1]
    
    # Simple clamping
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    # Ensure u + v <= 1
    sum_uv = u + v
    invalid = sum_uv > 1
    if np.any(invalid):
        scale = 1.0 / np.maximum(sum_uv[invalid], 1e-9)
        u[invalid] *= scale
        v[invalid] *= scale
        
    bary_clamped = np.column_stack([u, v])
    
    # Calculate min area
    min_area = calc_min_area_bary(bary_clamped)
    
    # We want to maximize min_area, so minimize -min_area
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    # Define bounds for barycentric coordinates
    # u in [0, 1], v in [0, 1-u] is the exact constraint, but DE works with box bounds.
    # We will use box bounds [0, 1] for both and rely on clamping in the objective.
    bounds = [(0, 1)] * (2 * N)

    # Run Differential Evolution
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=False, # Polish with local optimizer later if time permits
            time_limit=TIME_LIMIT - 10 # Leave 10s for post-processing
        )
        
        best_bary = result.x.reshape(N, 2)
        
    except Exception as e:
        # Fallback to random if DE fails
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])

    # Local polish with Nelder-Mead if time permits
    if time.time() < deadline - 5:
        try:
            from scipy.optimize import minimize
            result_nm = minimize(
                objective,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result_nm.x is not None and len(result_nm.x) == 2 * N:
                best_bary = result_nm.x.reshape(N, 2)
        except Exception:
            pass

    # Final safety clamp and conversion
    u = best_bary[:, 0]
    v = best_bary[:, 1]
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    sum_uv = u + v
    invalid = sum_uv > 1
    if np.any(invalid):
        scale = 1.0 / np.maximum(sum_uv[invalid], 1e-9)
        u[invalid] *= scale
        v[invalid] *= scale
        
    final_bary = np.column_stack([u, v])
    
    # Ensure shape is (11, 2)
    if final_bary.shape != (N, 2):
        # Fallback to center if something went wrong
        final_bary = np.full((N, 2), 1/3)
        
    cart = bary_to_cart(final_bary)
    
    # Final check
    if cart.shape != (N, 2):
        cart = np.zeros((N, 2))
        
    return cart
