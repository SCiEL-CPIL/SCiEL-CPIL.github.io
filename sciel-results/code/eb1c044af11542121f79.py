"""Combined program for Heilbronn-11: Barycentric optimization with soft-min annealing."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

# EVOLVE-BLOCK-START
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of equilateral triangle with side 1

# Safety margin from 120s limit. 
# The evaluator kills after 120s. We aim to return well before.
DEADLINE = 105.0 

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)  # 165

# Triangle vertices in Cartesian
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3H]])

def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    return bary[:, 0:1] * _VERTS[0] + bary[:, 1:2] * _VERTS[1] + bary[:, 2:3] * _VERTS[2]

def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric coordinates (N, 3)."""
    x = cart[:, 0]
    y = cart[:, 1]
    b2 = y / S3
    b1 = x - 0.5 * b2
    b0 = 1 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _compute_areas_cart(points):
    """Calculate all triangle areas for the points in Cartesian."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _vectorized_min_area_cart(points):
    """Calculate the minimum area of all triangles formed by the points in Cartesian."""
    areas = _compute_areas_cart(points)
    return np.min(areas)

def _soft_min_area_cart(points, alpha=100):
    """Calculate soft minimum of triangle areas using log-sum-exp."""
    areas = _compute_areas_cart(points)
    min_a = np.min(areas)
    return min_a - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - min_a))))

def _project_bary(bary):
    """Project barycentric coordinates to be valid (non-negative, sum to 1)."""
    bary = np.maximum(0, bary)
    row_sums = bary.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    bary /= row_sums
    return bary

def _project_cartesian(points):
    """Project points to be inside the triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    # Clip to triangle boundaries with multiple passes
    for _ in range(3):
        y = np.maximum(0, y)
        y = np.minimum(y, S3 * x)
        y = np.minimum(y, S3 - S3 * x)
        x = np.maximum(0, x)
        x = np.minimum(1, x)
    
    return np.column_stack([x, y])

def _is_valid_cart(points):
    """Check if all points are inside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    tol = 1e-9
    return (np.all(y >= -tol) and 
            np.all(y <= S3 * x + tol) and 
            np.all(S3 * x + y <= S3 + tol))

def _random_bary():
    """Generate random barycentric coordinates."""
    r = np.random.random((N, 3))
    r /= r.sum(axis=1, keepdims=True)
    return r

def _bary_to_valid_cart(flat_bary):
    """Convert flat barycentric to valid Cartesian with projection."""
    bary = flat_bary.reshape(N, 3)
    bary = _project_bary(bary)
    cart = _bary_to_cart(bary)
    cart = _project_cartesian(cart)
    return cart

def _bary_objective(flat_bary, alpha=100):
    """Objective function in barycentric coordinates (negative soft-min area)."""
    cart = _bary_to_valid_cart(flat_bary)
    soft_min = _soft_min_area_cart(cart, alpha)
    return -soft_min

def _cart_objective(flat_cart):
    """Objective function in Cartesian coordinates (negative min area)."""
    cart = flat_cart.reshape(N, 2)
    cart = _project_cartesian(cart)
    area = _vectorized_min_area_cart(cart)
    return -area

def _coordinate_descent_polish(cart, max_iters=200, tol=1e-10):
    """Polish using coordinate descent on each point's x and y."""
    cart = cart.copy()
    best_score = _vectorized_min_area_cart(cart)
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            # Try small perturbations in x
            for dx in [1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4, 1e-3, -1e-3]:
                new_cart = cart.copy()
                new_cart[i, 0] += dx
                new_cart = _project_cartesian(new_cart)
                new_score = _vectorized_min_area_cart(new_cart)
                if new_score > best_score + tol:
                    best_score = new_score
                    cart = new_cart
                    improved = True
                    break
            
            # Try small perturbations in y
            for dy in [1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4, 1e-3, -1e-3]:
                new_cart = cart.copy()
                new_cart[i, 1] += dy
                new_cart = _project_cartesian(new_cart)
                new_score = _vectorized_min_area_cart(new_cart)
                if new_score > best_score + tol:
                    best_score = new_score
                    cart = new_cart
                    improved = True
                    break
        
        if not improved:
            break
    
    return cart, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    np.random.seed(42)
    
    # Initial best
    best_bary = _random_bary()
    best_cart = _bary_to_cart(best_bary)
    best_cart = _project_cartesian(best_cart)
    best_score = _vectorized_min_area_cart(best_cart)
    
    # Annealing schedule with more levels
    alphas = [5, 10, 25, 50, 100, 200, 500, 1000]
    
    for alpha in alphas:
        if time.time() > deadline - 15:  # Leave time for final polish
            break
        
        # Multiple restarts at each alpha level
        num_restarts = 15
        for r in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            # Start from current best or random
            if r == 0:
                x0 = best_bary.flatten()
            elif r % 3 == 0:
                # Perturbed best
                x0 = best_bary.flatten() + np.random.normal(0, 0.05, size=N*3)
            else:
                x0 = _random_bary().flatten()
            
            # Optimize with L-BFGS-B (faster than Nelder-Mead in high dim)
            res = minimize(_bary_objective, x0, method='L-BFGS-B',
                          args=(alpha,),
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000})
            
            # Evaluate the result
            new_cart = _bary_to_valid_cart(res.x)
            new_score = _vectorized_min_area_cart(new_cart)
            
            if new_score > best_score:
                best_score = new_score
                best_cart = new_cart
                best_bary = _cart_to_bary(new_cart)
    
    # Final refinement with hard minimum using Nelder-Mead in Cartesian
    if time.time() < deadline - 5:
        x0 = best_cart.flatten()
        
        res = minimize(_cart_objective, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.fun < -best_score:
            final_cart = res.x.reshape(N, 2)
            final_cart = _project_cartesian(final_cart)
            final_score = _vectorized_min_area_cart(final_cart)
            if final_score > best_score:
                best_cart = final_cart
                best_score = final_score
    
    # Coordinate descent polish
    if time.time() < deadline - 2:
        polished_cart, polished_score = _coordinate_descent_polish(best_cart, max_iters=100)
        if polished_score > best_score:
            best_cart = polished_cart
            best_score = polished_score
    
    return best_cart
# EVOLVE-BLOCK-END
