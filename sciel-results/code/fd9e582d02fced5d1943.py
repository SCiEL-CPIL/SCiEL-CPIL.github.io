"""Combined program for Heilbronn-11: Barycentric optimization with soft-min annealing and Delaunay-based active set refinement."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of equilateral triangle with side 1

# Safety margin from 150s limit
DEADLINE_MARGIN = 140.0

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)  # 165

# Triangle vertices in Cartesian
_VERTS = np.array([[0, 0], [1, 0], [0.5, S3H]])

def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    return bary[:, 0:1] * _VERTS[0] + bary[:, 1:2] * _VERTS[1] + bary[:, 2:3] * _VERTS[2]

def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric coordinates (N, 3)."""
    x = cart[:, 0]
    y = cart[:, 1]
    # b2 = y / S3
    b2 = y / S3
    # b1 = x - 0.5*b2
    b1 = x - 0.5 * b2
    # b0 = 1 - b1 - b2
    b0 = 1 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _vectorized_min_area_cart(points):
    """Calculate the minimum area of all triangles formed by the points in Cartesian."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _soft_min_area_cart(points, alpha=100):
    """Calculate soft minimum of triangle areas using log-sum-exp."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    # Soft min: -1/alpha * log(sum(exp(-alpha * areas)))
    min_a = np.min(areas)
    return min_a - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - min_a))))

def _project_cartesian(points):
    """Project points to be inside the triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    # Clip to triangle boundaries
    # y >= 0
    y = np.maximum(0, y)
    # y <= sqrt(3) * x
    y = np.minimum(y, S3 * x)
    # sqrt(3) * x + y <= sqrt(3)
    y = np.minimum(y, S3 - S3 * x)
    
    # Re-clip x after y adjustments
    x = np.maximum(0, x)
    x = np.minimum(1, x)
    # y >= 0
    y = np.maximum(0, y)
    # y <= sqrt(3) * x
    y = np.minimum(y, S3 * x)
    # sqrt(3) * x + y <= sqrt(3)
    y = np.minimum(y, S3 - S3 * x)
    
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

def _bary_objective(flat_bary, alpha=100):
    """Objective function in barycentric coordinates."""
    bary = flat_bary.reshape(N, 3)
    # Ensure valid barycentric (non-negative, sum to 1)
    bary = np.maximum(0, bary)
    bary /= bary.sum(axis=1, keepdims=True)
    
    cart = _bary_to_cart(bary)
    cart = _project_cartesian(cart)
    
    # Convert back to barycentric for consistency
    bary_new = _cart_to_bary(cart)
    
    # Soft minimum area
    soft_min = _soft_min_area_cart(cart, alpha)
    
    # Penalty for points outside triangle (should be small after projection)
    x = cart[:, 0]
    y = cart[:, 1]
    pen1 = np.sum(np.maximum(0, -y)**2)
    pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
    pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
    pen = pen1 + pen2 + pen3
    
    return -soft_min + 100 * pen

def _get_delaunay_triples(points):
    """Get the indices of triangles in the Delaunay triangulation."""
    try:
        tri = Delaunay(points)
        return tri.simplices
    except Exception:
        return np.array([]).reshape(0, 3)

def _delaunay_min_area(points):
    """Calculate the minimum area among Delaunay triangles only."""
    simplices = _get_delaunay_triples(points)
    if len(simplices) == 0:
        return _vectorized_min_area_cart(points)
    
    areas = []
    for tri in simplices:
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        area = 0.5 * abs(cross)
        areas.append(area)
    
    if len(areas) == 0:
        return _vectorized_min_area_cart(points)
    
    return min(areas)

def _delaunay_active_set_objective(flat_cart, alpha=50):
    """
    Objective function that focuses on Delaunay triangles.
    Uses soft-min over Delaunay triangles plus a penalty for non-Delaunay
    triangles that are smaller than the current Delaunay minimum.
    """
    cart = flat_cart.reshape(N, 2)
    cart = _project_cartesian(cart)
    
    # Get Delaunay triangles
    simplices = _get_delaunay_triples(cart)
    
    if len(simplices) == 0:
        # Fallback to full objective
        return _bary_objective(_cart_to_bary(cart).flatten(), alpha)
    
    # Calculate areas for Delaunay triangles
    del_areas = []
    for tri in simplices:
        p1 = cart[tri[0]]
        p2 = cart[tri[1]]
        p3 = cart[tri[2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        area = 0.5 * abs(cross)
        del_areas.append(area)
    
    del_areas = np.array(del_areas)
    
    # Soft min over Delaunay triangles
    min_del = np.min(del_areas)
    soft_min_del = min_del - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (del_areas - min_del))))
    
    # Also compute full min area to check if Delaunay assumption holds
    full_min = _vectorized_min_area_cart(cart)
    
    # If full min is significantly less than Delaunay min, add penalty
    # This means the minimum triangle is not a Delaunay triangle (should be rare)
    gap_penalty = 0.0
    if full_min < min_del * 0.9:
        gap_penalty = 10 * (min_del - full_min)
    
    # Penalty for points outside triangle
    x = cart[:, 0]
    y = cart[:, 1]
    pen1 = np.sum(np.maximum(0, -y)**2)
    pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
    pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
    pen = pen1 + pen2 + pen3
    
    return -soft_min_del + gap_penalty + 100 * pen

def _delaunay_refine(points, max_iter=50, time_limit=5.0):
    """
    Refine points using Delaunay-based active set optimization.
    Iteratively recompute Delaunay triangulation and optimize.
    """
    start = time.time()
    cart = points.copy()
    best_score = _vectorized_min_area_cart(cart)
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Get current Delaunay triangles
        simplices = _get_delaunay_triples(cart)
        if len(simplices) == 0:
            break
        
        # Optimize using Delaunay-focused objective
        x0 = cart.flatten()
        
        res = minimize(_delaunay_active_set_objective, x0, method='Nelder-Mead',
                      args=(50,),
                      options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < _delaunay_active_set_objective(x0, 50):
            new_cart = res.x.reshape(N, 2)
            new_cart = _project_cartesian(new_cart)
            
            new_score = _vectorized_min_area_cart(new_cart)
            if new_score > best_score:
                best_score = new_score
                cart = new_cart
            else:
                # No improvement, break
                break
    
    return cart, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    np.random.seed(42)
    
    # Initial best
    best_bary = _random_bary()
    best_cart = _bary_to_cart(best_bary)
    best_cart = _project_cartesian(best_cart)
    best_score = _vectorized_min_area_cart(best_cart)
    
    # Annealing schedule
    alphas = [10, 50, 100, 200, 500, 1000]
    
    for alpha in alphas:
        if time.time() > deadline:
            break
        
        # Multiple restarts at each alpha level
        num_restarts = 10
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Start from current best or random
            if r == 0:
                x0 = best_bary.flatten()
            else:
                x0 = _random_bary().flatten()
            
            # Optimize with Nelder-Mead in barycentric space
            res = minimize(_bary_objective, x0, method='Nelder-Mead',
                          args=(alpha,),
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < _bary_objective(best_bary.flatten(), alpha):
                new_bary = res.x.reshape(N, 3)
                new_bary = np.maximum(0, new_bary)
                new_bary /= new_bary.sum(axis=1, keepdims=True)
                
                new_cart = _bary_to_cart(new_bary)
                new_cart = _project_cartesian(new_cart)
                
                new_score = _vectorized_min_area_cart(new_cart)
                if new_score > best_score:
                    best_score = new_score
                    best_bary = new_bary
                    best_cart = new_cart
    
    # Delaunay-based active set refinement
    if time.time() < deadline - 20:
        refined_cart, refined_score = _delaunay_refine(best_cart, max_iter=30, time_limit=15.0)
        if refined_score > best_score:
            best_score = refined_score
            best_cart = refined_cart
    
    # Final refinement with hard minimum
    if time.time() < deadline - 10:
        x0 = best_cart.flatten()
        
        def cart_objective(flat_cart):
            cart = flat_cart.reshape(N, 2)
            cart = _project_cartesian(cart)
            area = _vectorized_min_area_cart(cart)
            return -area
        
        res = minimize(cart_objective, x0, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7})
        
        if res.fun < -best_score:
            final_cart = res.x.reshape(N, 2)
            final_cart = _project_cartesian(final_cart)
            final_score = _vectorized_min_area_cart(final_cart)
            if final_score > best_score:
                best_cart = final_cart
                best_score = final_score
    
    return best_cart
# EVOLVE-BLOCK-END
