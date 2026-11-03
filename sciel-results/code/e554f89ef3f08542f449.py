"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
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
    # Numerical stability: shift by min_a
    exp_terms = np.exp(-alpha * (areas - min_a))
    return min_a - (1.0 / alpha) * np.log(np.sum(exp_terms))

def _project_bary(bary):
    """Project barycentric coordinates to the valid simplex (non-negative, sum 1)."""
    bary = np.maximum(0, bary)
    sums = bary.sum(axis=1, keepdims=True)
    # Avoid division by zero
    sums[sums == 0] = 1
    bary /= sums
    return bary

def _bary_objective(flat_bary, alpha=100):
    """Objective function in barycentric coordinates."""
    bary = flat_bary.reshape(N, 3)
    bary = _project_bary(bary)
    
    cart = _bary_to_cart(bary)
    
    # Soft minimum area
    soft_min = _soft_min_area_cart(cart, alpha)
    
    return -soft_min

def _structured_init():
    """Generate a structured initial configuration based on a hexagonal/grid pattern."""
    # Try a 3x3 grid + 2 extra points, or a distorted hex grid
    # A good heuristic for N=11 is to start with a regular grid and let it relax
    # Grid in barycentric coordinates
    # Rows 0, 1, 2, 3
    # Row 0: 3 points
    # Row 1: 3 points
    # Row 2: 3 points
    # Row 3: 2 points
    # Total 11
    
    bary = np.zeros((N, 3))
    # Define grid lines in barycentric space
    # y-coordinate in barycentric corresponds to distance from base
    # Let's use a simple mapping
    # b2 is "height"
    # b1 is "right-ness"
    
    # 4 rows
    heights = [0.1, 0.3, 0.5, 0.7] # b2 values
    # For each height, distribute points along the width
    
    # Row 0 (bottom): 3 points
    bary[0] = [0.8, 0.1, 0.1]
    bary[1] = [0.5, 0.4, 0.1]
    bary[2] = [0.2, 0.7, 0.1]
    
    # Row 1: 3 points
    bary[3] = [0.65, 0.25, 0.3]
    bary[4] = [0.4, 0.5, 0.3]
    bary[5] = [0.15, 0.75, 0.3]
    
    # Row 2: 3 points
    bary[6] = [0.5, 0.2, 0.5]
    bary[7] = [0.3, 0.4, 0.5]
    bary[8] = [0.1, 0.6, 0.5]
    
    # Row 3 (top): 2 points
    bary[9] = [0.35, 0.15, 0.7]
    bary[10] = [0.15, 0.35, 0.7]
    
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    np.random.seed(42)
    
    # Initial best: Use structured init
    best_bary = _structured_init()
    best_cart = _bary_to_cart(best_bary)
    best_score = _vectorized_min_area_cart(best_cart)
    
    # Also try a few random starts to see if they are better
    for _ in range(5):
        r = np.random.random((N, 3))
        r /= r.sum(axis=1, keepdims=True)
        cart = _bary_to_cart(r)
        score = _vectorized_min_area_cart(cart)
        if score > best_score:
            best_score = score
            best_bary = r
            best_cart = cart

    # Annealing schedule
    # Start with low alpha (smooth) and increase to high alpha (sharp)
    alphas = [10, 50, 100, 200, 500, 1000, 2000]
    
    for alpha in alphas:
        if time.time() > deadline:
            break
        
        # Multiple restarts at each alpha level
        # Reduce restarts as alpha increases to spend time on refinement
        num_restarts = max(2, 10 - len([a for a in alphas if a <= alpha]))
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Start from current best or random
            if r == 0:
                x0 = best_bary.flatten()
            else:
                # Perturb the best or random
                if np.random.random() < 0.5:
                    x0 = (best_bary + np.random.normal(0, 0.05, (N, 3))).flatten()
                else:
                    x0 = _structured_init().flatten()
                    x0 += np.random.normal(0, 0.1, (N, 3))
            
            # Optimize with Nelder-Mead in barycentric space
            res = minimize(_bary_objective, x0, method='Nelder-Mead',
                          args=(alpha,),
                          options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            
            # Evaluate result
            new_bary = _project_bary(res.x.reshape(N, 3))
            new_cart = _bary_to_cart(new_bary)
            new_score = _vectorized_min_area_cart(new_cart)
            
            if new_score > best_score:
                best_score = new_score
                best_bary = new_bary
                best_cart = new_cart

    # Final refinement with high precision on the hard minimum (or very high alpha soft min)
    # Use the best configuration found
    if time.time() < deadline - 5:
        # Use a very high alpha soft min as a smooth approximation of the hard min
        alpha_final = 5000
        x0 = best_bary.flatten()
        
        res = minimize(_bary_objective, x0, method='Nelder-Mead',
                      args=(alpha_final,),
                      options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
        
        final_bary = _project_bary(res.x.reshape(N, 3))
        final_cart = _bary_to_cart(final_bary)
        final_score = _vectorized_min_area_cart(final_cart)
        
        if final_score > best_score:
            best_cart = final_cart
            best_score = final_score

    return best_cart
# EVOLVE-BLOCK-END
