"""Optimized Heilbronn triangle solver using barycentric coordinates and DE."""
import numpy as np
import time
from scipy.optimize import differential_evolution

N = 11

# EVOLVE-BLOCK-START

# Triangle vertices in Cartesian
_A = np.array([0.0, 0.0])
_B = np.array([1.0, 0.0])
_C = np.array([0.5, np.sqrt(3) / 2])

# Barycentric to Cartesian: point = a*A + b*B + c*C, with a+b+c=1, a,b,c>=0
# We parameterize with (b, c) so a = 1 - b - c >= 0
# x = b + 0.5*c
# y = (sqrt(3)/2)*c


def _bary_to_cart(bary):
    """Convert array of (b, c) barycentric pairs to (x, y) Cartesian.
    bary shape: (n, 2) -> returns (n, 2)
    """
    b = bary[:, 0]
    c = bary[:, 1]
    x = b + 0.5 * c
    y = (np.sqrt(3) / 2) * c
    return np.column_stack([x, y])


def _cart_to_bary(cart):
    """Convert (x, y) Cartesian to (b, c) barycentric."""
    x = cart[:, 0]
    y = cart[:, 1]
    c = y / (np.sqrt(3) / 2)
    b = x - 0.5 * c
    return np.column_stack([b, c])


def _min_triangle_area_bary(bary):
    """Compute minimum triangle area from barycentric coordinates.
    bary: (n, 2) array of (b, c) pairs.
    Returns the minimum area of any triangle formed by 3 points.
    """
    n = len(bary)
    pts = _bary_to_cart(bary)
    # Vectorized: compute all triangle areas
    # For each triplet (i, j, k), area = 0.5 * |cross(pj-pi, pk-pi)|
    min_area = np.inf
    for i in range(n):
        pi = pts[i]
        # All vectors from pi to other points
        vecs = pts[i + 1:] - pi
        # For each pair in vecs, compute cross product
        # vecs has shape (m, 2) where m = n - i - 1
        m = len(vecs)
        if m < 2:
            continue
        # Cross products of all pairs
        # vecs[a] x vecs[b] = vecs[a,0]*vecs[b,1] - vecs[a,1]*vecs[b,0]
        v0 = vecs[:, 0]
        v1 = vecs[:, 1]
        # All pairwise cross products
        cross = v0[:, None] * v1[None, :] - v1[:, None] * v0[None, :]
        # Only upper triangle (a < b)
        cross = np.triu(cross, 1)
        areas = 0.5 * np.abs(cross)
        min_in_i = np.min(areas)
        if min_in_i < min_area:
            min_area = min_in_i
    return min_area


def _objective(bary_flat):
    """Objective function for minimization: negative of min area."""
    bary = bary_flat.reshape(-1, 2)
    return -_min_triangle_area_bary(bary)


def construct():
    t_start = time.time()
    deadline = t_start + 130.0  # Safety margin before 150s kill
    
    best_bary = None
    best_area = 0.0
    
    # Strategy: Multiple restarts with differential evolution
    rng_seed = 42
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Random initial population in barycentric space
        # b, c in [0, 1] with b + c <= 1
        # We'll use bounds [0,1] for each and clip/penalize later
        # Better: use a transformation that ensures b+c<=1
        # Let b = u * (1-v), c = v * (1-u) ... no, simpler:
        # Sample u, v in [0,1], if u+v < 1, use (u,v), else reject
        # But for DE, we need a box. Use bounds [0, 0.99] for b and c,
        # and in the objective, clip to valid region.
        
        # Actually, let's use a different parameterization:
        # Let s in [0,1], t in [0, 1-s]
        # b = s, c = t
        # To make it a box: use u in [0,1], v in [0,1]
        # b = u * (1 - v), c = v * (1 - u)
        # This maps unit square to triangle interior
        
        # For DE, we'll work in (u, v) space with bounds [0,1] for each
        # and convert to (b, c) internally
        
        def _uv_to_bary(uv):
            u = uv[:, 0]
            v = uv[:, 1]
            b = u * (1 - v)
            c = v * (1 - u)
            return np.column_stack([b, c])
        
        def _uv_objective(uv_flat):
            uv = uv_flat.reshape(-1, 2)
            bary = _uv_to_bary(uv)
            return -_min_triangle_area_bary(bary)
        
        # Differential evolution in (u, v) space
        bounds = [(0.0, 1.0)] * (2 * N)
        
        # Use a moderate number of iterations
        # Check time for this restart
        est_time_per_restart = 4.0
        max_iter = max(50, int((deadline - time.time()) / est_time_per_restart))
        max_iter = min(max_iter, 300)
        
        try:
            result = differential_evolution(
                _uv_objective,
                bounds,
                maxiter=max_iter,
                popsize=15,
                tol=1e-6,
                seed=rng_seed + restart,
                mutation=(0.3, 1.0),
                recombination=0.8,
                init='latinhypercube',
                polish=True,
                polish_options={'xatol': 1e-8},
            )
            
            if result.success or result.fun < 0:
                uv_best = result.x.reshape(-1, 2)
                bary_best = _uv_to_bary(uv_best)
                area = -result.fun
                if area > best_area:
                    best_area = area
                    best_bary = bary_best.copy()
        except Exception:
            continue
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # If DE didn't find anything good, use a simple grid/random fallback
    if best_bary is None or best_area <= 0:
        # Fallback: random points
        np.random.seed(7)
        bary = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = np.random.random()
                v = np.random.random()
                if u + v < 0.98:
                    bary[i] = [u, v]
                    break
        best_bary = bary
        best_area = _min_triangle_area_bary(bary)
    
    # Convert to Cartesian and return
    cart = _bary_to_cart(best_bary)
    
    # Verify all points are inside the triangle
    # For equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)
    # Point (x,y) is inside if:
    # y >= 0
    # y <= sqrt(3)/2
    # y <= sqrt(3)*x  (left edge from (0,0) to (0.5, sqrt(3)/2))
    # y <= -sqrt(3)*(x - 1)  (right edge from (1,0) to (0.5, sqrt(3)/2))
    
    # Clip to be safe
    eps = 1e-10
    for i in range(N):
        x, y = cart[i]
        # Ensure inside triangle
        if y < eps:
            y = eps
        if y > np.sqrt(3) / 2 - eps:
            y = np.sqrt(3) / 2 - eps
        # Left edge: y <= sqrt(3) * x
        if y > np.sqrt(3) * x + eps:
            # Project to left edge
            x = y / np.sqrt(3) + eps
        # Right edge: y <= -sqrt(3) * (x - 1) = sqrt(3) * (1 - x)
        if y > np.sqrt(3) * (1 - x) + eps:
            # Project to right edge
            x = 1 - y / np.sqrt(3) - eps
        # Final bounds
        x = np.clip(x, eps, 1 - eps)
        y = np.clip(y, eps, np.sqrt(3) / 2 - eps)
        cart[i] = [x, y]
    
    return cart

# EVOLVE-BLOCK-END
