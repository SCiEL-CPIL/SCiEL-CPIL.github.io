"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples (C(11,3) = 165)
# Storing as a single array for vectorized operations
_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_I_IDX = _TRIPLES[:, 0]
_J_IDX = _TRIPLES[:, 1]
_K_IDX = _TRIPLES[:, 2]

def _bary_to_cart(bary):
    """Convert barycentric coordinates (lambda1, lambda2) to Cartesian coordinates.
    lambda3 is implicit: 1 - lambda1 - lambda2.
    Point = lambda1 * V1 + lambda2 * V2 + lambda3 * V0
    Since V0 is (0,0), Point = lambda1 * V1 + lambda2 * V2
    """
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _clamp_bary(bary):
    """Ensure barycentric coordinates are valid (non-negative, sum <= 1) with a safety margin."""
    bary = np.array(bary, copy=True)
    # Clip to non-negative
    bary[bary < 0] = 0.0
    
    # Calculate sum of first two barycentric coordinates
    s = bary[:, 0] + bary[:, 1]
    
    # If sum > 1, scale down to ensure sum <= 1 - epsilon
    # Using 1.0 - 1e-9 to stay strictly inside
    mask = s > (1.0 - 1e-9)
    if np.any(mask):
        # Scale factor to bring sum to exactly 1.0 - 1e-9
        scale = (1.0 - 1e-9) / s[mask]
        bary[mask] = bary[mask] * scale[:, None]
        
    return bary

def _all_triangle_areas(points):
    """Compute all triangle areas formed by triples of points."""
    pi = points[_I_IDX]
    pj = points[_J_IDX]
    pk = points[_K_IDX]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _min_triangle_area(points):
    """Compute the minimum triangle area."""
    return np.min(_all_triangle_areas(points))

def _softmin_objective(x, beta=100.0):
    """
    Objective function using soft-min (log-sum-exp) approximation of the minimum area.
    x is a flat array of barycentric coordinates (size 2*N).
    """
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    areas = _all_triangle_areas(points)
    
    # Soft min: min(a) ~ min(a) - (1/beta) * log(sum(exp(-beta * (a - min(a)))))
    # This is smooth and differentiable.
    min_area = np.min(areas)
    # Numerical stability: shift by min_area
    shifted = areas - min_area
    # Avoid overflow in exp
    # exp(-beta * shifted) is safe if shifted >= 0
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_area - log_sum / beta
    return -soft_min

def _hard_objective(x):
    """Objective function using the true minimum area."""
    bary = x.reshape(N, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area(points)

def _generate_structural_initialization(mode, rng):
    """Generate diverse initial configurations based on symmetry and heuristics."""
    if mode == 0:
        # Symmetric: Vertices, Midpoints, Centroid, + 4 others
        bary = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [1/3, 1/3],
            [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.5, 0.1]
        ])
    elif mode == 1:
        # Hexagonal-ish packing
        bary = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [0.33, 0.33],
            [0.15, 0.15], [0.65, 0.15], [0.15, 0.65], [0.5, 0.2]
        ])
    elif mode == 2:
        # Random spread
        bary = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.uniform(0.0, 1.0)
                v = rng.uniform(0.0, 1.0)
                if u + v < 1.0:
                    bary[i] = [u, v]
                    break
    else:
        # Pure random
        bary = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.uniform(0.0, 1.0)
                v = rng.uniform(0.0, 1.0)
                if u + v < 1.0:
                    bary[i] = [u, v]
                    break
    
    # Add small jitter to break symmetry if needed
    jitter = rng.normal(0, 0.02, size=bary.shape)
    bary = bary + jitter
    bary = _clamp_bary(bary)
    return bary

def _refine_with_annealing(bary0, time_limit=10.0):
    """
    Refine a configuration using beta annealing with L-BFGS-B.
    Starts with smooth objective (low beta) and increases sharpness.
    """
    start_time = time.time()
    bary = bary0.copy()
    
    # Stages: (beta, maxiter)
    # Low beta = smooth, easier to navigate global landscape
    # High beta = sharp, approaches true min
    stages = [
        (20, 50),
        (50, 100),
        (100, 200),
        (200, 300)
    ]
    
    bounds = [(0, 1)] * (2 * N)
    
    for beta, maxiter in stages:
        if time.time() - start_time > time_limit:
            break
            
        try:
            x0 = bary.flatten()
            res = minimize(
                _softmin_objective, x0, args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
            )
            bary = _clamp_bary(res.x.reshape(N, 2))
        except Exception:
            pass
            
    return bary

def _final_polish(bary0, time_limit=5.0):
    """
    Final polish using Nelder-Mead on the hard objective.
    """
    start_time = time.time()
    try:
        x0 = bary0.flatten()
        res = minimize(
            _hard_objective, x0,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt
    except Exception:
        return bary0

def _is_inside_triangle(points):
    """Check if all points are strictly inside or on the boundary of the reference triangle."""
    # Reference triangle: (0,0), (1,0), (0.5, sqrt(3)/2)
    # We can check barycentric coordinates directly if we keep them, 
    # but here we have Cartesian points. 
    # Simpler: convert back to barycentric and check? 
    # Or use cross products to check if points are on the same side of edges.
    
    # Let's just re-verify using the barycentric logic which is safer.
    # The function construct() returns Cartesian, but we have best_bary.
    # We will add a final safety check in construct().
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Tighter safety margin
    
    def time_left():
        return deadline - time.time()
        
    best_bary = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # 1. Global Search with Differential Evolution
    if time_left() > 40.0:
        try:
            bounds = [(0, 1)] * (2 * N)
            def de_obj(x):
                return _softmin_objective(x, beta=50.0)
            
            de_result = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=20, 
                popsize=10,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                updating='deferred'
            )
            bary_de = _clamp_bary(de_result.x.reshape(N, 2))
            
            # Refine DE result
            bary_refined = _refine_with_annealing(bary_de, time_limit=10.0)
            score = -_hard_objective(bary_refined.flatten())
            if score > best_score:
                best_score = score
                best_bary = bary_refined
        except Exception:
            pass

    # 2. Local Optimization from Structured Initializations
    initializations = []
    for mode in range(4):
        bary_init = _generate_structural_initialization(mode, rng)
        initializations.append(bary_init)
        
    for _ in range(3):
        bary_init = _generate_structural_initialization(3, rng)
        initializations.append(bary_init)

    for bary_init in initializations:
        if time_left() < 15.0:
            break
            
        bary_refined = _refine_with_annealing(bary_init, time_limit=8.0)
        
        if time_left() > 5.0:
            bary_polished = _final_polish(bary_refined, time_limit=4.0)
        else:
            bary_polished = bary_refined
            
        score = -_hard_objective(bary_polished.flatten())
        if score > best_score:
            best_score = score
            best_bary = bary_polished

    # 3. Random restarts
    while time_left() > 10.0:
        bary_init = _generate_structural_initialization(3, rng)
        bary_refined = _refine_with_annealing(bary_init, time_limit=5.0)
        if time_left() > 3.0:
            bary_polished = _final_polish(bary_refined, time_limit=2.0)
        else:
            bary_polished = bary_refined
            
        score = -_hard_objective(bary_polished.flatten())
        if score > best_score:
            best_score = score
            best_bary = bary_polished

    if best_bary is None:
        best_bary = np.zeros((N, 2))
        best_bary[0] = [0.0, 0.0]
        best_bary[1] = [1.0, 0.0]
        best_bary[2] = [0.0, 1.0]
        best_bary[3] = [0.5, 0.5]
        for i in range(4, N):
            best_bary[i] = [1/3, 1/3] + rng.normal(0, 0.1, 2)
        best_bary = _clamp_bary(best_bary)
        
    # Final Safety Check: Ensure points are strictly inside
    # Convert to Cartesian
    points = _bary_to_cart(best_bary)
    
    # Verify each point is inside the triangle (0,0)-(1,0)-(0.5, sqrt(3)/2)
    # A point (x,y) is inside if:
    # y >= 0
    # y <= sqrt(3) * x  (for x < 0.5)
    # y <= sqrt(3) * (1-x) (for x > 0.5)
    # And x between 0 and 1.
    
    valid = True
    for p in points:
        x, y = p
        if x < -1e-9 or x > 1 + 1e-9 or y < -1e-9:
            valid = False
            break
        # Check line 1: y <= sqrt(3) * x
        if y > np.sqrt(3) * x + 1e-9:
            valid = False
            break
        # Check line 2: y <= sqrt(3) * (1-x)
        if y > np.sqrt(3) * (1 - x) + 1e-9:
            valid = False
            break
            
    if not valid:
        # Fallback to a known valid configuration: Grid-like or random valid
        print("Invalid points detected, using fallback.")
        # Generate a valid random configuration
        fallback_bary = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.uniform(0.0, 1.0)
                v = rng.uniform(0.0, 1.0)
                if u + v <= 1.0:
                    fallback_bary[i] = [u, v]
                    break
        # Scale slightly inwards to be safe
        fallback_bary = fallback_bary * 0.95
        fallback_bary = _clamp_bary(fallback_bary)
        points = _bary_to_cart(fallback_bary)

    return np.array(points)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
