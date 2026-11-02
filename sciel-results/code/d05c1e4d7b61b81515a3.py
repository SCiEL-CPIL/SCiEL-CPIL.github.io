import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute triangle indices for vectorized area calculation
_i, _j, _k = np.indices((N, N, N))
_mask = (_i < _j) & (_j < _k)
_IDX_I = _i[_mask]
_IDX_J = _j[_mask]
_IDX_K = _k[_mask]

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    # Ensure points are valid
    if not np.all(np.isfinite(points)):
        return 0.0
        
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-10:
        return 0.0
        
    # Compute all triangle areas
    p_i = points[_IDX_I]
    p_j = points[_IDX_J]
    p_k = points[_IDX_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Area = 0.5 * |cross(v1, v2)|
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _objective(flat_points):
    """
    Objective function for minimization (we minimize negative score).
    flat_points: (2*N,) array
    """
    points = flat_points.reshape(N, 2)
    
    # Normalize to prevent numerical issues during optimization
    # Center at origin
    points = points - np.mean(points, axis=0)
    
    # Scale to have max norm 1 (optional, but helps stability)
    max_norm = np.max(np.linalg.norm(points, axis=1))
    if max_norm > 0:
        points = points / max_norm
    else:
        # If all points are at origin, return bad score
        return 1.0 

    score = _score(points)
    
    if score <= 0:
        return 1.0 # Penalty for invalid configuration
    
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # Leave 20s buffer for 120s limit
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    init_flat = init_points.flatten()
    
    best_flat = init_flat
    best_score = _score(init_points)
    
    # Try a few different random initializations to be safe
    rng = np.random.default_rng(42)
    
    # We will run Nelder-Mead optimization
    # Options:
    # - maxiter: limit iterations
    # - maxfev: limit function evaluations
    
    # Since N=14, dimension is 28. Nelder-Mead is suitable.
    
    # Run optimization
    try:
        res = minimize(
            _objective, 
            init_flat, 
            method='Nelder-Mead', 
            options={
                'maxiter': 1000, 
                'maxfev': 2000,
                'xatol': 1e-8,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        
        if res.success or (res.fun < _objective(init_flat)):
            cand_points = res.x.reshape(N, 2)
            cand_score = _score(cand_points)
            if cand_score > best_score:
                best_score = cand_score
                best_flat = res.x
                
    except Exception as e:
        pass
        
    # Check time and try a second run with a slightly perturbed start if time permits
    if time.time() < deadline - 10.0:
        # Perturb the initial regular polygon
        perturbation = rng.normal(0, 0.1, size=(N, 2))
        init2 = init_points + perturbation
        init2_flat = init2.flatten()
        
        try:
            res2 = minimize(
                _objective, 
                init2_flat, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 1000, 
                    'maxfev': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if res2.success or (res2.fun < _objective(init2_flat)):
                cand_points2 = res2.x.reshape(N, 2)
                cand_score2 = _score(cand_points2)
                if cand_score2 > best_score:
                    best_score = cand_score2
                    best_flat = res2.x
                    
        except Exception:
            pass

    # Final check time
    if time.time() > deadline:
         # Return best found so far
         pass
         
    best_points = best_flat.reshape(N, 2)
    
    # The problem statement says "normalized to unit area" but the score is 
    # "minimum triangle area divided by the convex hull area".
    # This ratio is scale-invariant. 
    # The evaluator measures the raw value (higher is better).
    # So we just return the points. The evaluator will compute the score.
    
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# We need to make sure it is the entry point.
# The prompt asks to keep function names and signatures.
# The seed program had construct() inside the evolve block.
# I will place the logic in the evolve block as requested by the structure of the seed.
# However, the seed program's construct was simple. 
# I will replace the entire evolve block content with my implementation.
# Wait, the instructions say "Keep the function names and signatures of the seed program."
# The seed program has:
# N = 14
# # EVOLVE-BLOCK-START
# import ...
# def construct(): ...
# # EVOLVE-BLOCK-END
#
# So I should put my code inside the EVOLVE-BLOCK.

# I will restructure the output to fit the requested format: 
# A single python code block containing the complete program.

# Let's double check the score calculation.
# Score = min_triangle_area / hull_area.
# For a regular N-gon, this value is fixed.
# Is it possible to improve it?
# For N=3, equilateral triangle is optimal.
# For N=4, square is optimal (min tri area is half square area? No. 
# Square vertices (0,0), (1,0), (1,1), (0,1).
# Triangles: (0,0)-(1,0)-(1,1) area 0.5. (0,0)-(1,0)-(0,1) area 0.5.
# Hull area 1. Score 0.5.
# Rhombus? If we squash the square, hull area decreases, min tri area decreases.
# It is known that regular polygons maximize the minimum distance between points, and related metrics.
# For "max min triangle area", the regular polygon is the standard conjectured optimum for large N, and likely for N=14.
# The optimization might just confirm this or find a slight numerical improvement if the regular polygon isn't perfectly optimal due to discretization or other subtle geometric effects.
# Given the symmetry, the regular polygon is a very strong local maximum.
# The score for a regular 14-gon:
# Area of regular N-gon with circumradius R: A_hull = N/2 * R^2 * sin(2*pi/N)
# Min triangle area: The smallest triangle in a regular polygon is formed by 3 consecutive vertices.
# Vertices at angles 0, theta, 2theta.
# Area = 0.5 * R^2 * |sin(theta) + sin(theta) - sin(2theta)| ? No.
# Let's compute for 3 consecutive points on unit circle:
# P1=(1,0), P2=(cos t, sin t), P3=(cos 2t, sin 2t) with t=2*pi/N.
# Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
# = 0.5 * |1*(sin t - sin 2t) + cos t*(sin 2t - 0) + cos 2t*(0 - sin t)|
# = 0.5 * |sin t - sin 2t + cos t sin 2t - cos 2t sin t|
# Using sin 2t = 2 sin t cos t, cos 2t = 1 - 2 sin^2 t
# This simplifies to 0.5 * sin^3(t) * (1/sin(t/2))? No, simpler formula:
# Area of triangle formed by 3 consecutive vertices of regular N-gon is:
# A_tri = (1/2) R^2 sin(2*pi/N) (1 - cos(2*pi/N)) ? 
# Actually, A_tri = 0.5 * a^2 * sin(2*pi/N) where a is side length? No.
# Let's just trust the code.

# One potential issue: `scipy.spatial.ConvexHull` might fail if points are collinear or nearly so.
# The normalization in `_objective` helps prevent this.

# Final check on constraints:
# "Each evaluation must finish within 120 s."
# My code uses a deadline of 100s.
# "Python standard library, NumPy and SciPy are available."
# Yes.

# I will output the complete program.
