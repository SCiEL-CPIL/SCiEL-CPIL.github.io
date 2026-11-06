"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import cos, sin, pi, log, sqrt, exp
from scipy.optimize import minimize

def _log_product_of_distances(pts):
    """Compute log of product of all ordered pair distances (28 pairs)."""
    n = len(pts)
    log_prod = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-300:
                return -1e300
            log_prod += log(d)
    return log_prod

def _diameter(pts):
    """Compute the diameter (max pairwise distance)."""
    n = len(pts)
    dmax = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d > dmax:
                dmax = d
    return dmax

def _score_from_points(pts):
    """Compute the score: Delta / n^n where Delta is product of ordered-pair distances
    after scaling to diameter 2. For n=8, ordered pairs = 8*7 = 56, so product of 
    all 56 ordered distances. But actually 'product over ordered pairs' likely means
    product over all (i,j) with i != j, which is 56 terms. However, the regular 
    octagon scores exactly 1. Let me think...
    
    Actually, re-reading: "Delta is the product over ordered pairs of distances after 
    scaling the set to diameter 2". Ordered pairs means (i,j) for i != j, so n*(n-1) = 56 terms.
    The score is Delta / n^n = Delta / 8^8.
    
    For a regular octagon with diameter 2 (so circumradius = 1/sin(pi/8) * ... let me compute).
    Regular octagon vertices on circle of radius R. Diameter = 2R if opposite vertices.
    For regular octagon, opposite vertices are at distance 2R. So diameter = 2R = 2, R = 1.
    
    Distances in regular octagon (R=1):
    - Adjacent: 2*sin(pi/8)
    - Skip 1: 2*sin(2*pi/8) = 2*sin(pi/4)
    - Skip 2: 2*sin(3*pi/8)
    - Opposite: 2
    
    There are 8 of each distance type (each vertex has 4 distinct distances to other 7 vertices,
    but counting ordered pairs: each unordered pair counted twice).
    
    Unordered pairs: 28 total.
    - 8 adjacent pairs: distance 2*sin(pi/8)
    - 8 skip-1 pairs: distance 2*sin(pi/4)  
    - 8 skip-2 pairs: distance 2*sin(3*pi/8)
    - 4 opposite pairs: distance 2
    
    Product of unordered pair distances = (2*sin(pi/8))^8 * (2*sin(pi/4))^8 * (2*sin(3*pi/8))^8 * 2^4
    Product of ordered pair distances = (product of unordered)^2
    
    Score = (product of ordered) / 8^8
    
    For regular octagon this should equal 1.
    """
    n = len(pts)
    d = _diameter(pts)
    if d < 1e-300:
        return 0.0
    # Scale to diameter 2
    scaled = [p / d * 2.0 for p in pts]
    # Compute product of ordered pair distances
    log_prod = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = np.linalg.norm(scaled[i] - scaled[j])
            if dist < 1e-300:
                return 0.0
            log_prod += log(dist)
    # Score = exp(log_prod) / n^n
    score = exp(log_prod - n * log(n))
    return score

def _neg_score(pts_flat):
    """Negative score for minimization."""
    n = 8
    pts = np.array(pts_flat).reshape(n, 2)
    s = _score_from_points(pts.tolist())
    return -s

def _cage_topology():
    """Create a cage topology: two concentric squares (inner and outer)."""
    # Outer square vertices
    outer = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    # Inner square vertices (rotated 45 degrees, smaller radius)
    r = 0.5
    inner = [[r * cos(pi/4 + 2*pi*k/4), r * sin(pi/4 + 2*pi*k/4)] for k in range(4)]
    return outer + inner

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_score = 0.0
    best_points = None
    
    # Initialize with regular octagon
    from math import cos, sin, pi
    octagon = [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
    oct_score = _score_from_points(octagon)
    best_score = oct_score
    best_points = octagon
    
    # Try cage topology
    cage = _cage_topology()
    cage_score = _score_from_points(cage)
    if cage_score > best_score:
        best_score = cage_score
        best_points = cage
    
    # Use multi-stage penalty BFGS optimization
    # We optimize over 16 coordinates (8 points x 2)
    # Objective: maximize score, which is product of ordered distances / 8^8 after scaling to diameter 2
    
    # For optimization, we work in log-space to avoid numerical issues
    # The score is scale-invariant (scaling to diameter 2 makes it so), so we can fix one constraint
    # to avoid the scale ambiguity. We'll use the fact that score is invariant under scaling.
    
    # Actually, let's directly optimize the log-score with BFGS.
    # We need to be careful about the diameter constraint - but since score is computed
    # by scaling to diameter 2, the objective is scale-invariant.
    
    # Let's use a different approach: optimize the points directly, computing score each time.
    # Use BFGS with multiple restarts from different initializations.
    
    initializations = [octagon, cage]
    
    # Add some random perturbations
    np.random.seed(42)
    for _ in range(5):
        perturbed = [[p[0] + np.random.randn() * 0.1, p[1] + np.random.randn() * 0.1] for p in cage]
        initializations.append(perturbed)
    
    for init in initializations:
        if time.time() > deadline:
            break
        
        init_flat = [coord for p in init for coord in p]
        
        # Use BFGS to maximize score (minimize negative score)
        try:
            result = minimize(
                _neg_score,
                init_flat,
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < 0:
                opt_points = np.array(result.x).reshape(8, 2).tolist()
                opt_score = -result.fun
                
                # Verify and update best
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points
                    
        except Exception:
            pass
    
    # Final polish: try to improve with tighter tolerance
    if best_points is not None and time.time() < deadline:
        best_flat = [coord for p in best_points for coord in p]
        try:
            result = minimize(
                _neg_score,
                best_flat,
                method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-12}
            )
            if result.fun < 0 and -result.fun > best_score:
                best_score = -result.fun
                best_points = np.array(result.x).reshape(8, 2).tolist()
        except Exception:
            pass
    
    # Ensure points are distinct and coordinates are within bounds
    if best_points is not None:
        # Check distinctness
        for i in range(8):
            for j in range(i + 1, 8):
                if np.linalg.norm(np.array(best_points[i]) - np.array(best_points[j])) < 1e-10:
                    # Perturb to make distinct
                    best_points[j][0] += 1e-6
                    best_points[j][1] += 1e-6
        
        # Ensure coordinates are within bounds
        for p in best_points:
            p[0] = max(-1e5, min(1e5, p[0]))
            p[1] = max(-1e5, min(1e5, p[1]))
    
    return best_points
# EVOLVE-BLOCK-END
