"""
Strategy: Hexagonal Lattice Grid Parameterization
Exploits the high symmetry of hexagonal packings.
Models 14 points as a 2x7 grid on a lattice defined by two basis vectors.
This reduces the search space from 28 (free coordinates) to 4 (lattice parameters).
Uses log-sum-exp for a smooth, differentiable surrogate of the min-triangle-area objective.
"""

N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns 0 if invalid.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    if n < 3:
        return 0.0
        
    # Center points to remove translation invariance
    points = points - np.mean(points, axis=0)
    
    # Check for degenerate points
    if np.allclose(points, 0):
        return 0.0

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
        
        # Scale to unit area
        scale = np.sqrt(hull_area)
        scaled_points = points / scale
        
        # Calculate min triangle area
        min_area = float('inf')
        for i in range(n - 2):
            p1 = scaled_points[i]
            for j in range(i + 1, n - 1):
                p2 = scaled_points[j]
                v1 = p2 - p1
                for k in range(j + 1, n):
                    p3 = scaled_points[k]
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    if area < min_area:
                        min_area = area
                        
        return min_area
    except Exception:
        return 0.0

def _generate_lattice_points(params):
    """
    Generate 14 points from lattice parameters.
    params: [a, b, theta, aspect]
    - a: length of basis vector 1
    - b: length of basis vector 2
    - theta: angle of basis vector 2 relative to basis vector 1 (in radians)
    - aspect: scaling factor for the 7th dimension (width of grid)
    
    The grid is 2 rows x 7 columns.
    Row 0: (i * v1) for i in 0..6
    Row 1: (i * v1 + v2 * aspect) for i in 0..6
    """
    a, b, theta, aspect = params
    
    # Basis vectors
    v1 = np.array([a, 0.0])
    v2 = np.array([b * np.cos(theta), b * np.sin(theta)])
    
    points = np.zeros((N, 2))
    
    # Generate 7 points along v1
    for i in range(7):
        points[i] = i * v1
        
    # Generate 7 points along v1 + v2*aspect
    for i in range(7):
        points[i + 7] = i * v1 + v2 * aspect
        
    return points

def _objective_lattice(params):
    """
    Objective function for lattice optimization.
    We want to maximize min_triangle_area / hull_area.
    Since scaling the lattice scales both areas by the same factor (area^2 for total, area for triangles? No.)
    
    Let's check scaling:
    If we scale points by s:
    Triangle areas scale by s^2.
    Hull area scales by s^2.
    Ratio is invariant to scaling.
    So we can fix one parameter to remove scale invariance, or just let the optimizer handle it.
    However, log-sum-exp is sensitive to scale.
    It is better to normalize inside the objective or fix a scale.
    
    Let's fix the area of the parallelogram spanned by v1 and v2 to 1?
    Or simply normalize the points to unit hull area inside the score calculation, 
    but for the surrogate, we need a smooth proxy.
    
    Smooth proxy for min_area:
    -logsumexp(-areas / T) * T  approximates min(areas) as T -> 0.
    
    We want to maximize min_area / hull_area.
    Let A_min be the min triangle area.
    Let A_hull be the hull area.
    Score = A_min / A_hull.
    
    Surrogate:
    S(T) = (-1/T * log(sum(exp(-A_i/T)))) / A_hull
    As T -> 0, numerator -> A_min.
    
    We minimize -S(T).
    """
    try:
        points = _generate_lattice_points(params)
        
        # Center points
        points = points - np.mean(points, axis=0)
        
        # Calculate all triangle areas
        n = points.shape[0]
        areas = []
        for i in range(n - 2):
            p1 = points[i]
            for j in range(i + 1, n - 1):
                p2 = points[j]
                v1 = p2 - p1
                for k in range(j + 1, n):
                    p3 = points[k]
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    areas.append(area)
        
        areas = np.array(areas)
        
        if np.any(areas < 1e-12):
            return 1e6 # Degenerate
            
        # Calculate hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 1e6
            
        if hull_area < 1e-12:
            return 1e6
            
        # Log-sum-exp approximation for min area
        # Use a small temperature for a tight approximation
        T = 0.01 
        # To avoid overflow, shift by max
        shifted = -areas / T
        shifted -= np.max(shifted)
        log_sum_exp = np.log(np.sum(np.exp(shifted)))
        
        min_area_approx = -T * log_sum_exp
        
        score = min_area_approx / hull_area
        
        return -score
        
    except Exception:
        return 1e6

def _objective_lattice_soft(params):
    """
    A softer version for initial exploration, using a larger T.
    """
    try:
        points = _generate_lattice_points(params)
        points = points - np.mean(points, axis=0)
        
        n = points.shape[0]
        areas = []
        for i in range(n - 2):
            p1 = points[i]
            for j in range(i + 1, n - 1):
                p2 = points[j]
                v1 = p2 - p1
                for k in range(j + 1, n):
                    p3 = points[k]
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    areas.append(area)
        
        areas = np.array(areas)
        
        if np.any(areas < 1e-12):
            return 1e6
            
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 1e6
            
        if hull_area < 1e-12:
            return 1e6
            
        T = 0.1
        shifted = -areas / T
        shifted -= np.max(shifted)
        log_sum_exp = np.log(np.sum(np.exp(shifted)))
        
        min_area_approx = -T * log_sum_exp
        
        score = min_area_approx / hull_area
        
        return -score
        
    except Exception:
        return 1e6

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Initial guesses for lattice parameters [a, b, theta, aspect]
    # Hexagonal lattice: theta = pi/3, a = b
    # Grid 2x7.
    # We want to maximize min triangle area.
    # In a regular triangular lattice, the smallest triangles are the elementary ones.
    # For a 2x7 grid, the "aspect" determines the vertical offset.
    # If aspect = 1, it's a parallelogram.
    
    rng = np.random.default_rng(42)
    
    inits = []
    
    # 1. Standard hexagonal
    inits.append(np.array([1.0, 1.0, np.pi/3, 1.0]))
    
    # 2. Perturbed hexagonal
    for _ in range(5):
        a = 1.0 + rng.uniform(-0.1, 0.1)
        b = 1.0 + rng.uniform(-0.1, 0.1)
        theta = np.pi/3 + rng.uniform(-0.1, 0.1)
        aspect = 1.0 + rng.uniform(-0.1, 0.1)
        inits.append(np.array([a, b, theta, aspect]))
        
    # 3. Rectangular-ish (theta = pi/2)
    inits.append(np.array([1.0, 1.0, np.pi/2, 1.0]))
    
    # 4. Random
    for _ in range(5):
        a = rng.uniform(0.5, 2.0)
        b = rng.uniform(0.5, 2.0)
        theta = rng.uniform(np.pi/6, 5*np.pi/6)
        aspect = rng.uniform(0.5, 1.5)
        inits.append(np.array([a, b, theta, aspect]))

    # Phase 1: Soft optimization to find basin
    for init in inits:
        if time.time() > deadline - 20:
            break
            
        try:
            res = minimize(_objective_lattice_soft, init, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < 1e6:
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_params = res.x
        except Exception:
            continue

    # Phase 2: Hard optimization (polish) from best soft result
    if best_score > 0:
        if time.time() < deadline - 10:
            try:
                # Use the best parameters found so far as the start for the hard objective
                # We might need to re-evaluate best_params since it was only tracked implicitly
                # Let's re-run minimization from the best few candidates found in Phase 1
                # For simplicity, just run one more polish from the best init
                # Actually, let's just run the hard objective on the best init from phase 1
                # We didn't store all results, so let's just re-run from the standard hex init
                # and the best found if we stored it.
                
                # Let's store the best params from Phase 1 properly
                # I'll modify the loop to store best_params
                pass 
            except Exception:
                pass
    
    # Redo Phase 1 to ensure we capture the best params
    best_params = None
    best_soft_score = -np.inf
    
    for init in inits:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(_objective_lattice_soft, init, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < 1e6:
                s = -res.fun
                if s > best_soft_score:
                    best_soft_score = s
                    best_params = res.x
        except Exception:
            continue

    # Phase 2: Polish with hard objective
    if best_params is not None and time.time() < deadline - 10:
        try:
            res = minimize(_objective_lattice, best_params, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < 1e6:
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_params = res.x
        except Exception:
            pass

    # Generate final points
    if best_params is not None:
        final_points = _generate_lattice_points(best_params)
        # Normalize to unit area hull? 
        # The score function in the evaluator handles normalization.
        # But we should return a valid set of points.
        # The construct() function returns points.
        # The evaluator calculates score.
        
        # Check if this point set is valid and calculate its score to update best_score if needed
        # (In case the surrogate score was slightly off)
        s = _score(final_points)
        if s > best_score:
            best_score = s
            best_points = final_points
    else:
        # Fallback to regular polygon if lattice optimization failed
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _score(best_points)

    return best_points
# EVOLVE-BLOCK-END
