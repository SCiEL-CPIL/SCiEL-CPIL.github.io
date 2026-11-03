"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])
TRI_AREA = np.sqrt(3)/4

# Precompute indices for all triplets
indices = list(itertools.combinations(range(N), 3))
num_triplets = len(indices)

def _get_areas(points):
    """
    Calculate areas of all triangles formed by the points.
    points: shape (N, 2)
    returns: array of shape (num_triplets,)
    """
    areas = np.zeros(num_triplets)
    for k, (i, j, k_idx) in enumerate(indices):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k_idx]
        # Cross product magnitude / 2
        area = 0.5 * np.abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        areas[k] = area
    return areas

def _min_area(points):
    """
    Calculate the minimum area of any triangle.
    """
    return np.min(_get_areas(points))

def _objective_flat(x, beta):
    """
    Objective function for optimization.
    x: flat array of length 2*N
    beta: temperature parameter for log-sum-exp approximation of min
    We minimize - (approx min area)
    """
    points = x.reshape(N, 2)
    areas = _get_areas(points)
    # Log-sum-exp trick for minimum:
    # min(a) = -lim_{beta->inf} (1/beta) log(sum(exp(-beta * a)))
    # To avoid overflow, we can shift, but areas are small.
    # Actually, if areas are close to 0, exp(-beta * area) can be large if area is negative? 
    # Area is always non-negative.
    # If area is 0, exp(0)=1.
    # We want to maximize min area, so we minimize -min area.
    # Let f = -LSE = (1/beta) log(sum(exp(-beta * a)))
    # We minimize f.
    
    # Stable log-sum-exp:
    # log(sum(exp(-beta * a))) = -beta * min(a) + log(sum(exp(-beta * (a - min(a)))))
    # This is stable.
    min_a = np.min(areas)
    # If min_a is very small, the term dominates.
    # Let's just compute directly with clipping to prevent issues
    # exp(-beta * a)
    # If a is 0, exp(0)=1.
    # If a is large, exp(-beta*a) -> 0.
    
    # To handle the non-differentiability of abs in area calculation if we used signed area,
    # we used abs in _get_areas, so areas are positive.
    
    # Calculate LSE
    # We want to minimize (1/beta) * log( sum( exp(-beta * areas) ) )
    # This is equivalent to minimizing the "soft minimum".
    
    # Shift for numerical stability
    max_exp_arg = np.max(-beta * areas)
    if max_exp_arg > 700: # unlikely since areas > 0
         pass
    
    sum_exp = np.sum(np.exp(-beta * areas))
    if sum_exp == 0:
        return 0.0
    lse = (1.0 / beta) * np.log(sum_exp)
    
    return lse

def _constraints(x):
    """
    Constraints to keep points inside the triangle.
    x: flat array
    Returns list of constraint dicts for SLSQP.
    type='ineq' means f(x) >= 0
    """
    points = x.reshape(N, 2)
    cons = []
    for i in range(N):
        x_i, y_i = points[i]
        # 1. y >= 0
        cons.append({'type': 'ineq', 'fun': lambda xi, yi: yi, 'args': (x_i, y_i)})
        # 2. y <= sqrt(3) * x  =>  sqrt(3)*x - y >= 0
        cons.append({'type': 'ineq', 'fun': lambda xi, yi: np.sqrt(3)*xi - yi, 'args': (x_i, y_i)})
        # 3. y <= -sqrt(3) * x + sqrt(3) => sqrt(3)*x + y - sqrt(3) >= 0
        cons.append({'type': 'ineq', 'fun': lambda xi, yi: np.sqrt(3)*xi + yi - np.sqrt(3), 'args': (x_i, y_i)})
    return cons

def _random_point_in_triangle(rng):
    """Generate a random point inside the triangle."""
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    # Barycentric coordinates
    p = u * V0 + v * V1 + (1 - u - v) * V2
    return p

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 110s safety margin
    
    best_score = -1.0
    best_points = None
    
    # Multi-start strategy
    rng = np.random.default_rng(42)
    
    # Initial guesses: 
    # 1. Random
    # 2. Grid-like
    # 3. Vertices + Centroid + others
    
    num_restarts = 20
    beta_values = [5, 10, 20, 50, 100, 200, 500]
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial points
        if r == 0:
            # Try a structured initial guess: vertices, midpoints, centroid
            points = np.zeros((N, 2))
            points[0] = V0
            points[1] = V1
            points[2] = V2
            points[3] = (V0+V1)/2
            points[4] = (V1+V2)/2
            points[5] = (V2+V0)/2
            points[6] = (V0+V1+V2)/3
            # Add a few random ones
            for i in range(7, N):
                points[i] = _random_point_in_triangle(rng)
        else:
            points = np.array([_random_point_in_triangle(rng) for _ in range(N)])
            
        x0 = points.flatten()
        
        # Run continuation method
        current_x = x0
        for beta in beta_values:
            if time.time() > deadline:
                break
                
            res = minimize(
                _objective_flat, 
                current_x, 
                args=(beta,), 
                method='SLSQP', 
                constraints=_constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if res.success:
                current_x = res.x
                # Check validity
                pts = current_x.reshape(N, 2)
                # Verify inside
                valid = True
                for pt in pts:
                    if pt[1] < -1e-9 or pt[1] > np.sqrt(3)*pt[0] + 1e-9 or pt[1] > -np.sqrt(3)*pt[0] + np.sqrt(3) + 1e-9:
                        valid = False
                        break
                if valid:
                    score = _min_area(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
            else:
                # If optimization failed, try to recover or just move to next beta with same x
                pass
                
    # Final check and return
    if best_points is None:
        # Fallback to random if everything failed
        best_points = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        
    return best_points
