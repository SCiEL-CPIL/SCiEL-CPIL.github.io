import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

def _score_from_points(points):
    """Calculate min triangle area / hull area for given points."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return -1.0
        
    if hull_area < 1e-10:
        return -1.0
        
    # Calculate minimum triangle area
    min_area = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
                    
    return min_area / hull_area

def _points_from_fourier(params):
    """
    Convert Fourier parameters to points.
    params: [r0, a1, b1, a2, b2]
    r(i) = r0 + a1*cos(2*pi*i/N + phase1) + a2*cos(4*pi*i/N + phase2)
    where phase is encoded in b1, b2 via arctan2
    """
    r0 = params[0]
    a1 = params[1]
    b1 = params[2]
    a2 = params[3]
    b2 = params[4]
    
    # Compute radius for each vertex
    i = np.arange(N)
    base_angle = 2 * np.pi * i / N
    
    # Mode 1: cos(theta + phi1) where phi1 = arctan2(b1, a1)
    # Actually, let's use r(i) = r0 + a1*cos(base_angle) + b1*sin(base_angle) + a2*cos(2*base_angle) + b2*sin(2*base_angle)
    # This is more direct. Let's redefine params as [r0, c1, s1, c2, s2]
    # where r(i) = r0 + c1*cos(base_angle) + s1*sin(base_angle) + c2*cos(2*base_angle) + s2*sin(2*base_angle)
    
    c1 = params[1]
    s1 = params[2]
    c2 = params[3]
    s2 = params[4]
    
    radii = r0 + c1*np.cos(base_angle) + s1*np.sin(base_angle) + c2*np.cos(2*base_angle) + s2*np.sin(2*base_angle)
    
    # Ensure radii are positive (add small epsilon if needed, or clip)
    radii = np.maximum(radii, 0.1)
    
    # Convert to Cartesian
    x = radii * np.cos(base_angle)
    y = radii * np.sin(base_angle)
    points = np.column_stack((x, y))
    
    return points

def _negative_score_fourier(params):
    points = _points_from_fourier(params)
    score = _score_from_points(points)
    if score < 0:
        return 1.0  # Large penalty
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    # Initial guess: Regular 14-gon (r0=1, all other coefficients 0)
    params_init = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    
    best_score = _score_from_points(_points_from_fourier(params_init))
    best_params = params_init.copy()
    
    # Try multiple random starts
    np.random.seed(42)
    starts = [params_init]
    for _ in range(20):
        # Random small perturbations
        r0 = np.random.uniform(0.8, 1.2)
        c1 = np.random.uniform(-0.3, 0.3)
        s1 = np.random.uniform(-0.3, 0.3)
        c2 = np.random.uniform(-0.3, 0.3)
        s2 = np.random.uniform(-0.3, 0.3)
        starts.append(np.array([r0, c1, s1, c2, s2]))
    
    for init in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(_negative_score_fourier, init, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_params = res.x
        except Exception:
            pass
            
    # Convert best parameters to points
    points = _points_from_fourier(best_params)
    
    return points
