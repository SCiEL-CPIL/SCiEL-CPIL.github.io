"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy: Smoothed global search using soft-min approximation and L-BFGS-B.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 115.0  # seconds, safety margin before 120s limit

def _min_triangle_area(pts):
    """
    Compute the minimum area of any triangle formed by three points.
    Returns the raw area (not normalized).
    """
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                cross = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _check_inside(points):
    """
    Check if all points are inside or on the boundary of the equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, S3H)
    """
    for p in points:
        x, y = p[0], p[1]
        # y >= 0
        # y <= S3H * x (left edge)
        # y <= -S3H * x + S3H (right edge)
        if y < -1e-9:
            return False
        if y > S3H * x + 1e-9:
            return False
        if y > -S3H * x + S3H + 1e-9:
            return False
    return True

def _project_to_triangle(p):
    """
    Project a point to the nearest point inside the triangle if it's outside.
    This is a simple heuristic projection.
    """
    x, y = p[0], p[1]
    # If y < 0, project to y=0
    if y < 0:
        y = 0.0
    # If y > S3H * x, project to line y = S3H * x
    # Line: S3H*x - y = 0. Normal: (S3H, -1).
    # Distance d = (S3H*x - y) / sqrt(S3H^2 + 1)
    # If d > 0, we are outside (above the line).
    # Closest point on line: p - d * n
    # Let's just clamp for simplicity in the optimizer constraints if possible, 
    # but L-BFGS-B uses bounds, so we need a different approach or penalty.
    # Actually, for L-BFGS-B, we can use bounds on x and y, but the triangle is not a box.
    # We will use a penalty function or just rely on the fact that the optimum is interior 
    # and initialize inside. If it goes outside, we penalize.
    
    # For now, return the point as is, but the objective will penalize violations.
    return np.array([x, y])

def _all_triangle_areas(pts):
    """
    Compute all 165 triangle areas.
    """
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                cross = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                areas.append(0.5 * cross)
    return np.array(areas)

def _soft_min_objective(flat_pts, beta=50.0):
    """
    Objective function for minimization.
    We want to maximize the minimum area, so we minimize the negative soft-min.
    Soft min: -1/beta * log(sum(exp(-beta * a_i)))
    Gradient is available via autograd or we can just use numerical, but L-BFGS-B uses numerical.
    To make it smooth, we use the log-sum-exp trick.
    """
    pts = flat_pts.reshape(N, 2)
    
    # Check constraints (penalty if outside)
    # y >= 0, y <= S3H*x, y <= -S3H*x + S3H
    penalty = 0.0
    for p in pts:
        x, y = p[0], p[1]
        if y < 0:
            penalty += (y)**2
        if y > S3H * x:
            penalty += (y - S3H * x)**2
        if y > -S3H * x + S3H:
            penalty += (y - (-S3H * x + S3H))**2
            
    if penalty > 1e-6:
        return penalty * 100.0 # Heavy penalty

    areas = _all_triangle_areas(pts)
    
    # Soft min approximation
    # To avoid overflow/underflow, subtract max
    max_area = np.max(areas)
    # We want min(areas). 
    # log-sum-exp(-beta * areas) approximates -beta * min(areas) for large beta.
    # So -1/beta * log(sum(exp(-beta * areas))) approximates min(areas).
    # We want to maximize min(areas), so minimize -min(areas).
    # Obj = - ( -1/beta * log(sum(exp(-beta * areas))) ) = 1/beta * log(sum(exp(-beta * areas)))
    
    exp_terms = np.exp(-beta * (areas - max_area))
    log_sum = np.log(np.sum(exp_terms)) + (-beta * max_area)
    soft_min_val = -1.0 / beta * log_sum
    
    return -soft_min_val

def _random_point_in_triangle():
    """Generate a random point uniformly inside the equilateral triangle."""
    u = random.random()
    v = random.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    return np.array([u + 0.5 * v, S3H * v])

def _initial_guess():
    """
    Generate a structured initial guess.
    """
    # Start with a grid-like distribution in barycentric coordinates
    # Or just random points
    pts = []
    for _ in range(N):
        pts.append(_random_point_in_triangle())
    return np.array(pts)

def construct():
    start_time = time.time()
    
    best_pts = None
    best_score = -1
    
    # Use multiple restarts with different random seeds and initial guesses
    # And vary the 'beta' (sharpness of soft-min)
    
    np.random.seed(42)
    
    # Initial structured guess: vertices and midpoints
    structured_guess = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.25, S3H/2],
        [0.75, S3H/2],
        [0.5, S3H/4],
        [0.25, S3H/4],
        [0.75, S3H/4],
        [0.5, S3H/2 + 0.1], # Slightly perturb to avoid symmetry issues
        [0.3, 0.1],
        [0.7, 0.1]
    ])
    
    # Ensure structured guess is inside
    if not _check_inside(structured_guess):
        structured_guess = _initial_guess()
    
    initial_guesses = [structured_guess]
    for _ in range(10):
        initial_guesses.append(_initial_guess())
    
    for guess in initial_guesses:
        if time.time() - start_time > DEADLINE:
            break
            
        flat_guess = guess.flatten()
        
        # Bounds for L-BFGS-B
        # x in [0, 1], y in [0, S3H]
        # But we need tighter bounds to keep them inside the triangle roughly
        # We'll rely on the penalty function for strict triangle constraints
        bounds = [(0, 1), (0, S3H)] * N
        
        # Try different betas
        for beta in [20.0, 50.0, 100.0]:
            if time.time() - start_time > DEADLINE:
                break
                
            try:
                res = minimize(
                    _soft_min_objective,
                    flat_guess,
                    method='L-BFGS-B',
                    bounds=bounds,
                    args=(beta,),
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                if res.success:
                    pts = res.x.reshape(N, 2)
                    if _check_inside(pts):
                        score = _min_triangle_area(pts)
                        if score > best_score:
                            best_score = score
                            best_pts = pts.copy()
            except Exception:
                pass

    # If we didn't find a good solution, fallback to random search with local improvement
    if best_pts is None or best_score < 0.01:
        # Simple random restarts
        for _ in range(50):
            if time.time() - start_time > DEADLINE:
                break
            pts = _initial_guess()
            score = _min_triangle_area(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
                
    # Final polish: local search on the best point found
    if best_pts is not None:
        current_pts = best_pts.copy()
        current_score = best_score
        step_size = 0.05
        for _ in range(1000):
            if time.time() - start_time > DEADLINE:
                break
            idx = random.randint(0, N-1)
            old_p = current_pts[idx].copy()
            new_p = old_p + np.random.normal(0, step_size, 2)
            
            # Simple projection
            if new_p[1] < 0: new_p[1] = 0
            if new_p[1] > S3H * new_p[0]: new_p[1] = S3H * new_p[0]
            if new_p[1] > -S3H * new_p[0] + S3H: new_p[1] = -S3H * new_p[0] + S3H
            if new_p[0] < 0: new_p[0] = 0
            if new_p[0] > 1: new_p[0] = 1
            
            current_pts[idx] = new_p
            new_score = _min_triangle_area(current_pts)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            else:
                current_pts[idx] = old_p
                
            # Reduce step size occasionally
            if _ % 100 == 0:
                step_size *= 0.9

    if best_pts is None:
        # Absolute fallback
        best_pts = _initial_guess()
        
    return best_pts

# For testing
if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Min area (normalized):", _min_triangle_area(pts) / (S3H / 2))
