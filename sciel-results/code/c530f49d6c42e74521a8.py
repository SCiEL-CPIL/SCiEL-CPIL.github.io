import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11

# Vertices of the equilateral triangle
# A = (0,0), B = (1,0), C = (0.5, sqrt(3)/2)
# Area = sqrt(3)/4
AREA_TRI = np.sqrt(3) / 4.0

# Pre-compute indices for all C(11,3) = 165 triangles
tri_indices = list(combinations(range(N), 3))
tri_indices_arr = np.array(tri_indices)

def calculate_min_area(points):
    """
    points: (N, 2) array
    Returns: minimum area of any triangle formed by 3 points
    """
    # Extract coordinates for all triangles
    p1 = points[tri_indices_arr[:, 0]]
    p2 = points[tri_indices_arr[:, 1]]
    p3 = points[tri_indices_arr[:, 2]]
    
    # Area = 0.5 * | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |
    # Vectorized calculation
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def check_inside(points):
    """
    Check if all points are inside or on the boundary of the triangle.
    Returns True if valid, False otherwise.
    """
    # y >= 0
    if np.any(points[:, 1] < -1e-9):
        return False
    # y <= sqrt(3) * x  => sqrt(3)*x - y >= 0
    if np.any(np.sqrt(3) * points[:, 0] - points[:, 1] < -1e-9):
        return False
    # y <= -sqrt(3) * (x - 1) => -sqrt(3)*x + y + sqrt(3) >= 0
    if np.any(-np.sqrt(3) * points[:, 0] + points[:, 1] + np.sqrt(3) < -1e-9):
        return False
    return True

def objective_de(x):
    """
    Objective function for Differential Evolution.
    Returns a value to be minimized.
    We want to maximize min_area, so we minimize -min_area.
    We add a penalty if points are outside.
    """
    points = x.reshape(N, 2)
    
    # Penalty for being outside
    # Calculate distance to boundaries
    # 1. y < 0
    pen1 = np.sum(np.maximum(0, -points[:, 1]))
    # 2. sqrt(3)*x - y < 0
    pen2 = np.sum(np.maximum(0, -(np.sqrt(3) * points[:, 0] - points[:, 1])))
    # 3. -sqrt(3)*x + y + sqrt(3) < 0
    pen3 = np.sum(np.maximum(0, -(-np.sqrt(3) * points[:, 0] + points[:, 1] + np.sqrt(3))))
    
    total_penalty = 100.0 * (pen1 + pen2 + pen3)
    
    if total_penalty > 1e-6:
        return 1000.0 + total_penalty
        
    min_area = calculate_min_area(points)
    return -min_area

def objective_lbfgs(x, k=50.0):
    """
    Smoothed objective for L-BFGS-B.
    Minimizes -log(sum(exp(-k * area_i))) / k
    This is approx -min(area_i) for large k.
    """
    points = x.reshape(N, 2)
    
    # Check boundaries, if outside, return large value
    # We assume DE found a valid point, but L-BFGS might step out.
    # Simple penalty
    if not check_inside(points):
        return 1000.0

    p1 = points[tri_indices_arr[:, 0]]
    p2 = points[tri_indices_arr[:, 1]]
    p3 = points[tri_indices_arr[:, 2]]
    
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    # Avoid log(0)
    areas = np.maximum(areas, 1e-10)
    
    # Log-sum-exp trick for stability
    # -1/k * log(sum(exp(-k * areas)))
    # Let m = min(areas). sum(exp(-k*a)) = exp(-k*m) * sum(exp(-k*(a-m)))
    # log(sum) = -k*m + log(sum(exp(-k*(a-m))))
    # result = -1/k * (-k*m + log(...)) = m - 1/k * log(...)
    # This approximates min(areas) as k -> inf.
    # We want to MAXIMIZE min area, so we MINIMIZE -min area.
    # So objective is - (m - 1/k * log(...)) = -m + 1/k * log(...)
    
    m = np.min(areas)
    shifted = areas - m
    log_sum = np.log(np.sum(np.exp(-k * shifted)))
    smooth_min = m - (1.0 / k) * log_sum
    
    return -smooth_min

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    # Bounds for DE: x in [0, 1], y in [0, sqrt(3)/2]
    # To be safe, we can use slightly wider bounds and rely on penalty
    bounds = [(0.0, 1.0), (0.0, np.sqrt(3)/2)] * N
    
    # Initial DE run
    # Maxiter is tricky. Let's run for a fixed time or iterations.
    # DE is slow. Let's try to run it for a portion of the time.
    
    best_x = None
    best_val = np.inf
    
    # We will run DE for a set number of generations, checking time.
    # Or use 'maxiter' and 'polish'
    
    # To ensure we don't timeout, we can wrap the DE in a loop or use a callback.
    # Scipy DE doesn't have a simple time-based stop callback in older versions, 
    # but we can just run a reasonable number of iterations.
    # 11 points * 2 coords = 22 dims.
    # Let's try maxiter=50, popsize=15.
    
    try:
        # Run DE
        # tolerance is for convergence
        res = differential_evolution(
            objective_de, 
            bounds, 
            maxiter=30, 
            popsize=15, 
            tol=1e-4,
            seed=42,
            polish=False # We will do our own polish
        )
        
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            
        # Check time
        elapsed = time.time() - start_time
        if elapsed > timeout - 20:
            # Not enough time for polish, return best from DE
            points = best_x.reshape(N, 2)
            if check_inside(points):
                return points
            else:
                # Fallback to a known good configuration if DE failed badly
                # Simple grid or random? Let's return the center if invalid.
                # But DE penalty should keep it valid.
                return np.zeros((N, 2)) # Placeholder, should not happen
        
        # Polish with L-BFGS-B
        # Use the best result from DE as initial guess
        x0 = best_x
        
        # We might want to try polishing from a few different starting points if time permits
        # But L-BFGS-B is fast.
        
        res_lbfgs = minimize(
            objective_lbfgs, 
            x0, 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        if res_lbfgs.fun < best_val:
            best_x = res_lbfgs.x
            best_val = res_lbfgs.fun
            
    except Exception as e:
        pass

    points = best_x.reshape(N, 2)
    
    # Final validation
    if not check_inside(points):
        # If invalid, try to project or return a safe default
        # Simple projection: clip to triangle? Hard.
        # Let's just return the points and hope the evaluator handles slight boundary violations?
        # No, "on or inside".
        # If DE penalty worked, it should be inside.
        # If not, we have a problem.
        
        # Let's try a quick local repair: move points towards centroid if outside?
        # Centroid is (0.5, sqrt(3)/6)
        # This is risky.
        
        # Alternative: If invalid, return a grid of points?
        # For N=11, a simple grid might be better than invalid.
        # But the prompt says "invalid result... scores 0".
        # So we must return valid points.
        
        # Let's try to fix by moving points that are outside towards the center.
        center = np.array([0.5, np.sqrt(3)/6])
        for i in range(N):
            p = points[i]
            # Check if p is outside
            # We can just check the 3 inequalities
            if p[1] < 0:
                points[i, 1] = 1e-6
            if np.sqrt(3)*p[0] - p[1] < 0:
                # Move towards line y=sqrt(3)x
                # Project p onto line? Or just move to center?
                points[i] = center
            if -np.sqrt(3)*p[0] + p[1] + np.sqrt(3) < 0:
                points[i] = center
                
        # Re-check
        if not check_inside(points):
            # Last resort: return center for all? That gives min area 0.
            # Better to return a known valid configuration.
            # Let's generate a simple valid configuration:
            # 1 center, 10 on boundary?
            # Or just return the initial random seed if it was valid?
            # The seed program generated random points.
            
            # Let's just return the center point repeated? No, min area 0.
            # Let's return a grid.
            # 3x3 grid + 2 extra?
            # Just return the points from the first valid DE iteration?
            # We didn't store that.
            
            # Let's assume DE penalty is strong enough.
            # If it fails, we return a safe static configuration.
            # Static config:
            # (0.5, sqrt(3)/6) center
            # Others distributed.
            
            # Let's just use the points from `res` if it exists, otherwise static.
            # If `res` exists but is invalid, something is wrong.
            
            # Let's try to generate a simple valid set:
            # 11 points.
            # Maybe a regular hexagon + center + 4 more?
            # Let's just return the points and trust the penalty.
            # If it's invalid, the score is 0.
            # To be safe, let's implement a simple fallback.
            
            # Fallback: 11 points on a grid inside the triangle.
            # This is not optimal but valid.
            pts = []
            # Center
            pts.append([0.5, np.sqrt(3)/6])
            # Ring 1
            for i in range(6):
                angle = i * np.pi / 3
                r = 0.2
                x = 0.5 + r * np.cos(angle)
                y = np.sqrt(3)/6 + r * np.sin(angle)
                # Check if inside
                if check_inside(np.array([x,y])):
                    pts.append([x,y])
            # Fill remaining with random valid points
            rng = np.random.default_rng(42)
            while len(pts) < N:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = np.sqrt(3) / 2 * v
                    pts.append([x,y])
            points = np.array(pts[:N])
            
    return points

