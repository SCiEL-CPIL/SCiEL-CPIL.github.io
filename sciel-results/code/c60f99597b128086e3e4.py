import itertools
import numpy as np
import time
import random

N = 11
# Pre-compute indices for combinations of 3 from 11
# 11 choose 3 = 165
tri_indices = np.array(list(itertools.combinations(range(N), 3)))

def min_triangle_area(pts):
    # pts: (11, 2)
    p1 = pts[tri_indices[:, 0]]
    p2 = pts[tri_indices[:, 1]]
    p3 = pts[tri_indices[:, 2]]
    
    # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    areas = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    return np.min(areas)

def is_inside(p):
    x, y = p
    if x < 0 or x > 1 or y < 0:
        return False
    h = np.sqrt(3) / 2
    if y > h:
        return False
    # Check against slanted sides
    # Left side: y <= sqrt(3) * x
    # Right side: y <= sqrt(3) * (1 - x)
    if x < 0.5:
        if y > np.sqrt(3) * x + 1e-9:
            return False
    else:
        if y > np.sqrt(3) * (1 - x) + 1e-9:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s buffer
    
    # Initial points
    # Use a symmetric initialization
    # Vertices
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    p2 = np.array([0.5, np.sqrt(3)/2])
    
    # Midpoints
    p3 = np.array([0.5, 0.0])
    p4 = np.array([0.25, np.sqrt(3)/4])
    p5 = np.array([0.75, np.sqrt(3)/4])
    
    # Centroid
    p6 = np.array([0.5, np.sqrt(3)/6])
    
    # 4 more points: Let's place them randomly inside
    # Or better, place them on a smaller similar triangle?
    # Let's just use random for the rest to avoid bias, but ensure they are inside.
    
    points = [p0, p1, p2, p3, p4, p5, p6]
    
    # Add 4 random points
    while len(points) < N:
        x = random.random()
        y = random.random() * np.sqrt(3) * min(x, 1-x)
        points.append(np.array([x, y]))
        
    points = np.array(points)
    current_val = min_triangle_area(points)
    best_points = points.copy()
    best_val = current_val
    
    # SA Parameters
    T = 0.01  # Initial temperature (scale of area)
    # Max area is ~ 0.433. Min area we care about is likely ~ 0.01-0.05.
    # If T is too high, we accept too many bad moves. If too low, we get stuck.
    # Let's estimate typical move change.
    # A move of 0.01 in x changes area by ~ 0.001.
    # So T=0.01 is reasonable for early stage.
    
    alpha = 0.999  # Cooling rate
    min_T = 1e-6
    
    step_size = 0.1  # Initial perturbation size
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Pick a random point to move
        idx = rng.integers(0, N)
        
        # Generate perturbation
        # Use Gaussian noise for smoother moves
        dx = rng.normal(0, step_size)
        dy = rng.normal(0, step_size)
        
        new_points = points.copy()
        new_points[idx] = new_points[idx] + np.array([dx, dy])
        
        # Check bounds
        if is_inside(new_points[idx]):
            new_val = min_triangle_area(new_points)
            
            # Acceptance criterion
            if new_val > current_val:
                points = new_points
                current_val = new_val
                if current_val > best_val:
                    best_val = current_val
                    best_points = points.copy()
            else:
                # Delta is negative
                delta = new_val - current_val
                # Avoid log(0)
                if delta < -1e-9:
                    accept_prob = np.exp(delta / T)
                else:
                    accept_prob = 1.0
                
                if rng.random() < accept_prob:
                    points = new_points
                    current_val = new_val
        
        # Cool down
        T *= alpha
        # Optionally adjust step size
        # step_size *= 0.9999
        
    return best_points
