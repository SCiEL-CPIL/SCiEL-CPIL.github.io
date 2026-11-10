"""Improved circle packing using Simulated Annealing with Hexagonal Initialization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
import random

def _calc_radius(points):
    """
    Calculate the maximum radius for the given points.
    Returns (radius, min_pair_dist, min_boundary_dist).
    """
    # Boundary distances
    # Points are in [-0.5, 0.5]
    # Distance to nearest edge
    dx = np.minimum(0.5 - points[:, 0], 0.5 + points[:, 0])
    dy = np.minimum(0.5 - points[:, 1], 0.5 + points[:, 1])
    boundary_dists = np.minimum(dx, dy)
    min_boundary = np.min(boundary_dists)
    
    # Pairwise distances
    # Using cdist for speed
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    
    radius = min(min_pair / 2.0, min_boundary)
    return radius

def _hex_init():
    """
    Generate a hexagonal lattice initialization.
    """
    # Approximate grid size
    # Hex packing density is pi / (2 * sqrt(3)) ~ 0.9069
    # Square packing density is pi / 4 ~ 0.7854
    # For N=200, sqrt(N) ~ 14.14
    # Let's try to fit into a square.
    # Hex lattice:
    # Row height: s * sqrt(3)/2
    # Col width: s
    # Offset: s/2
    
    best_points = None
    best_r = 0
    
    # Try different grid configurations
    for n_rows in range(13, 18):
        for n_cols in range(13, 18):
            if n_rows * n_cols < N:
                continue
            if n_rows * n_cols > N + 10:
                continue
            
            # Calculate spacing to fit in [-0.5, 0.5]
            # We want to center the lattice.
            # Width = (n_cols - 1) * s_x + s_x (for offset?) 
            # Actually, for hex, x-coords are j*s, y-coords are i*s*sqrt(3)/2.
            # If we offset every other row by s/2, the width is still roughly (n_cols-1)*s + s/2?
            # Let's just generate points in [0, 1] and scale/shift to [-0.5, 0.5].
            
            s_x = 1.0 / (n_cols - 1) if n_cols > 1 else 1.0
            s_y = s_x * np.sqrt(3) / 2.0
            
            points = []
            for i in range(n_rows):
                for j in range(n_cols):
                    x = j * s_x
                    y = i * s_y
                    if i % 2 == 1:
                        x += s_x / 2.0
                    points.append([x, y])
            
            points = np.array(points)
            
            # Normalize to fit in [0, 1] x [0, 1]
            min_x, min_y = points.min(axis=0)
            max_x, max_y = points.max(axis=0)
            
            # Scale to fit in 1x1 box, preserving aspect ratio? 
            # Or just stretch to fit? Stretching distorts the hex shape.
            # Better: scale by max range to fit, then center.
            scale_x = 1.0 / (max_x - min_x) if max_x > min_x else 1.0
            scale_y = 1.0 / (max_y - min_y) if max_y > min_y else 1.0
            # To keep circles circular, we should scale x and y by the same factor.
            # The limiting factor is the larger range.
            scale = 1.0 / max(max_x - min_x, max_y - min_y)
            
            points = (points - (min_x, min_y)) * scale
            # Center in [-0.5, 0.5]
            # Current points are in [0, 1] roughly?
            # After scaling by 1/max_range, the max dimension is 1.
            # So points are in [0, 1] in the dominant dimension, and smaller in the other.
            # Center them.
            center = points.mean(axis=0)
            points = points - center
            # Now points are centered at 0, with max extent ~0.5.
            # Check if they fit in [-0.5, 0.5].
            # If the aspect ratio is bad, they might stick out.
            # Clip or adjust? 
            # For initialization, clipping is okay, or we can just use this as a start.
            # Better to ensure they are inside.
            points = np.clip(points, -0.49, 0.49)
            
            # Trim or pad to N
            if len(points) > N:
                # Remove furthest from center? Or random?
                # Random removal might be better to avoid bias.
                idx = np.random.choice(len(points), N, replace=False)
                points = points[idx]
            elif len(points) < N:
                # Pad with random points
                pad = np.random.uniform(-0.4, 0.4, size=(N - len(points), 2))
                points = np.vstack([points, pad])
            
            r = _calc_radius(points)
            if r > best_r:
                best_r = r
                best_points = points.copy()
                
    return best_points

def _simulated_annealing(points, max_time=100.0):
    """
    Run Simulated Annealing to maximize the minimum radius.
    """
    start_time = time.time()
    best_points = points.copy()
    best_radius = _calc_radius(points)
    
    # Temperature schedule
    T = 0.01
    cooling = 0.9999
    
    # Step size
    step_size = 0.05
    
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() - start_time < max_time:
        # Pick a random point
        i = np.random.randint(N)
        
        # Save original position
        orig_pos = best_points[i].copy()
        
        # Generate new position
        # Gaussian step
        new_pos = best_points[i] + np.random.normal(0, step_size, 2)
        
        # Ensure it stays within bounds? 
        # If it goes out, radius becomes 0 or negative, which is bad.
        # Better to reflect or clip? 
        # Clipping creates a "wall" effect. Reflection is smoother.
        # Or just let the radius calculation handle it (distance to boundary becomes negative? No, min(0.5-x, 0.5+x) is always positive if x in [-0.5, 0.5]. If x > 0.5, dist is negative? 
        # My _calc_radius uses min(0.5 - x, 0.5 + x). If x=0.6, min(-0.1, 1.1) = -0.1.
        # So radius becomes negative. This is a strong penalty. Good.
        
        best_points[i] = new_pos
        new_radius = _calc_radius(best_points)
        
        # Acceptance criteria
        if new_radius > best_radius or np.random.rand() < np.exp((new_radius - best_radius) / T):
            best_radius = new_radius
            no_improve = 0
        else:
            # Revert
            best_points[i] = orig_pos
            no_improve += 1
            
        # Cool down
        T *= cooling
        
        # If stuck, reheat or perturb
        if no_improve > max_no_improve:
            # Perturb best solution
            best_points = best_points + np.random.normal(0, 0.1, (N, 2))
            best_points = np.clip(best_points, -0.5, 0.5)
            best_radius = _calc_radius(best_points)
            T = 0.05
            no_improve = 0
            
    return best_points, best_radius

def construct():
    """
    Construct centers of N equal circles in unit square.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # 1. Initialization
    points = _hex_init()
    if points is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:N]
        
    current_radius = _calc_radius(points)
    
    # 2. Simulated Annealing
    # Allocate most of the time to SA
    sa_time = min(100.0, deadline - start_time - 10)
    if sa_time > 5:
        points, current_radius = _simulated_annealing(points, max_time=sa_time)
        
    # 3. Local Refinement (Optional)
    # Try a few steps of local search to polish
    # We can use a simple hill climbing with smaller steps
    if time.time() < deadline - 5:
        T = 0.001
        step_size = 0.01
        for _ in range(5000):
            if time.time() > deadline - 5:
                break
            i = np.random.randint(N)
            orig_pos = points[i].copy()
            new_pos = points[i] + np.random.normal(0, step_size, 2)
            points[i] = new_pos
            new_r = _calc_radius(points)
            if new_r > current_radius:
                current_radius = new_r
            else:
                points[i] = orig_pos
                
    # Final validation and clipping
    points = np.clip(points, -0.5, 0.5)
    
    return points
# EVOLVE-BLOCK-END
