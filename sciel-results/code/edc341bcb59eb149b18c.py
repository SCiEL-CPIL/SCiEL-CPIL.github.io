import math
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate hex grid coordinates (axial) for radius 2
    # Axial to Cartesian (flat-top hexagons, side 1)
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return x, y

    # Get all coordinates in radius 2
    coords = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Check if within hex radius 2
            # distance is max(|q|, |r|, |q+r|)
            if max(abs(q), abs(r), abs(q + r)) <= 2:
                coords.append((q, r))
                
    # There are 19 such coordinates
    # We need to choose 12
    # Convert to Cartesian array
    cart_coords = np.array([axial_to_cartesian(q, r) for q, r in coords])
    
    best_score = float('inf')
    best_indices = None
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    # Pre-compute normals for various outer hexagon angles
    # The width in direction n is max(n . P) - min(n . P)
    # S = max_width / sqrt(3)
    # We want to minimize S over angle theta.
    # The normals for a hexagon with angle theta are at theta + 30, theta + 90, theta + 150 (and opposites)
    # Actually, the 3 independent directions are separated by 60 degrees.
    # Let's just sample angles.
    
    # To speed up, we can limit the search space or use a smarter approach.
    # But 50k * 5 angles is fine.
    
    angles_to_try = [0, 15, 30, 45, 60]
    
    for i in range(len(coords)):
        if time.time() > deadline:
            break
            
        # Use itertools.combinations for the rest
        # This might be slow if done in pure Python loops.
        # Let's try to be efficient.
        
        # Actually, let's just iterate all combinations
        # 19 choose 12 is 50388.
        # For each, we calculate the score.
        
        # We can vectorize the width calculation for a given set of points.
        
        for combo in combinations(range(i+1, len(coords)), 11-i):
            # Construct the full index list
            idx_list = list(range(i)) + [i] + list(combo)
            # Wait, the logic above is wrong for generating combinations.
            # Use standard combinations.
            pass

    # Correct loop for combinations
    indices = range(len(coords))
    
    for combo in combinations(indices, 12):
        if time.time() > deadline:
            break
            
        points = cart_coords[list(combo)]
        
        # Calculate centroid to center the points?
        # The formula S = max_width / sqrt(3) assumes we can shift the center.
        # Yes, the minimal enclosing hexagon of a fixed orientation is centered at the midpoint of the widths.
        # So we don't need to find the center explicitly for the score, just the width.
        
        min_s_for_this_combo = float('inf')
        best_theta_for_combo = 0
        
        for theta_deg in angles_to_try:
            theta = math.radians(theta_deg)
            # Normals at theta + 30, theta + 90, theta + 150
            # Actually, the edge normals of a flat-top hexagon (angle 0) are at 90, 150, 210...
            # Let's define the 3 directions.
            # Direction 1: theta + pi/6
            # Direction 2: theta + pi/6 + pi/3
            # Direction 3: theta + pi/6 + 2*pi/3
            
            max_width = 0.0
            for k in range(3):
                angle = theta + math.pi/6 + k * math.pi/3
                nx = math.cos(angle)
                ny = math.sin(angle)
                
                # Project points onto this normal
                projections = points[:, 0] * nx + points[:, 1] * ny
                width = np.max(projections) - np.min(projections)
                if width > max_width:
                    max_width = width
            
            s = max_width / sqrt3
            if s < min_s_for_this_combo:
                min_s_for_this_combo = s
                best_theta_for_combo = theta_deg
        
        if min_s_for_this_combo < best_score:
            best_score = min_s_for_this_combo
            best_indices = list(combo)
            best_angle = best_theta_for_combo
            
    if best_indices is None:
        # Fallback
        return [], (0,0), 10.0, 0.0
        
    # Reconstruct the best solution
    best_points = cart_coords[best_indices]
    
    # Calculate the optimal center for this configuration and angle
    # The center should be the midpoint of the bounding box in the hexagonal frame.
    # Or simply the centroid of the points? No, the midpoint of the widths.
    theta = math.radians(best_angle)
    
    # Find the center C such that the max projection is minimized.
    # For a fixed orientation, the optimal center is the intersection of the midlines of the widths.
    # Let's just compute the min and max projections for the 3 axes and take the midpoints.
    
    centers = []
    for k in range(3):
        angle = theta + math.pi/6 + k * math.pi/3
        nx = math.cos(angle)
        ny = math.sin(angle)
        projections = best_points[:, 0] * nx + best_points[:, 1] * ny
        min_p = np.min(projections)
        max_p = np.max(projections)
        mid_p = (min_p + max_p) / 2.0
        # The center C must satisfy C . n = mid_p
        # This is a system of 3 equations (but only 2 independent).
        # We can solve for C using least squares.
        centers.append((nx, ny, mid_p))
        
    # Solve A C = b
    A = np.array([centers[0][:2], centers[1][:2]])
    b = np.array([centers[0][2], centers[1][2]])
    
    try:
        C = np.linalg.lstsq(A, b, rcond=None)[0]
    except:
        C = np.mean(best_points, axis=0)
        
    # Add a small margin for safety
    side = best_score * 1.001
    
    # Format output
    inner = []
    for p in best_points:
        inner.append((p[0], p[1], 0.0)) # Angle 0 for inner hexagons
        
    return inner, (C[0], C[1]), side, best_angle

