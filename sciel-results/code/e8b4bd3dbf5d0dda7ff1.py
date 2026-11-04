import numpy as np
import time

N = 11

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Define the 11 hexagon centers
    # 1 center
    # 6 neighbors at distance sqrt(3)
    # 4 second-ring points at distance 3 (angles 30, 90, 150, 270)
    
    s3 = np.sqrt(3)
    
    centers = [
        (0.0, 0.0),
        (s3, 0.0),
        (s3/2, 1.5),
        (-s3/2, 1.5),
        (-s3, 0.0),
        (-s3/2, -1.5),
        (s3/2, -1.5),
        (3*s3/2, 1.5),
        (0.0, 3.0),
        (-3*s3/2, 1.5),
        (0.0, -3.0)
    ]
    
    centers = np.array(centers)
    
    # Facet normals of the outer hexagon (fixed)
    # Angles: 30, 90, 150, 210, 270, 330 degrees
    angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    def cost(theta):
        # Rotate centers by theta
        c = np.cos(theta)
        s = np.sin(theta)
        rot_matrix = np.array([[c, -s], [s, c]])
        
        rotated_centers = centers @ rot_matrix.T
        
        # Project onto normals
        # projections shape: (11, 6)
        projections = rotated_centers @ normals.T
        
        # Max projection in each direction
        max_projections = np.max(projections, axis=0)
        
        # The cost is the maximum of these max projections
        # This value is the required apothem
        return np.max(max_projections)
    
    # Search for optimal theta
    # The function is periodic with 60 degrees (pi/3)
    # We can search in [0, pi/3]
    
    best_theta = 0.0
    best_cost = cost(0.0)
    
    # Coarse grid search
    num_steps = 100
    thetas = np.linspace(0, np.pi/3, num_steps)
    
    for t in thetas:
        c = cost(t)
        if c < best_cost:
            best_cost = c
            best_theta = t
            
    if time.time() > deadline:
        pass # Use best found
    
    # Fine search around best_theta
    if time.time() < deadline:
        delta = np.pi / (3 * num_steps)
        fine_thetas = np.linspace(best_theta - delta, best_theta + delta, 100)
        for t in fine_thetas:
            c = cost(t)
            if c < best_cost:
                best_cost = c
                best_theta = t
                
    # Calculate side length
    # Apothem A = best_cost
    # Side S = 2 * A / sqrt(3)
    side_length = 2.0 * best_cost / np.sqrt(3)
    
    # Apply rotation to centers for output
    c = np.cos(best_theta)
    s = np.sin(best_theta)
    rot_matrix = np.array([[c, -s], [s, c]])
    final_centers = centers @ rot_matrix.T
    
    # Format output
    # inner: list of [x, y, angle_degrees]
    # The angle of each inner hexagon is 0 (aligned with lattice)
    # The outer hexagon angle is 0 (we rotated the inner cluster)
    
    inner = []
    for x, y in final_centers:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), side_length, 0.0
