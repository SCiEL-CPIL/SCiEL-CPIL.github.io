import numpy as np
import time

N = 11

def construct():
    # Generate 11 hexagons in a hexagonal lattice
    # Axial coordinates for 11 hexes
    # Center (0,0)
    # 6 neighbors
    # 4 from the next ring
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, -1), (2, -2), (1, -2), (0, -2) # 4 more
    ]
    
    # Convert axial to cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    points = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        points.append([x, y])
    
    points = np.array(points)
    
    # The inradius of a unit hexagon is sqrt(3)/2
    r_in = np.sqrt(3) / 2.0
    
    best_R = float('inf')
    best_theta = 0.0
    best_center = (0.0, 0.0)
    
    # The cluster is centered at (0,0) in our coordinate system.
    # We can rotate the cluster (or equivalently the container) to minimize R.
    # Since the cluster is not perfectly symmetric, we should check multiple angles.
    
    # We also need to ensure the hexagons don't overlap.
    # In the hex lattice, they are tangent, so distance is exactly sqrt(3).
    # This is valid.
    
    # Optimize container angle
    # The container hexagon has normals at angles theta, theta+60, ...
    # We check theta from 0 to 60 degrees
    
    start_time = time.time()
    timeout = 110 # Safety margin
    
    # Coarse search
    thetas = np.linspace(0, np.pi/3, 600)
    
    for theta in thetas:
        if time.time() - start_time > timeout:
            break
            
        # Normals of the container
        # The sides of the hexagon are perpendicular to these vectors
        # Normal vectors point from center to side midpoints
        angles = theta + np.arange(6) * (np.pi / 3)
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Calculate max projection
        # proj[i, j] = points[i] dot normals[j]
        projs = points @ normals.T
        
        # Max projection for each normal
        max_projs = np.max(projs, axis=0)
        
        # Constraint: max_proj + r_in <= R * sqrt(3)/2
        # R >= (max_proj + r_in) * 2 / sqrt(3)
        required_R = np.max(max_projs + r_in) * 2.0 / np.sqrt(3)
        
        if required_R < best_R:
            best_R = required_R
            best_theta = theta
            
    # Fine search around best_theta
    if time.time() - start_time < timeout:
        thetas_fine = np.linspace(best_theta - 0.01, best_theta + 0.01, 100)
        for theta in thetas_fine:
            if time.time() - start_time > timeout:
                break
            angles = theta + np.arange(6) * (np.pi / 3)
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            projs = points @ normals.T
            max_projs = np.max(projs, axis=0)
            required_R = np.max(max_projs + r_in) * 2.0 / np.sqrt(3)
            if required_R < best_R:
                best_R = required_R
                best_theta = theta

    # Construct the output
    # Inner hexagons: (x, y, angle_degrees)
    # We keep them aligned with the lattice (angle 0)
    # The container is rotated by best_theta
    inner = []
    for p in points:
        inner.append([p[0], p[1], 0.0])
        
    center = (0.0, 0.0)
    side = best_R
    angle_deg = np.degrees(best_theta)
    
    return inner, center, side, angle_deg

# EVOLVE-BLOCK-START
def construct():
    # Reusing the logic above but wrapped in the required function
    # The logic is deterministic and fast.
    
    # Generate 11 hexagons in a hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, -1), (2, -2), (1, -2), (0, -2)
    ]
    
    points = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        points.append([x, y])
    
    points = np.array(points)
    r_in = np.sqrt(3) / 2.0
    
    best_R = float('inf')
    best_theta = 0.0
    
    # Coarse search
    thetas = np.linspace(0, np.pi/3, 360)
    
    for theta in thetas:
        angles = theta + np.arange(6) * (np.pi / 3)
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        projs = points @ normals.T
        max_projs = np.max(projs, axis=0)
        required_R = np.max(max_projs + r_in) * 2.0 / np.sqrt(3)
        
        if required_R < best_R:
            best_R = required_R
            best_theta = theta

    # Fine search
    thetas_fine = np.linspace(best_theta - 0.005, best_theta + 0.005, 50)
    for theta in thetas_fine:
        angles = theta + np.arange(6) * (np.pi / 3)
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        projs = points @ normals.T
        max_projs = np.max(projs, axis=0)
        required_R = np.max(max_projs + r_in) * 2.0 / np.sqrt(3)
        if required_R < best_R:
            best_R = required_R
            best_theta = theta

    inner = [[p[0], p[1], 0.0] for p in points]
    center = (0.0, 0.0)
    side = best_R
    angle_deg = np.degrees(best_theta)
    
    return inner, center, side, angle_deg
# EVOLVE-BLOCK-END
