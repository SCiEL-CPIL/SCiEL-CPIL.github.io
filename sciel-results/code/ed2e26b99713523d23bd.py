import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, minimizing the side length of the outer hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Deadline guard
    start_time = time.time()
    timeout = 590  # Safety margin for 600s limit
    
    n = 12
    
    # Axes for separation check (normals to edges of unit hexagon with angle 0)
    u1 = np.array([np.sqrt(3)/2, 0.5])
    u2 = np.array([0, 1])
    u3 = np.array([-np.sqrt(3)/2, 0.5])
    
    axes = np.array([u1, u2, u3])
    # The width of a unit hexagon along these axes is sqrt(3)
    min_dist = np.sqrt(3)
    # We need separation > 1e-6, so we target a slightly larger distance in optimization
    target_dist = min_dist + 1e-4
    
    # Objective function
    def objective(x):
        # x is flat array of 24 coords: [x0, y0, x1, y1, ...]
        centers = x.reshape((n, 2))
        
        # Check for intersections first (Hard Constraint)
        # For aligned unit hexagons, they intersect if max projection on 3 axes <= sqrt(3)
        # We require separation > 1e-6, so we need max_proj > sqrt(3) + 1e-6
        min_allowed_proj = np.sqrt(3) + 1e-5
        
        for i in range(n):
            for j in range(i+1, n):
                diff = centers[i] - centers[j]
                projs = np.abs(diff @ axes.T)
                max_proj = np.max(projs)
                if max_proj <= min_allowed_proj:
                    # Return a large value to reject this configuration
                    return 1e10
        
        # Containment axes for outer hexagon (angle 0)
        # Normals to the edges of the outer hexagon (and inner hexagons)
        # Edge normals for angle 0 hexagon are at 30, 90, 150 degrees
        n1 = np.array([np.sqrt(3)/2, 0.5])
        n2 = np.array([0, 1])
        n3 = np.array([-np.sqrt(3)/2, 0.5])
        
        c_n1 = centers @ n1
        c_n2 = centers @ n2
        c_n3 = centers @ n3
        
        # Required side length s must satisfy:
        # max_proj * (2/sqrt(3)) + 1 <= s
        # Because apothem_outer = s * sqrt(3)/2
        # max_proj_inner + apothem_inner <= apothem_outer
        # max_proj_center + sqrt(3)/2 <= s * sqrt(3)/2
        # max_proj_center <= (s-1) * sqrt(3)/2
        # s >= 1 + (2/sqrt(3)) * max_proj_center
        
        scale = 2.0 / np.sqrt(3)
        
        s1 = scale * max(np.max(c_n1), -np.min(c_n1)) + 1
        s2 = scale * max(np.max(c_n2), -np.min(c_n2)) + 1
        s3 = scale * max(np.max(c_n3), -np.min(c_n3)) + 1
        
        s = max(s1, s2, s3)
        
        return s

    # Initial guess: Honeycomb 3x4 grid
    # e1 = (1.5, sqrt(3)/2), e2 = (0, sqrt(3))
    e1 = np.array([1.5, np.sqrt(3)/2])
    e2 = np.array([0, np.sqrt(3)])
    
    centers_init = []
    for i in range(3):
        for j in range(4):
            p = i * e1 + j * e2
            centers_init.append(p)
    
    centers_init = np.array(centers_init)
    # Center the grid
    centroid = np.mean(centers_init, axis=0)
    centers_init -= centroid
    
    # Expand slightly to ensure strict separation > 1e-5
    # Scale by 1.001
    centers_init *= 1.001
    
    x0 = centers_init.flatten()
    
    # Optimization
    best_x = x0
    best_f = objective(x0)
    
    # Nelder-Mead
    # We have ~590s.
    # Let's do a few restarts if time permits.
    
    while time.time() - start_time < timeout - 10:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
        
        # Check if we are out of time
        if time.time() - start_time > timeout - 5:
            break
            
        # Perturb for restart
        # Only if we haven't converged to a very good solution
        if best_f > 4.5: # Threshold for s ~ 3.464 + penalty
            x0 = best_x + np.random.normal(0, 0.2, size=24)
        else:
            break
            
    # Extract best result
    centers = best_x.reshape((n, 2))
    
    # Verify separation
    # If separation is not guaranteed, we might need to do a final check
    # But the penalty should have handled it.
    
    # Calculate final s
    n1 = np.array([np.sqrt(3)/2, 0.5])
    n2 = np.array([0, 1])
    n3 = np.array([-np.sqrt(3)/2, 0.5])
    
    c_n1 = centers @ n1
    c_n2 = centers @ n2
    c_n3 = centers @ n3
    
    scale = 2.0 / np.sqrt(3)
    
    s1 = scale * max(np.max(c_n1), -np.min(c_n1)) + 1
    s2 = scale * max(np.max(c_n2), -np.min(c_n2)) + 1
    s3 = scale * max(np.max(c_n3), -np.min(c_n3)) + 1
    s = max(s1, s2, s3)
    
    # Add a safety margin to ensure strict containment
    # This prevents floating point errors from causing "outside" verdicts
    s = s + 0.01
    
    # Construct return arrays
    inner_hex_data = np.zeros((n, 3))
    for i in range(n):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0  # Angle 0
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s
