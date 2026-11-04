import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    N = 11
    sqrt3 = np.sqrt(3)
    min_dist = sqrt3 + 1e-6  # Minimum center distance to avoid overlap
    
    # Initial configuration: Lattice cluster
    # Axial coordinates for a compact cluster of 11
    # Center (0,0) + 6 neighbors + 4 more
    coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2)
    ]
    
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        return [x, y]
        
    centers = np.array([axial_to_cartesian(q, r) for q, r in coords])
    angles = np.zeros(N)
    
    # Center the cluster
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    def get_bounding_hexagon_params(centers, angles):
        # Find minimal S and rotation alpha for outer hexagon
        # We assume outer hexagon is centered at the centroid of the inner hexagons.
        C = np.mean(centers, axis=0)
        best_S = float('inf')
        best_alpha = 0.0
        
        # Try rotations 0 to 60
        for alpha_deg in np.linspace(0, 60, 31):
            alpha = np.deg2rad(alpha_deg)
            normals = np.array([[np.cos(alpha + k * np.pi/3), np.sin(alpha + k * np.pi/3)] for k in range(6)])
            
            max_A = 0.0
            for k, n in enumerate(normals):
                # Calculate support function h(theta) for aligned unit hexagon
                theta = alpha + k * np.pi / 3
                # h(theta) = r_in / cos( (theta mod pi/3) - pi/6 )
                # r_in = sqrt(3)/2
                # Ensure cos is not zero (it won't be for hexagon normals)
                h_val = (sqrt3/2) / np.cos((theta % (np.pi/3)) - np.pi/6)
                
                # Project centers onto normal
                projs = (centers - C) @ n
                max_proj = np.max(projs)
                required_A = max_proj + h_val
                if required_A > max_A:
                    max_A = required_A
                    
            S = 2 * max_A / sqrt3
            if S < best_S:
                best_S = S
                best_alpha = alpha_deg
                
        return best_S, best_alpha, C
        
    # Check for overlaps in initial config
    def check_overlap(centers):
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    return True
        return False
        
    if check_overlap(centers):
        # Expand slightly to fix overlap
        for _ in range(10):
            if not check_overlap(centers):
                break
            centers *= 1.01
            
    # Local Search
    step_size = 0.1
    current_S, _, _ = get_bounding_hexagon_params(centers, angles)
    
    for i in range(500):
        if time.time() - start_time > timeout:
            break
            
        # Randomly perturb a few centers
        idxs = np.random.choice(N, 3, replace=False)
        old_centers = centers[idxs].copy()
        
        # Perturb
        noise = np.random.normal(0, step_size, (3, 2))
        centers[idxs] += noise
        
        # Check overlap
        if check_overlap(centers):
            centers[idxs] = old_centers
            step_size *= 0.9
            continue
            
        new_S, _, _ = get_bounding_hexagon_params(centers, angles)
        
        if new_S < current_S:
            current_S = new_S
            step_size *= 1.05
        else:
            centers[idxs] = old_centers
            step_size *= 0.95
            
    # Final calculation
    final_S, final_alpha, final_C = get_bounding_hexagon_params(centers, angles)
    
    # Construct return values
    inner = [[c[0], c[1], 0.0] for c in centers]
    outer_center = [final_C[0], final_C[1]]
    
    return inner, outer_center, final_S, final_alpha
