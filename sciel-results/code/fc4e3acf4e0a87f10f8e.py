import math
import itertools
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0
    
    # Hexagon geometry constants
    # Side length of inner hexagons = 1.0
    # Distance between centers of adjacent hexagons
    d = math.sqrt(3)
    
    # Generate centers for 1, 6, 12 rings
    centers = [(0.0, 0.0)]
    
    # Ring 1: 6 neighbors
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers.append((x, y))
        
    # Ring 2: 12 neighbors
    # Positions are at distance 2*d from center? No.
    # In a hex grid, ring 2 has 12 hexagons.
    # They are located at (2,0), (1,1), (0,2), etc in axial coords.
    # Let's generate them properly.
    # Axial coords (q, r).
    # Center (0,0).
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1) ? 
    # Standard axial to Cartesian:
    # x = d * (q + r/2)
    # y = d * (sqrt(3)/2) * r
    
    ring2_centers = []
    # Iterate q, r such that max(|q|, |r|, |q+r|) == 2
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                # Cartesian
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_centers.append((x, y))
                
    # ring2_centers should have 12 items
    # Verify
    if len(ring2_centers) != 12:
        # Fallback if generation is wrong, but logic is standard
        pass
        
    best_S = float('inf')
    best_config = None
    best_theta = 0.0
    
    # We need to choose 4 from 12 ring2 centers
    # To save time, we might not need ALL 495 if we time out
    # But 495 is small.
    # Sweeping theta: 0 to 60 deg. 600 steps (0.1 deg) is fast.
    
    all_combinations = list(itertools.combinations(range(12), 4))
    
    for combo in all_combinations:
        if time.time() > deadline - safety_margin:
            break
            
        # Construct the 11 centers
        current_centers = centers[:7] # Center + Ring 1
        for idx in combo:
            current_centers.append(ring2_centers[idx])
            
        # Get all vertices for these 11 hexagons
        # Hexagon vertices relative to center (cx, cy):
        # (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        # Note: This assumes "flat-top" or "pointy-top"?
        # If vertices are at 0, 60, 120..., it's pointy-top (vertex at 0).
        # Let's stick to this orientation for inner hexagons.
        verts_rel = []
        for i in range(6):
            angle = i * math.pi / 3
            vx = math.cos(angle)
            vy = math.sin(angle)
            verts_rel.append((vx, vy))
            
        points = []
        for cx, cy in current_centers:
            for vx, vy in verts_rel:
                points.append((cx + vx, cy + vy))
                
        # Find min S
        # Sweep theta for container
        # Container normals at theta + k*pi/3
        # Condition: max |P . n| <= S * sqrt(3)/2
        # S = 2/sqrt(3) * max |P . n|
        
        min_S_for_combo = float('inf')
        min_theta = 0.0
        
        # Optimization: The function is periodic with 60 deg.
        # We can sample theta.
        # Use vectorized numpy for speed.
        
        # Convert points to numpy array
        points_arr = np.array(points)  # shape (66, 2)
        
        # Precompute normal vectors for all 6 directions at each theta step
        # theta values: 0 to pi/3, 1000 steps
        theta_steps = np.linspace(0, math.pi / 3, 1000, endpoint=False)
        
        # For each theta, compute 6 normal vectors
        # Shape: (1000, 6, 2)
        k_angles = np.arange(6) * math.pi / 3  # shape (6,)
        all_angles = theta_steps[:, np.newaxis] + k_angles[np.newaxis, :]  # shape (1000, 6)
        normals = np.stack([np.cos(all_angles), np.sin(all_angles)], axis=-1)  # shape (1000, 6, 2)
        
        # Compute projections: points_arr @ normals.T
        # points_arr: (66, 2), normals: (1000, 6, 2)
        # We need for each theta (1000) and each normal (6): max over points (66) of |dot|
        # Reshape normals to (1000, 6, 1, 2) and points to (1, 1, 66, 2)
        normals_reshaped = normals[:, :, np.newaxis, :]  # (1000, 6, 1, 2)
        points_reshaped = points_arr[np.newaxis, np.newaxis, :, :]  # (1, 1, 66, 2)
        
        # Dot product: (1000, 6, 66)
        dots = np.sum(normals_reshaped * points_reshaped, axis=-1)  # (1000, 6, 66)
        abs_dots = np.abs(dots)  # (1000, 6, 66)
        
        # Max over points for each theta and normal direction
        max_per_normal = np.max(abs_dots, axis=-1)  # (1000, 6)
        
        # Max over normals for each theta
        max_proj_per_theta = np.max(max_per_normal, axis=-1)  # (1000,)
        
        # S for each theta
        S_per_theta = (2.0 / math.sqrt(3)) * max_proj_per_theta  # (1000,)
        
        # Find best theta
        best_idx = np.argmin(S_per_theta)
        min_S_for_combo = S_per_theta[best_idx]
        min_theta = theta_steps[best_idx]
                
        if min_S_for_combo < best_S:
            best_S = min_S_for_combo
            best_config = current_centers
            best_theta = min_theta

    # If no valid config found (shouldn't happen), fallback
    if best_config is None:
        # Just return a large valid config
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Refinement pass: fine-tune the best theta with higher resolution
    if time.time() < deadline - safety_margin:
        # Do a finer sweep around best_theta
        # Current step size was (pi/3)/1000
        # Do 1000 steps in a window of 2 * step_size around best_theta
        step_size = (math.pi / 3) / 1000.0
        window = 2 * step_size
        fine_theta_steps = np.linspace(best_theta - window, best_theta + window, 1000, endpoint=False)
        
        # Clip to valid range [0, pi/3]
        fine_theta_steps = fine_theta_steps[(fine_theta_steps >= 0) & (fine_theta_steps <= math.pi/3)]
        
        if len(fine_theta_steps) > 0:
            # Reconstruct points for best_config
            verts_rel = []
            for i in range(6):
                angle = i * math.pi / 3
                vx = math.cos(angle)
                vy = math.sin(angle)
                verts_rel.append((vx, vy))
            
            points_list = []
            for cx, cy in best_config:
                for vx, vy in verts_rel:
                    points_list.append((cx + vx, cy + vy))
            
            points_arr = np.array(points_list)
            
            k_angles = np.arange(6) * math.pi / 3
            all_angles = fine_theta_steps[:, np.newaxis] + k_angles[np.newaxis, :]
            normals = np.stack([np.cos(all_angles), np.sin(all_angles)], axis=-1)
            
            normals_reshaped = normals[:, :, np.newaxis, :]
            points_reshaped = points_arr[np.newaxis, np.newaxis, :, :]
            
            dots = np.sum(normals_reshaped * points_reshaped, axis=-1)
            abs_dots = np.abs(dots)
            max_per_normal = np.max(abs_dots, axis=-1)
            max_proj_per_theta = np.max(max_per_normal, axis=-1)
            S_per_theta = (2.0 / math.sqrt(3)) * max_proj_per_theta
            
            fine_best_idx = np.argmin(S_per_theta)
            if S_per_theta[fine_best_idx] < best_S:
                best_S = S_per_theta[fine_best_idx]
                best_theta = fine_theta_steps[fine_best_idx]

    # Format output
    # inner: list of [x, y, angle_degrees]
    # Our inner hexagons are all rotated 0 degrees relative to the grid.
    # The container is rotated by best_theta.
    # The task asks for inner hexagon angle.
    # We can keep inner angle 0.0.
    # Container rotation is best_theta (radians). Convert to degrees.
    
    inner_rows = []
    for cx, cy in best_config:
        inner_rows.append([cx, cy, 0.0])
        
    center = (0.0, 0.0)
    side = best_S
    angle_deg = math.degrees(best_theta)
    
    return inner_rows, center, side, angle_deg
# EVOLVE-BLOCK-END
