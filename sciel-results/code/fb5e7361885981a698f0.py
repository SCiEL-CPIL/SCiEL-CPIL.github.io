import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin for 150s limit
    
    # Define centers of 11 unit hexagons
    # 1 center
    centers = [(0.0, 0.0)]
    
    # 6 neighbors (Ring 1)
    # Axial coords (q, r) to Cartesian (flat-top)
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * q)
    ring1_axial = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    for q, r in ring1_axial:
        x = 1.5 * q
        y = np.sqrt(3) * (r + 0.5 * q)
        centers.append((x, y))
        
    # 4 more (Ring 2 subset)
    # Let's try (2,0), (0,2), (-2,0), (0,-2)
    ring2_axial = [(2, 0), (0, 2), (-2, 0), (0, -2)]
    for q, r in ring2_axial:
        x = 1.5 * q
        y = np.sqrt(3) * (r + 0.5 * q)
        centers.append((x, y))
        
    # Ensure we have exactly N centers
    if len(centers) > N:
        centers = centers[:N]
    elif len(centers) < N:
        # Fallback: add more from ring 2
        more_ring2 = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        for q, r in more_ring2:
            if len(centers) >= N: break
            x = 1.5 * q
            y = np.sqrt(3) * (r + 0.5 * q)
            centers.append((x, y))

    # Generate vertices for all inner hexagons
    # Unit hexagon vertices relative to center (flat-top, angle 0)
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    rel_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    all_verts = []
    for cx, cy in centers:
        verts = rel_verts + np.array([cx, cy])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Find minimal outer hexagon side L
    # Outer hexagon is regular, centered at (0,0)
    # We can rotate the outer hexagon by angle alpha
    # The constraints for a point (x,y) to be inside a regular hexagon of side L, rotated by alpha:
    # The support function in direction theta is L * cos(theta - alpha - pi/6) * 2/sqrt(3)?
    # Simpler: The 6 normal directions of the outer hexagon are alpha + k*pi/3.
    # The distance from center to each side is apothem = L * sqrt(3) / 2.
    # So for each normal direction n_k, max(dot(verts, n_k)) <= apothem.
    # L >= 2 * max(dot(verts, n_k)) / sqrt(3)
    
    best_L = float('inf')
    best_alpha = 0.0
    
    # Search alpha from 0 to 30 degrees (pi/6)
    # Due to symmetry, 0 to 30 is enough?
    # The inner shape has 6-fold symmetry? 
    # Our chosen 11 centers: 
    # (0,0)
    # Ring 1: 6-fold symmetric
    # Ring 2 subset: (2,0), (0,2), (-2,0), (0,-2). This is 4-fold symmetric (square), not 6-fold.
    # So we should search 0 to 360, or at least 0 to 90.
    # Let's search 0 to 60 degrees with fine steps.
    
    for alpha_deg in np.linspace(0, 60, 121):
        if time.time() > deadline:
            break
        alpha = np.radians(alpha_deg)
        # Normals are at alpha + k*pi/3
        max_proj = 0
        for k in range(6):
            theta = alpha + k * np.pi / 3
            n = np.array([np.cos(theta), np.sin(theta)])
            projs = all_verts @ n
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        
        L = 2 * max_proj / np.sqrt(3)
        if L < best_L:
            best_L = L
            best_alpha = alpha_deg
            
    # Convert centers to array of (x, y, angle)
    # Angle is 0.0 for all inner hexagons
    inner_array = np.array([(cx, cy, 0.0) for cx, cy in centers])
    
    return inner_array, (0.0, 0.0), best_L, best_alpha
