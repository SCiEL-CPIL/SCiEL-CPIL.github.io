import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # 150s limit, keep 10s margin
    
    # 1. Generate hexagonal lattice centers
    # Unit hexagon side length = 1.
    # Distance between centers of adjacent hexagons = sqrt(3)
    d = math.sqrt(3)
    
    centers = []
    # Generate enough points to cover N=12
    # We need radius in lattice steps. 
    # N=12 fits in radius 2 (7+12=19 > 12)
    max_r = 3
    for q in range(-max_r, max_r + 1):
        for r in range(-max_r, max_r + 1):
            # Axial to Cartesian (pointy-top hexagons, but we define vertices later)
            # Standard mapping for hex grid with horizontal rows:
            # x = d * (q + r/2.0)
            # y = d * (r * math.sqrt(3) / 2.0)
            # Wait, let's use a standard consistent system.
            # Let's use the system where neighbors are at 60 deg intervals.
            # x = d * (q + r/2)
            # y = d * (r * sqrt(3)/2)
            x = d * (q + r / 2.0)
            y = d * (r * math.sqrt(3) / 2.0)
            centers.append((x, y))
            
    # Sort by distance from origin
    centers.sort(key=lambda c: c[0]**2 + c[1]**2)
    
    # Take first N
    selected_centers = centers[:N]
    
    # 2. Calculate centroid of selected centers
    cx = sum(c[0] for c in selected_centers) / N
    cy = sum(c[1] for c in selected_centers) / N
    
    # 3. Generate vertices for all selected hexagons
    # Unit hexagon vertices relative to center (angle 0)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    hex_verts_rel = [(math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)) for i in range(6)]
    
    all_points = []
    for (hx, hy) in selected_centers:
        for (vx, vy) in hex_verts_rel:
            all_points.append((hx + vx, hy + vy))
    
    all_points = np.array(all_points)
    
    # 4. Find minimal enclosing regular hexagon
    # We check rotations from 0 to 30 degrees.
    # Due to symmetry of the lattice and container, checking 0 to 30 is sufficient.
    best_S = float('inf')
    best_angle = 0.0
    best_center = (cx, cy)
    
    # We can also optimize the center slightly, but for a symmetric cluster, centroid is good.
    # To be safe, we'll just use the centroid.
    
    # Optimization: The cluster is small, so we can check a fine grid of angles.
    # However, the minimum is likely at 0 or 30 or a specific symmetry axis.
    # Let's check 0 to 30 in steps of 0.1 degrees for safety, it's fast enough.
    for angle_deg in np.arange(0, 30.1, 0.1):
        theta = math.radians(angle_deg)
        # Normals of the 6 sides of the outer hexagon
        # Side normals are at angles theta, theta+60, ..., theta+300
        # Distance from center to side is A = S * sqrt(3) / 2
        # Condition: max projection of (p - C) onto normal <= A
        # So S >= 2/sqrt(3) * max projection
        
        max_proj = 0
        for k in range(6):
            n_angle = theta + k * math.pi / 3
            nx = math.cos(n_angle)
            ny = math.sin(n_angle)
            # Project all points onto this normal
            # p - C
            dx = all_points[:, 0] - cx
            dy = all_points[:, 1] - cy
            proj = dx * nx + dy * ny
            # We need the max absolute projection? 
            # No, the hexagon is centered at C. The condition is that all points are within distance A from the center along the normal.
            # i.e. -A <= proj <= A.
            # So max |proj| <= A.
            max_abs_proj = np.max(np.abs(proj))
            if max_abs_proj > max_proj:
                max_proj = max_abs_proj
        
        S = (2.0 / math.sqrt(3)) * max_proj
        if S < best_S:
            best_S = S
            best_angle = angle_deg
            
    # The inner hexagons are aligned with angle 0 in our generation.
    # The outer hexagon is rotated by best_angle.
    
    # Construct the return value
    # inner: list of [x, y, angle_degrees]
    inner = []
    for (hx, hy) in selected_centers:
        inner.append([hx, hy, 0.0])
        
    outer_center = [cx, cy]
    outer_side = best_S
    outer_angle = best_angle
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
