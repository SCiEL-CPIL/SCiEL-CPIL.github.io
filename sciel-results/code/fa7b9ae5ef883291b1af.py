import numpy as np
import time

def construct():
    # Strategy: Use an asymmetric 11-hexagon cluster to reduce the bounding box size.
    # The symmetric 13-hexagon cluster requires S=4.0.
    # An asymmetric 11-hexagon cluster can be shifted to reduce the span in the limiting dimension.
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    eps = 1e-5
    
    centers = []
    # Ring 0: center
    centers.append((0.0, 0.0))
    
    # Ring 1: 6 hexagons at distance sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        
    # Ring 2: 4 hexagons at distance 3 + 2*eps
    # Angles 30, 90, 150, 210 (Asymmetric selection to allow shifting)
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        
    # Calculate the bounding box of the cluster to estimate center and initial side
    # Inner hexagons are flat-top (angle 30 means flat top? 
    # Standard definition: angle 0 is pointy top (vertex at 0 deg).
    # Angle 30 is flat top (edge at 0 deg? No, usually rotation is around center).
    # Let's stick to the previous assumption: inner angle 30.
    # Outer angle 30.
    
    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    
    for x, y in centers:
        # Approximate extent for initial guess
        min_x = min(min_x, x - 1.5)
        max_x = max(max_x, x + 1.5)
        min_y = min(min_y, y - 1.5)
        max_y = max(max_y, y + 1.5)
        
    # Center the cluster to minimize the maximum distance from origin
    # The outer hexagon is centered at (0,0).
    # We can shift the inner cluster by (-center_x, -center_y)
    # to center its bounding box at the origin.
    
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    
    # Shifted centers
    shifted_centers = [(x - center_x, y - center_y) for x, y in centers]
    
    # Function to check if a point is inside a regular hexagon
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        # Translate point to hexagon center
        dx = px - cx
        dy = py - cy
        
        # Rotate point by -angle to align hexagon with axes (assuming angle 0 is pointy top)
        # If angle=0, vertices are at 0, 60, ...
        # If angle=30, vertices are at 30, 90, ... (flat top)
        # Let's use a general check: distance to origin in polar coords?
        # Simpler: Check if point is inside the 6 half-planes defined by the edges.
        
        rad = np.deg2rad(angle_deg)
        # Vertices of hexagon of side S centered at 0,0 with rotation rad
        # Vertex k: (S * cos(rad + k*pi/3), S * sin(rad + k*pi/3))
        
        # For a convex polygon, a point is inside if it is on the same side of each edge as the center.
        # We can define the 6 supporting lines.
        # Edge normal for edge k (between vertex k and k+1) is perpendicular to the edge.
        # Actually, simpler: The regular hexagon is the intersection of 6 half-planes.
        # The distance from center to each edge (apothem) is S * sqrt(3) / 2.
        # The normals to the edges are at angles: rad + 30 + k*60 ?
        # Let's verify.
        # If angle=0 (pointy top), vertices at 0, 60, 120, 180, 240, 300.
        # Edges connect (0,60), (60,120), etc.
        # Midpoint of edge (0,60) is at angle 30. Normal is at angle 30.
        # So normals are at 30, 90, 150, 210, 270, 330.
        # General formula: Normal angle = angle_deg + 30 + k*60.
        
        apothem = side * np.sqrt(3) / 2.0
        
        for k in range(6):
            normal_angle = np.deg2rad(angle_deg + 30 + k * 60)
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            
            # Project point onto normal
            proj = dx * nx + dy * ny
            
            # The point must be within apothem distance from center along each normal
            if proj > apothem + 1e-8:
                return False
            if proj < -apothem - 1e-8:
                return False
                
        return True

    # Function to check if all inner hexagons are inside the outer hexagon
    def is_valid_configuration(side, outer_angle, inner_angle):
        for cx, cy in shifted_centers:
            # Generate vertices of the inner hexagon
            for v in range(6):
                v_angle = np.deg2rad(inner_angle + v * 60)
                vx = cx + np.cos(v_angle)
                vy = cy + np.sin(v_angle)
                
                if not point_in_hexagon(vx, vy, 0.0, 0.0, side, outer_angle):
                    return False
        return True

    # Binary search for the minimum side length
    # Lower bound: based on bounding box of a single hexagon or just 1.0
    # Upper bound: based on bounding box of the cluster
    width = max_x - min_x
    height = max_y - min_y
    
    # For outer hexagon with angle 30 (flat top):
    # Width = sqrt(3) * S
    # Height = 2 * S
    # So S >= width/sqrt(3) and S >= height/2
    s_lower = max(width / np.sqrt(3), height / 2.0) * 0.8 # Start a bit lower to find exact
    s_upper = max(width / np.sqrt(3), height / 2.0) * 1.2
    
    # If lower bound is too low, it will fail. Let's start from a safe lower bound.
    # The minimum side length must be at least 1.0 (to contain one hexagon)
    s_lower = max(1.0, s_lower)
    
    # Ensure upper bound is valid
    while not is_valid_configuration(s_upper, 30.0, 30.0):
        s_upper *= 1.1
        if time.time() > deadline:
            break

    if time.time() > deadline:
        # Return a valid but potentially non-optimal result if we run out of time
        # We know s_upper is valid from the loop above (unless we broke before checking)
        # If we broke, we might not have a valid upper bound.
        # Let's just return a large side to be safe if we timed out during expansion
        if not is_valid_configuration(s_upper, 30.0, 30.0):
             s_upper = 10.0
        side = s_upper
    else:
        # Binary search
        for _ in range(50):
            if time.time() > deadline:
                break
            mid = (s_lower + s_upper) / 2.0
            if is_valid_configuration(mid, 30.0, 30.0):
                s_upper = mid
            else:
                s_lower = mid
                
        side = s_upper

    # Prepare output
    inner = []
    for x, y in shifted_centers:
        inner.append([x, y, 30.0])
        
    return inner, (0.0, 0.0), side, 30.0
