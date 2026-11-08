"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Place 12 unit hexagons in a compact hexagonal cluster
    # Using axial coordinates converted to Cartesian
    # For flat-top hexagons, center-to-center distance = sqrt(3)
    
    # Define 12 positions in a compact arrangement
    # Using a "hexagonal" cluster: center + 6 ring + 5 more
    # Let's use a 3-4-3-2 row arrangement for compactness
    
    sqrt3 = np.sqrt(3.0)
    
    # Row-based arrangement (flat-top hexagons)
    # Each row offset by sqrt(3)/2 vertically, and by 1.5 horizontally for alternating rows
    # Actually for flat-top, adjacent rows are offset by sqrt(3)/2 in y and 1.5 in x? 
    # Let me use the standard hex grid:
    # For flat-top: 
    #   col spacing = 3/2 * s = 1.5 (for s=1)
    #   row spacing = sqrt(3) * s = sqrt(3)
    #   odd columns offset by sqrt(3)/2 in y
    
    # Let me use pointy-top for simplicity:
    # Pointy-top: width = sqrt(3), height = 2
    # Row spacing = 3/2 = 1.5
    # Col spacing = sqrt(3)
    # Odd rows offset by sqrt(3)/2 in x
    
    # 12 hexagons in a 4-3-4-1 or similar arrangement
    # Let's try: rows of 4, 3, 4, 1 = 12? That's asymmetric.
    # Better: 3, 4, 3, 2 = 12 or 4, 4, 4 = 12
    
    # Let me use 3 rows of 4:
    # Row 0 (y=0): x = -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3
    # Row 1 (y=1.5): x = -sqrt3, 0, sqrt3, 2*sqrt3  (offset by sqrt3/2)
    # Row 2 (y=3.0): x = -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3
    
    # This gives a parallelogram shape. Let me compute the bounding hexagon.
    
    positions = []
    # Row 0
    for i in range(4):
        x = (i - 1.5) * sqrt3
        y = 0.0
        positions.append((x, y, 0.0))
    # Row 1 (offset by sqrt3/2 in x)
    for i in range(4):
        x = (i - 1.0) * sqrt3
        y = 1.5
        positions.append((x, y, 0.0))
    # Row 2
    for i in range(4):
        x = (i - 1.5) * sqrt3
        y = 3.0
        positions.append((x, y, 0.0))
    
    inner = [[x, y, a] for x, y, a in positions]
    
    # Now find the minimal regular hexagon (center, side, rotation) that contains all 12
    # A regular hexagon of side S, center (cx, cy), rotation theta contains point (px, py)
    # if for all 6 sides, the distance from center to the side (in the inward normal direction) >= S*sqrt(3)/2
    # The 6 side normals are at angles theta + k*pi/3 for k=0..5
    # For each normal direction, the projection of (px-cx, py-cy) onto the normal must be <= S*sqrt(3)/2
    
    # We need to find (cx, cy, S, theta) minimizing S such that all points are inside.
    # This is a convex optimization problem. Let's use a grid search over theta and then
    # for each theta, find the optimal (cx, cy) and S.
    
    # For a fixed theta, the constraint for point i is:
    #   dot((px_i - cx, py_i - cy), n_k) <= S * sqrt(3)/2 for all k=0..5
    # where n_k = (cos(theta + k*pi/3), sin(theta + k*pi/3))
    
    # This is equivalent to:
    #   cx * n_k_x + cy * n_k_y >= min_i (px_i * n_k_x + py_i * n_k_y) - S*sqrt(3)/2
    #   cx * n_k_x + cy * n_k_y <= max_i (px_i * n_k_x + py_i * n_k_y) + S*sqrt(3)/2
    
    # The minimal S for given (cx, cy, theta) is:
    #   S = max_k [max_i dot(p_i - c, n_k) - min_i dot(p_i - c, n_k)] / sqrt(3)
    
    # We want to minimize S over (cx, cy, theta).
    
    # Let's do a grid search over theta in [0, pi/3) and for each theta,
    # find the optimal (cx, cy) by checking the centroid or using a simple optimization.
    
    best_S = float('inf')
    best_cx = 0.0
    best_cy = 0.0
    best_theta = 0.0
    
    # Grid search over theta
    for theta_deg in range(0, 60, 1):
        theta = np.radians(theta_deg)
        normals = []
        for k in range(6):
            angle = theta + k * np.pi / 3
            normals.append((np.cos(angle), np.sin(angle)))
        
        # For this theta, find optimal (cx, cy)
        # The optimal center is the center of the minimal-width band in each normal direction
        # Let's try the centroid of the points as a starting point
        cx0 = np.mean([p[0] for p in positions])
        cy0 = np.mean([p[1] for p in positions])
        
        # Try a small grid around the centroid
        for dx in np.linspace(-2, 2, 20):
            for dy in np.linspace(-2, 2, 20):
                cx = cx0 + dx
                cy = cy0 + dy
                S_req = 0.0
                for nx, ny in normals:
                    projections = [px * nx + py * ny for px, py, _ in positions]
                    proj_max = max(projections)
                    proj_min = min(projections)
                    # The width in this direction is proj_max - proj_min
                    # We need S * sqrt(3) >= proj_max - proj_min + 2 * (distance from center to nearest side)
                    # Actually: for the hexagon to contain all points, for each normal n_k:
                    #   max_i dot(p_i, n_k) - dot(c, n_k) <= S * sqrt(3)/2
                    #   dot(c, n_k) - min_i dot(p_i, n_k) <= S * sqrt(3)/2
                    # So: dot(c, n_k) >= max_i dot(p_i, n_k) - S*sqrt(3)/2
                    #     dot(c, n_k) <= min_i dot(p_i, n_k) + S*sqrt(3)/2
                    # This requires: max_i dot(p_i, n_k) - S*sqrt(3)/2 <= min_i dot(p_i, n_k) + S*sqrt(3)/2
                    # i.e., S*sqrt(3) >= max_i dot(p_i, n_k) - min_i dot(p_i, n_k)
                    # So S >= (max_i dot(p_i, n_k) - min_i dot(p_i, n_k)) / sqrt(3)
                    # But this is independent of (cx, cy)! The minimal S for a given theta is:
                    #   S(theta) = max_k [max_i dot(p_i, n_k) - min_i dot(p_i, n_k)] / sqrt(3)
                    # And the center (cx, cy) must satisfy:
                    #   max_i dot(p_i, n_k) - S*sqrt(3)/2 <= dot(c, n_k) <= min_i dot(p_i, n_k) + S*sqrt(3)/2 for all k
                    # The feasible region for (cx, cy) is a hexagon. If non-empty, we can pick any point.
                    width = proj_max - proj_min
                    S_candidate = width / sqrt3
                    if S_candidate > S_req:
                        S_req = S_candidate
                
                if S_req < best_S:
                    # Check if there exists a valid (cx, cy)
                    # For simplicity, use the midpoint of the feasible region
                    feasible = True
                    cx_val = 0.0
                    cy_val = 0.0
                    for nx, ny in normals:
                        proj_max = max(px * nx + py * ny for px, py, _ in positions)
                        proj_min = min(px * nx + py * ny for px, py, _ in positions)
                        # dot(c, n) must be in [proj_max - S_req*sqrt3/2, proj_min + S_req*sqrt3/2]
                        # This is [proj_max - S_req*sqrt3/2, proj_min + S_req*sqrt3/2]
                        # Note: proj_max - S_req*sqrt3/2 <= proj_min + S_req*sqrt3/2 since S_req*sqrt3 >= proj_max - proj_min
                        low = proj_max - S_req * sqrt3 / 2
                        high = proj_min + S_req * sqrt3 / 2
                        # We need dot(c, n) in [low, high]
                        # This is a half-plane constraint. For now, just record that it's feasible.
                        pass
                    # For now, accept this S and use centroid as center
                    if S_req < best_S:
                        best_S = S_req
                        best_cx = cx0
                        best_cy = cy0
                        best_theta = theta
    
    # Use the best theta found
    # Re-center the arrangement so the outer hexagon is centered at origin
    # The optimal center for the outer hexagon is the average of the min and max projections in each direction
    theta = best_theta
    normals = []
    for k in range(6):
        angle = theta + k * np.pi / 3
        normals.append((np.cos(angle), np.sin(angle)))
    
    # Compute the optimal center as the intersection of midlines
    # For each normal, the center's projection should be at (max + min)/2
    cx_sum = 0.0
    cy_sum = 0.0
    for nx, ny in normals:
        projections = [px * nx + py * ny for px, py, _ in positions]
        proj_max = max(projections)
        proj_min = min(projections)
        mid = (proj_max + proj_min) / 2
        # We want cx*nx + cy*ny = mid
        # This is underdetermined; use least squares
        cx_sum += mid * nx
        cy_sum += mid * ny
    
    cx = cx_sum / 6
    cy = cy_sum / 6
    
    # Recompute S with this center
    S_final = 0.0
    for nx, ny in normals:
        projections = [px * nx + py * ny for px, py, _ in positions]
        proj_max = max(projections)
        proj_min = min(projections)
        # Distance from center to farthest point in this direction
        center_proj = cx * nx + cy * ny
        dist_max = proj_max - center_proj
        dist_min = center_proj - proj_min
        S_candidate = max(dist_max, dist_min) / (sqrt3 / 2)
        if S_candidate > S_final:
            S_final = S_candidate
    
    # Translate all inner hexagons so the outer hexagon is centered at origin
    inner = [[x - cx, y - cy, a] for x, y, a in positions]
    
    return inner, (0.0, 0.0), S_final, np.degrees(theta)
# EVOLVE-BLOCK-END

