import numpy as np
import time

def _calc_side_length(centers, inner_angles, outer_angle_deg):
    # Calculate minimum side length of outer hexagon with given rotation
    # to contain all inner hexagons
    # Outer hexagon center at (0,0)
    
    # Normals of the outer hexagon sides
    # For a regular hexagon with circumradius S (side length S),
    # the apothem is A = S * sqrt(3) / 2.
    # The sides are at distance A from the center.
    # The normals to the sides are at angles outer_angle_deg + 60*k?
    # If vertices are at outer_angle_deg + 60*k, normals are at outer_angle_deg + 60*k + 30?
    # Let's verify.
    # Vertex at 0. Side to 60. Midpoint at 30. Normal at 30.
    # So if vertices are at theta_v, normals are at theta_v + 30.
    # In our case, we define the hexagon by its side length S and rotation.
    # Let's define rotation as the angle of the first vertex.
    # Then normals are at angle + 30.
    
    theta_v = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        ang = theta_v + np.deg2rad(30) + np.deg2rad(60 * k)
        normals.append((np.cos(ang), np.sin(ang)))
        
    max_proj = 0.0
    for cx, cy, ang_deg in zip(centers[:, 0], centers[:, 1], inner_angles):
        # Vertices of this inner hexagon
        theta_in = np.deg2rad(ang_deg)
        for m in range(6):
            vx = cx + np.cos(theta_in + np.deg2rad(60 * m))
            vy = cy + np.sin(theta_in + np.deg2rad(60 * m))
            for nx, ny in normals:
                proj = abs(vx * nx + vy * ny)
                if proj > max_proj:
                    max_proj = proj
                    
    # A = max_proj
    # S = 2 * A / sqrt(3)
    return 2 * max_proj / np.sqrt(3)

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    # Initial configuration
    eps = 1e-9
    centers = []
    angles = []
    
    # Ring 0
    centers.append((0.0, 0.0))
    angles.append(30.0)
    
    # Ring 1
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        angles.append(30.0)
        
    # Ring 2
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        angles.append(30.0)
        
    centers_arr = np.array(centers)
    angles_arr = np.array(angles)
    
    # Optimize outer rotation
    best_side = float('inf')
    best_outer_angle = 30.0
    
    # Sample outer angles
    # Since the cluster is symmetric across Y axis, and outer hex is 6-fold symmetric,
    # we can search 0 to 60.
    # Step size 1 degree might be enough, then refine.
    for outer_ang in np.arange(0, 60, 1.0):
        if time.time() > DEADLINE:
            break
        s = _calc_side_length(centers_arr, angles_arr, outer_ang)
        if s < best_side:
            best_side = s
            best_outer_angle = outer_ang
            
    # Refine around best
    if time.time() < DEADLINE - 10:
        for outer_ang in np.arange(best_outer_angle - 2, best_outer_angle + 2, 0.1):
            if outer_ang < 0 or outer_ang >= 60: continue
            s = _calc_side_length(centers_arr, angles_arr, outer_ang)
            if s < best_side:
                best_side = s
                best_outer_angle = outer_ang

    # Final result
    inner = []
    for x, y, a in zip(centers_arr[:, 0], centers_arr[:, 1], angles_arr):
        inner.append([x, y, a])
        
    return inner, (0.0, 0.0), best_side, best_outer_angle
