import numpy as np
import time

N = 12

def hexagon_sdf(p, center, side, angle_deg):
    """
    Signed distance function for a regular hexagon.
    p: point (x, y)
    center: (cx, cy)
    side: side length
    angle_deg: rotation of hexagon
    """
    cx, cy = center
    theta = np.radians(angle_deg)
    # Rotate point to hexagon frame
    dx = p[0] - cx
    dy = p[1] - cy
    x = dx * np.cos(-theta) - dy * np.sin(-theta)
    y = dx * np.sin(-theta) + dy * np.cos(-theta)
    
    # Hexagon SDF
    # Flat-top hexagon with side 'side'
    # Vertices at (side, 0), (side/2, side*sqrt(3)/2), ...
    # Inradius r = side * sqrt(3) / 2
    r = side * np.sqrt(3) / 2
    
    # Transform to "hexagon coordinates"
    # For a flat-top hexagon, the boundaries are at x = +/- side, y = +/- r, and slanted sides.
    # A simpler SDF:
    # Distance to the 6 lines forming the hexagon.
    # Line 1: x = side
    # Line 2: x = -side
    # Line 3: sqrt(3)x + y = 2r  => sqrt(3)x + y = side*sqrt(3)
    # Line 4: sqrt(3)x - y = -side*sqrt(3)
    # Line 5: -sqrt(3)x + y = side*sqrt(3)
    # Line 6: -sqrt(3)x - y = -side*sqrt(3)
    
    d1 = side - x
    d2 = side + x
    d3 = side * np.sqrt(3) - (np.sqrt(3)*x + y)
    d4 = side * np.sqrt(3) + (np.sqrt(3)*x - y)
    d5 = side * np.sqrt(3) - (-np.sqrt(3)*x + y)
    d6 = side * np.sqrt(3) + (-np.sqrt(3)*x - y)
    
    # The SDF is the minimum of the distances to the boundaries, signed.
    # Inside: all d > 0.
    # SDF = min(d1, d2, d3, d4, d5, d6) / sqrt(1 + slope^2)?
    # Actually, for lines Ax+By+C=0, distance is |Ax+By+C|/sqrt(A^2+B^2).
    # Here, we want the distance to the closest boundary.
    
    # Normalized distances
    dists = [
        d1 / 1.0,
        d2 / 1.0,
        d3 / np.sqrt(3 + 1), # sqrt(4)=2
        d4 / np.sqrt(3 + 1),
        d5 / np.sqrt(3 + 1),
        d6 / np.sqrt(3 + 1)
    ]
    
    return min(dists)

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Initial guess: Grid layout
    # 12 hexagons.
    # Let's try a 3x3 grid + 3 extra
    # Or a hexagonal cluster of 12.
    # Hex grid coords (q, r)
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)... 
    # We need 12.
    # Let's use 1 center + 6 ring 1 + 5 from ring 2.
    
    coords = [(0,0)]
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    coords.extend(ring1)
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,1)]
    coords.extend(ring2[:5])
    
    # Convert hex grid coords to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # This is for pointy-top?
    # Let's use flat-top.
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    positions = []
    for q, r in coords:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2.0)
        positions.append([x, y])
        
    positions = np.array(positions)
    angles = np.zeros(N)
    
    # Optimize L and positions
    # We want to minimize L such that all hexagons are inside container of side L.
    # And no collisions.
    
    # Simple iterative approach:
    # Start with L = 5.0
    # Check if valid. If valid, try smaller L.
    # If invalid, adjust positions.
    
    best_L = 5.0
    best_pos = positions.copy()
    best_ang = angles.copy()
    
    # Helper to check validity
    def check_validity(pos, ang, L):
        # Check boundary
        for i in range(N):
            # Sample points on the boundary of the i-th hexagon
            # To be safe, check vertices and midpoints of edges
            cx, cy = pos[i]
            theta = ang[i]
            # Vertices of unit hexagon
            verts = np.array([
                [1.0, 0.0],
                [0.5, np.sqrt(3)/2],
                [-0.5, np.sqrt(3)/2],
                [-1.0, 0.0],
                [-0.5, -np.sqrt(3)/2],
                [0.5, -np.sqrt(3)/2]
            ])
            # Rotate
            c_t = np.cos(theta)
            s_t = np.sin(theta)
            rot = np.array([[c_t, -s_t], [s_t, c_t]])
            verts_rot = verts @ rot.T
            verts_rot[:, 0] += cx
            verts_rot[:, 1] += cy
            
            for v in verts_rot:
                sdf = hexagon_sdf(v, (0,0), L, 0)
                if sdf < 0:
                    return False
        
        # Check collisions
        # Approximate by checking distance between centers > 2 * inradius?
        # Inradius of unit hex is sqrt(3)/2.
        # If centers are closer than sqrt(3), they overlap.
        # This is a necessary condition, but not sufficient for rotated hexagons.
        # For a quick check, use center distance.
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < np.sqrt(3) - 1e-6:
                    return False
        return True

    # Iterative refinement
    # We will try to reduce L and adjust positions to fix violations.
    
    current_L = 5.0
    pos = positions.copy()
    ang = angles.copy()
    
    # Initial check
    if check_validity(pos, ang, current_L):
        best_L = current_L
        best_pos = pos.copy()
        best_ang = ang.copy()
    else:
        # If invalid, we need to spread them out or increase L.
        # Since we start with a tight grid, they might be too close.
        # Let's just increase L until valid, then try to decrease.
        while not check_validity(pos, ang, current_L) and time.time() < deadline:
            current_L += 0.1
        if check_validity(pos, ang, current_L):
            best_L = current_L
            best_pos = pos.copy()
            best_ang = ang.copy()
        else:
            # Fallback to seed
            return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    # Now try to reduce L
    # Use a simple gradient-free search or local relaxation.
    # For simplicity, we will just try to reduce L in small steps and if it fails, 
    # we will do a few iterations of "push apart" and "pull in".
    
    for _ in range(100): # Max 100 reduction attempts
        if time.time() > deadline:
            break
        
        new_L = best_L - 0.05
        if new_L < 3.0:
            break
            
        # Try to fix validity at new_L
        # Simple relaxation: 
        # 1. If boundary violation, move center towards origin.
        # 2. If collision, move centers apart.
        
        pos_try = best_pos.copy()
        ang_try = best_ang.copy()
        
        for iter in range(50):
            if time.time() > deadline:
                break
            
            # Check violations
            valid = True
            # Boundary
            for i in range(N):
                cx, cy = pos_try[i]
                theta = ang_try[i]
                verts = np.array([
                    [1.0, 0.0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2],
                    [-1.0, 0.0], [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]
                ])
                c_t = np.cos(theta)
                s_t = np.sin(theta)
                rot = np.array([[c_t, -s_t], [s_t, c_t]])
                verts_rot = verts @ rot.T
                verts_rot[:, 0] += cx
                verts_rot[:, 1] += cy
                
                for v in verts_rot:
                    sdf = hexagon_sdf(v, (0,0), new_L, 0)
                    if sdf < 0:
                        valid = False
                        # Move towards origin
                        pos_try[i] -= 0.01 * np.array([v[0], v[1]])
                        
            # Collisions
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.linalg.norm(pos_try[i] - pos_try[j])
                    if dist < np.sqrt(3):
                        valid = False
                        diff = pos_try[i] - pos_try[j]
                        norm = np.linalg.norm(diff)
                        if norm > 0:
                            diff = diff / norm
                        else:
                            diff = np.array([1.0, 0.0])
                        pos_try[i] += 0.01 * diff
                        pos_try[j] -= 0.01 * diff
            
            if valid:
                break
                
        if valid and check_validity(pos_try, ang_try, new_L):
            best_L = new_L
            best_pos = pos_try
            best_ang = ang_try
        else:
            # If we can't fix it easily, stop reducing
            break
            
    # Return best found
    # Format: inner rows (x, y, angle_degrees), outer center, side, angle
    inner = []
    for i in range(N):
        inner.append([best_pos[i][0], best_pos[i][1], np.degrees(best_ang[i])])
        
    return inner, (0.0, 0.0), best_L, 0.0

