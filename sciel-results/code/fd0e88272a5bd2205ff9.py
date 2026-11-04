import numpy as np
import time
import random

N = 12

def _hex_verts(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon."""
    angle_rad = np.deg2rad(angle_deg)
    # Vertices at angles 0, 60, 120, 180, 240, 300 relative to angle_rad
    # Side length s, circumradius s
    angles = angle_rad + np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack((xs, ys))

def _min_outer_side(inner_hexes, outer_center, outer_angle_deg):
    """
    Calculate the minimum side length L of the outer hexagon 
    centered at outer_center with rotation outer_angle_deg 
    that contains all inner_hexes.
    inner_hexes: list of (cx, cy, angle_deg)
    """
    max_L = 0.0
    ocx, ocy = outer_center
    o_rad = np.deg2rad(outer_angle_deg)
    
    for cx, cy, ang_deg in inner_hexes:
        verts = _hex_verts(cx, cy, 1.0, ang_deg)
        for vx, vy in verts:
            dx = vx - ocx
            dy = vy - ocy
            dist = np.sqrt(dx*dx + dy*dy)
            if dist == 0:
                continue
            angle = np.arctan2(dy, dx)
            # Angle relative to outer hexagon's symmetry axes
            # Outer hexagon vertices are at o_rad + 0, 60, ...
            # Outer hexagon edges midpoints are at o_rad + 30, 90, ...
            # The radius of the outer hexagon in direction theta is:
            # r(theta) = (L * sqrt(3) / 2) / cos((theta - o_rad) mod 60 - 30)
            # We need dist <= r(theta)
            # L >= dist * cos((theta - o_rad) mod 60 - 30) / (sqrt(3) / 2)
            
            # Normalize angle difference to [0, 60)
            diff = (angle - o_rad) % (np.pi / 3.0)
            # The angle relative to the nearest vertex is in [-30, 30] degrees
            # cos is maximized at 0 (vertex), minimized at 30 (edge)
            # Actually, the formula uses the angle relative to the edge normal?
            # Let's derive:
            # Hexagon with circumradius R=L.
            # Inradius r = L * sqrt(3) / 2.
            # Distance to edge in direction phi (0 is vertex) is r / cos(phi).
            # Wait, if phi=0 (vertex), dist = r / 1 = r? No, vertex dist is R.
            # If phi=0 is vertex, the boundary is at distance R.
            # If phi=30 is edge midpoint, the boundary is at distance r.
            # So dist(phi) = R / (2 * cos(phi - 30))? No.
            # Let's use the standard formula:
            # For a regular polygon with circumradius R and 2n sides (here 6 sides, n=3? No, 6 sides).
            # Distance from center to boundary at angle alpha (0 aligned with vertex):
            # d(alpha) = R * cos(pi/6) / cos(alpha - pi/6 * (2k+1)?)
            # Simpler: The hexagon is the intersection of 6 half-planes.
            # Or use the polar equation: r(theta) = (R * sqrt(3)/2) / cos((theta - theta_0) mod (pi/3) - pi/6)
            # Here R=L.
            # Let's check: if theta = theta_0 (vertex), mod is 0, cos(-pi/6) = sqrt(3)/2.
            # r = (L * sqrt(3)/2) / (sqrt(3)/2) = L. Correct.
            # If theta = theta_0 + pi/6 (edge mid), mod is pi/6, cos(0) = 1.
            # r = L * sqrt(3)/2. Correct.
            
            # So:
            # L >= dist * cos((theta - o_rad) mod (pi/3) - pi/6) / (sqrt(3)/2)
            
            cos_term = np.cos(diff - np.pi/6.0)
            if cos_term < 1e-9:
                continue # Should not happen for valid points
            L_req = dist * cos_term / (np.sqrt(3.0) / 2.0)
            if L_req > max_L:
                max_L = L_req
                
    return max_L

def construct():
    start_time = time.time()
    timeout = 140.0 # 10s margin
    
    # Initial configuration: 1 center, 6 ring, 5 extra
    # Center at (0,0)
    # Ring at distance sqrt(3)
    # Extra 5 attached to ring
    
    def init_config():
        hexes = []
        # Center
        hexes.append((0.0, 0.0, 0.0))
        # Ring of 6
        d = np.sqrt(3.0)
        for i in range(6):
            angle = i * 60.0
            rad = np.deg2rad(angle)
            cx = d * np.cos(rad)
            cy = d * np.sin(rad)
            # Align with lattice? Angle 0
            hexes.append((cx, cy, 0.0))
        
        # 5 extra hexagons
        # Attach to ring indices 0, 1, 2, 3, 4
        for i in range(5):
            # Parent is ring hex i
            parent_angle = i * 60.0
            parent_rad = np.deg2rad(parent_angle)
            # New hex is further out by sqrt(3)
            new_d = 2.0 * np.sqrt(3.0)
            new_cx = new_d * np.cos(parent_rad)
            new_cy = new_d * np.sin(parent_rad)
            hexes.append((new_cx, new_cy, 0.0))
            
        return hexes

    current_hexes = init_config()
    current_L = 40.0 # Initial large value
    best_hexes = current_hexes[:]
    best_L = current_L
    best_outer = (0.0, 0.0, 0.0)
    
    # Optimize outer center and angle for current hexes
    def optimize_outer(hexes):
        # Try a few outer angles and centers
        best_L_local = float('inf')
        best_o = (0.0, 0.0, 0.0)
        
        # Centroid of hexagons
        xs = [h[0] for h in hexes]
        ys = [h[1] for h in hexes]
        cx = np.mean(xs)
        cy = np.mean(ys)
        
        # Try outer angles 0, 30, and some random
        angles_to_try = [0.0, 30.0, 15.0, 45.0]
        # Also try shifting center slightly
        centers_to_try = [(cx, cy), (0.0, 0.0)]
        
        for ox, oy in centers_to_try:
            for oa in angles_to_try:
                L = _min_outer_side(hexes, (ox, oy), oa)
                if L < best_L_local:
                    best_L_local = L
                    best_o = (ox, oy, oa)
                    
        return best_L_local, best_o

    current_L, current_outer = optimize_outer(current_hexes)
    if current_L < best_L:
        best_L = current_L
        best_hexes = current_hexes[:]
        best_outer = current_outer
        
    # Local Search
    step_size = 0.5
    iterations = 0
    max_iterations = 50000
    
    while time.time() - start_time < timeout and iterations < max_iterations:
        iterations += 1
        
        # Pick a random hexagon to perturb
        idx = random.randint(0, N - 1)
        cx, cy, ang = current_hexes[idx]
        
        # Perturb
        new_cx = cx + random.uniform(-step_size, step_size)
        new_cy = cy + random.uniform(-step_size, step_size)
        new_ang = ang + random.uniform(-10.0, 10.0)
        
        # Update
        new_hexes = current_hexes[:]
        new_hexes[idx] = (new_cx, new_cy, new_ang)
        
        # Check collisions? 
        # For simplicity, we rely on the outer size penalty. 
        # If they overlap, the required outer size will likely increase, 
        # or we might need a stricter check. 
        # However, if they overlap, they are not disjoint. 
        # The task requires disjoint hexagons.
        # We should add a collision check.
        
        def check_disjoint(hexes):
            for i in range(N):
                for j in range(i+1, N):
                    # Approximate check: distance between centers
                    dx = hexes[i][0] - hexes[j][0]
                    dy = hexes[i][1] - hexes[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    # Min distance for disjoint hexagons is 0? 
                    # They can touch. 
                    # But if they overlap, dist < min_dist_overlap.
                    # For hexagons of side 1, min center distance is ~1.0?
                    # If aligned, they touch at edges, dist = sqrt(3) ~ 1.732.
                    # If rotated, they can be closer?
                    # The minimum distance between centers of two disjoint unit hexagons is 1.0?
                    # No, if they touch at a vertex, dist can be smaller?
                    # Actually, the "width" of a hexagon is sqrt(3).
                    # If they are disjoint, the distance between centers must be at least 
                    # the sum of inradii? No.
                    # Let's use a conservative bound: if dist < 1.0, they definitely overlap.
                    # If dist < sqrt(3), they might overlap depending on angle.
                    # To be safe, if dist < 1.5, we should check more carefully or penalize.
                    if dist < 1.0:
                        return False
            return True

        if not check_disjoint(new_hexes):
            # Reject or penalize
            # Simple rejection
            continue
            
        new_L, new_outer = optimize_outer(new_hexes)
        
        if new_L < current_L:
            current_hexes = new_hexes
            current_L = new_L
            current_outer = new_outer
            
            if current_L < best_L:
                best_L = current_L
                best_hexes = current_hexes[:]
                best_outer = current_outer
                
        # Adaptive step size
        if iterations % 1000 == 0:
            step_size *= 0.95
            
    # Final optimization
    final_L, final_outer = optimize_outer(best_hexes)
    
    inner = []
    for cx, cy, ang in best_hexes:
        inner.append([cx, cy, ang])
        
    return inner, (final_outer[0], final_outer[1]), final_L, final_outer[2]

# EVOLVE-BLOCK-END
