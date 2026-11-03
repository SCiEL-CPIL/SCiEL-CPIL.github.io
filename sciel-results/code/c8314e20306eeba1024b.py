import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # Generate candidate configurations of 11 unit hexagons
    # Each config is a list of (x, y) positions for hexagon centers
    # Unit hexagon: side length 1, so center-to-center distance for adjacent = sqrt(3)
    
    configs = []
    
    # Config 1: Compact cluster - center + ring of 6 + 4 more
    # Hexagonal lattice: neighbors at 60 degree intervals, distance sqrt(3)
    d = np.sqrt(3)
    
    # Central hexagon at origin, 6 around it
    center_config = [(0, 0)]
    for i in range(6):
        angle = i * np.pi / 3
        center_config.append((d * np.cos(angle), d * np.sin(angle)))
    # Add 4 more in a compact way - extend one direction
    # Add at positions that fill gaps
    extra = [
        (2*d, 0),
        (d, d*np.sqrt(3)),
        (d, -d*np.sqrt(3)),
        (0, 2*d*np.sqrt(3)/2)  # This might not be right, let me think
    ]
    # Better: use hexagonal coordinates
    # Let me use axial coordinates for hex grid
    # Axial to pixel: x = d * (q + r/2), y = d * sqrt(3)/2 * r
    
    def axial_to_pixel(q, r):
        x = d * (q + r/2)
        y = d * np.sqrt(3)/2 * r
        return (x, y)
    
    # Generate hexagonal number 7 (center + 6) plus 4 more
    # Hex 7: q,r in {-1,0,1} with |q|+|r|+|q+r| <= 2? No, that's 7 points for radius 1
    # Actually hex radius 1 has 7 points: (0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)
    # Hex radius 2 has 19 points. We want 11.
    
    # Let me just enumerate several specific configurations
    
    # Config A: Line of 11
    line_config = [(i * d, 0) for i in range(-5, 6)]
    configs.append(line_config)
    
    # Config B: 2 rows of 5-6
    row1 = [(i * d, 0) for i in range(-2, 3)]  # 5
    row2 = [(i * d + d/2, d*np.sqrt(3)/2) for i in range(-2, 3)]  # 5
    row3 = [(0, -d*np.sqrt(3)/2)]  # 1
    configs.append(row1 + row2 + row3)
    
    # Config C: Compact cluster using axial coords
    # Pick 11 positions from hex grid
    hex_positions = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if abs(q) + abs(r) + abs(q+r) <= 3:
                hex_positions.append((q, r))
    # Sort by distance from center, take 11 closest
    hex_positions.sort(key=lambda p: p[0]**2 + p[1]**2 + p[0]*p[1])
    compact11 = [axial_to_pixel(q, r) for q, r in hex_positions[:11]]
    configs.append(compact11)
    
    # Config D: Another compact arrangement
    # Center + 6 ring + 4 extending in one direction
    base7 = [axial_to_pixel(q, r) for q, r in [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]]
    extend4 = [axial_to_pixel(q, r) for q, r in [(2,0),(2,1),(3,0),(3,1)]]
    configs.append(base7 + extend4)
    
    # Config E: 3-4-4 triangle
    tri = []
    for i in range(3):
        tri.append(axial_to_pixel(-i, 0))
    for i in range(4):
        tri.append(axial_to_pixel(i, 1))
    for i in range(4):
        tri.append(axial_to_pixel(i, -1))
    configs.append(tri)
    
    best_side = float('inf')
    best_inner = None
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    def compute_min_enclosing_hex_side(positions, theta):
        """Compute minimal side length of regular hexagon (rotated by theta) enclosing all positions."""
        # Regular hexagon of side S has inradius = S * sqrt(3)/2
        # The 6 face normals are at angles theta + k*60 degrees
        # For a point to be inside: for each normal n_i, dot(p, n_i) <= S*sqrt(3)/2
        # But we also need the opposite: dot(p, -n_i) <= S*sqrt(3)/2
        # So S >= max_i max_p |dot(p, n_i)| / (sqrt(3)/2)
        
        positions_arr = np.array(positions)
        # Center the positions to optimize
        centroid = positions_arr.mean(axis=0)
        centered = positions_arr - centroid
        
        max_extent = 0
        for k in range(6):
            angle = theta + k * np.pi / 3
            nx = np.cos(angle)
            ny = np.sin(angle)
            projections = centered @ np.array([nx, ny])
            extent = np.max(np.abs(projections))
            max_extent = max(max_extent, extent)
        
        # S * sqrt(3)/2 >= max_extent
        S = max_extent / (np.sqrt(3) / 2)
        return S, centroid
    
    for config in configs:
        if time.time() > deadline - 10:
            break
        
        positions = config
        # Optimize rotation angle
        best_S_for_config = float('inf')
        best_theta_for_config = 0.0
        best_centroid_for_config = np.array([0, 0])
        
        # Coarse search
        for theta in np.linspace(0, np.pi/3, 60):
            S, centroid = compute_min_enclosing_hex_side(positions, theta)
            if S < best_S_for_config:
                best_S_for_config = S
                best_theta_for_config = theta
                best_centroid_for_config = centroid
        
        # Fine search around best
        if time.time() < deadline - 5:
            for theta in np.linspace(best_theta_for_config - 0.05, best_theta_for_config + 0.05, 100):
                S, centroid = compute_min_enclosing_hex_side(positions, theta)
                if S < best_S_for_config:
                    best_S_for_config = S
                    best_theta_for_config = theta
                    best_centroid_for_config = centroid
        
        if best_S_for_config < best_side:
            best_side = best_S_for_config
            best_angle = np.degrees(best_theta_for_config)
            best_center = (float(best_centroid_for_config[0]), float(best_centroid_for_config[1]))
            # Store inner positions relative to global frame
            best_inner = [[float(p[0]), float(p[1]), 0.0] for p in positions]
    
    # If no valid config found, return a safe default
    if best_inner is None:
        inner = [[i * np.sqrt(3), 0.0, 0.0] for i in range(-5, 6)]
        return inner, (0.0, 0.0), 50.0, 0.0
    
    return best_inner, best_center, float(best_side), float(best_angle)
