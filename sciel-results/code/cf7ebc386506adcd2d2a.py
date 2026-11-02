import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 10.0  # seconds
    
    # Lattice vectors for honeycomb packing of unit hexagons (flat-top)
    # Distance between centers is sqrt(3)
    u = np.array([np.sqrt(3), 0])
    v = np.array([np.sqrt(3)/2, 1.5])
    
    # Define 11 points in the lattice
    # 7 core + 4 outer
    pts = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, 0), (-2, 0),
        (0, 2), (0, -2)
    ]
    
    centers = np.array([i*u + j*v for i, j in pts])
    
    # Vertices of a unit flat-top hexagon relative to center
    # Angles: 0, 60, 120, 180, 240, 300? No, flat top means vertices at 30, 90...
    # Let's use the standard flat-top vertices:
    # (sqrt(3)/2, 1/2), (0, 1), (-sqrt(3)/2, 1/2), (-sqrt(3)/2, -1/2), (0, -1), (sqrt(3)/2, -1/2)
    hex_verts = np.array([
        [np.sqrt(3)/2, 0.5],
        [0, 1],
        [-np.sqrt(3)/2, 0.5],
        [-np.sqrt(3)/2, -0.5],
        [0, -1],
        [np.sqrt(3)/2, -0.5]
    ])
    
    # All vertices of all inner hexagons
    all_verts = centers[:, np.newaxis, :] + hex_verts[np.newaxis, :, :]
    all_verts = all_verts.reshape(-1, 2)
    
    best_side = float('inf')
    best_theta = 0.0
    
    # Search for optimal rotation of outer hexagon
    # The outer hexagon side length S is determined by max width in 6 directions
    # Directions are theta + k*pi/3
    for theta in np.linspace(0, np.pi/3, 181):
        if time.time() > deadline:
            break
            
        max_width = 0
        for k in range(6):
            angle = theta + k * np.pi / 3
            n = np.array([np.cos(angle), np.sin(angle)])
            projections = all_verts @ n
            width = np.max(projections) - np.min(projections)
            if width > max_width:
                max_width = width
                
        # Side length S = (2/sqrt(3)) * (max_width / 2) = max_width / sqrt(3)
        side = max_width / np.sqrt(3)
        
        if side < best_side:
            best_side = side
            best_theta = theta
            
    # Return the configuration
    # inner: list of [x, y, angle_deg]
    # angle_deg is 0 for flat-top
    inner = [[c[0], c[1], 0.0] for c in centers]
    
    # outer center: (0,0) is optimal for this symmetric cluster
    # outer side: best_side
    # outer angle: best_theta (degrees)
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = np.degrees(best_theta)
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
