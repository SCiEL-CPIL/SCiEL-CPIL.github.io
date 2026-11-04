import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate a 3x4 grid of hexagons
    # Flat-top hexagons, side length 1
    # Centers in hex grid coordinates
    rows = 3
    cols = 4
    
    centers = []
    for r in range(rows):
        for c in range(cols):
            # Hex grid to Cartesian
            # x = 1.5 * c + 0.75 * (r % 2)
            # y = sqrt(3) * (r + 0.5 * (c % 2)) ? 
            # Standard flat-top:
            # x = 1.5 * c
            # y = sqrt(3) * (r + 0.5 * (c % 2))
            x = 1.5 * c
            y = np.sqrt(3) * (r + 0.5 * (c % 2))
            centers.append((x, y))
            
    # Convert to numpy array
    centers = np.array(centers)
    
    # Center the grid
    centers -= np.mean(centers, axis=0)
    
    # Calculate minimal L
    # Vertices of a unit flat-top hexagon centered at (0,0)
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    verts = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    max_L = 0.0
    for cx, cy in centers:
        for vx, vy in verts:
            px = cx + vx
            py = cy + vy
            # Constraints for flat-top hexagon of side L centered at origin:
            # |x| <= L
            # |sqrt(3)x + y| <= sqrt(3)L
            # |-sqrt(3)x + y| <= sqrt(3)L
            # Note: |sqrt(3)x + y| <= sqrt(3)L is equivalent to |x + y/sqrt(3)| <= L
            
            L1 = abs(px)
            L2 = abs(np.sqrt(3)*px + py) / np.sqrt(3)
            L3 = abs(-np.sqrt(3)*px + py) / np.sqrt(3)
            
            L_val = max(L1, L2, L3)
            if L_val > max_L:
                max_L = L_val
                
    # Prepare output
    inner = []
    for cx, cy in centers:
        inner.append([float(cx), float(cy), 0.0])
        
    return inner, (0.0, 0.0), float(max_L), 0.0
# EVOLVE-BLOCK-END
