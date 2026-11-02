import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Lattice vectors for unit hexagons (flat-topped)
    # Distance between centers is sqrt(3)
    a = np.array([np.sqrt(3), 0.0])
    b = np.array([np.sqrt(3)/2.0, 1.5])
    
    # Axial coordinates (q, r) for the 11 hexagons
    # Ring 0
    coords_axial = [(0, 0)]
    
    # Ring 1
    ring1 = [
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, 0),
        (-1, 1),
        (0, 1)
    ]
    coords_axial.extend(ring1)
    
    # Ring 2 (4 specific positions to minimize bounding box)
    ring2_subset = [
        (2, -1),
        (-1, 2),
        (-2, 1),
        (1, -2)
    ]
    coords_axial.extend(ring2_subset)
    
    # Convert to Cartesian coordinates
    inner = []
    for q, r in coords_axial:
        x = q * a[0] + r * b[0]
        y = q * a[1] + r * b[1]
        # Angle 0.0 corresponds to flat-topped hexagon
        inner.append([x, y, 0.0])
        
    # Outer hexagon parameters
    # Side length 4.0
    # Center (0,0)
    # Rotation 0.0
    return inner, (0.0, 0.0), 4.0, 0.0
# EVOLVE-BLOCK-END
