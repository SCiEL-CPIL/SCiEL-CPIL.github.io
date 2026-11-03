"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Unit hexagon vertices (flat-top, side=1, centered at origin)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    hex_verts = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in range(0, 360, 60)])
    
    # Hex lattice positions for 11 hexagons: center + 6 first-ring + 4 second-ring (alternating)
    # Using axial coordinates (q, r) -> Cartesian for flat-top hexagons
    # x = sqrt(3) * (q + r/2), y = 3/2 * r
    positions_axial = [
        (0, 0),    # center
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),  # first ring
        (2, 0), (2, -2), (0, -2), (-2, 0)  # 4 from second ring (alternating)
    ]
    
    sqrt3 = np.sqrt(3)
    cartesian = []
    for q, r in positions_axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        cartesian.append((x, y))
    
    # Find minimum S such that all hexagon vertices fit inside outer hexagon
    # Outer hexagon: flat-top, side S, centered at origin
    # A point (px, py) is inside if:
    # |px| <= S
    # |py| <= S * sqrt(3)/2
    # |px/2 + py/sqrt(3)| <= S/2  ... let me derive properly
    
    # For flat-top hex with side S, the six half-planes are:
    # x <= S, -x <= S, 
    # (sqrt(3)/2)*y - (1/2)*x <= S*sqrt(3)/2, etc.
    # Actually, the constraints are:
    # |x| <= S
    # |y| <= S*sqrt(3)/2  
    # |x/2 + y/sqrt(3)| <= S/2  ... 
    
    # Let me use: for flat-top hex, the apothem in x-direction is S, in y-direction is S*sqrt(3)/2
    # The six edges have normals at 0, 60, 120, 180, 240, 300 degrees
    # Constraint: dot(point, normal) <= S * cos(0) = S for each normal direction
    # Normals: (1,0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1,0), (-1/2,-sqrt(3)/2), (1/2,-sqrt(3)/2)
    
    normals = np.array([
        [1, 0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1, 0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])
    
    # For each inner hexagon vertex, compute the required S
    min_S = 0.0
    for cx, cy in cartesian:
        for vx, vy in hex_verts:
            px = cx + vx
            py = cy + vy
            for nx, ny in normals:
                val = nx * px + ny * py
                if val > min_S:
                    min_S = val
    
    # Round up slightly to account for floating point
    S = min_S + 1e-9
    
    # Build inner list: (x, y, angle_degrees)
    inner = [[cx, cy, 0.0] for cx, cy in cartesian]
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
