import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the 12 centers for a dense hexagonal packing
    # Using a hexagonal lattice with spacing sqrt(3)
    # We select the 12 points closest to the origin from the lattice
    
    # Lattice points:
    # (0,0)
    # Ring 1 (6 points)
    # Ring 2 (6 points, we take 5)
    
    centers = []
    
    # Center
    centers.append((0.0, 0.0))
    
    # Ring 1
    r1 = math.sqrt(3)
    centers.append((r1, 0.0))
    centers.append((-r1, 0.0))
    centers.append((r1/2, 1.5))
    centers.append((r1/2, -1.5))
    centers.append((-r1/2, 1.5))
    centers.append((-r1/2, -1.5))
    
    # Ring 2 (Select 5 to make total 12)
    # Points at distance 3 from origin
    # (0, 3), (0, -3)
    # (3*sqrt(3)/2, 1.5), (3*sqrt(3)/2, -1.5)
    # (-3*sqrt(3)/2, 1.5), (-3*sqrt(3)/2, -1.5)
    
    r2_x = 3 * math.sqrt(3) / 2
    r2_y = 1.5
    
    centers.append((0.0, 3.0))
    centers.append((0.0, -3.0))
    centers.append((r2_x, r2_y))
    centers.append((r2_x, -r2_y))
    centers.append((-r2_x, r2_y))
    # We omit (-r2_x, -r2_y) to keep count at 12
    
    inner = []
    for x, y in centers:
        # Rotate inner hexagons by 30 degrees to minimize bounding box
        inner.append([x, y, 30.0])
        
    outer_center = (0.0, 0.0)
    # Side length calculated to contain the cluster
    # Max extent is at the vertices of the outermost hexagons
    # With rotation 30, max radial extent of inner hex is 1.0?
    # No, with rotation 30 (pointy top relative to flat top outer),
    # the max projection on any normal is sqrt(3)/2.
    # The farthest center is at distance 3.
    # Direction of (0,3) is 90 deg.
    # Normal 60 deg projection: 3 * cos(30) = 3 * sqrt(3)/2.
    # Total apothem needed: 3*sqrt(3)/2 + sqrt(3)/2 = 2*sqrt(3).
    # S * sqrt(3)/2 = 2*sqrt(3) => S = 4.
    
    side = 4.0
    angle = 0.0
    
    return inner, outer_center, side, angle
# EVOLVE-BLOCK-END
