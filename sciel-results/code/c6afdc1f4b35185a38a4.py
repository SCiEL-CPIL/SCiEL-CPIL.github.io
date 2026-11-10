import math
import random

N = 11

def construct():
    # Compact hexagonal cluster packing for 11 unit hexagons.
    # The optimal arrangement for 11 hexagons in a hexagonal lattice is a central
    # hexagon surrounded by 6 neighbors (7 total), plus 4 additional hexagons
    # placed at specific positions to minimize the bounding hexagon side length.
    
    # Use axial coordinates for hexagonal grid
    # Center at (0,0), neighbors at distance 1 in 6 directions
    # The 7-hex cluster: (0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # For 11, add 4 more. The minimal bounding hexagon side for 11 in standard
    # orientation is 4.0. We can verify this and return a valid configuration.
    
    sqrt3 = math.sqrt(3)
    
    # Axial to pixel conversion for pointy-top hexagons with side length 1
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # But we need flat-top or pointy-top? Let's use a consistent coordinate system.
    # For regular hexagons with side length 1, the distance between centers of
    # adjacent hexagons is sqrt(3).
    
    # Let's use the standard hex grid with flat-top orientation.
    # Adjacent centers are separated by sqrt(3) in 6 directions.
    # Directions: 0°, 60°, 120°, 180°, 240°, 300°
    
    # Place 11 hexagons in a compact cluster
    # Using axial coordinates (q, r):
    # 7-hex core: (0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # 4 additional: (2,-1), (2,0), (-2,1), (-2,0) -- need to check if this minimizes
    
    # Actually, the known optimal for 11 hexagons in a hexagonal container is side=4.
    # Let's construct a valid packing with side=4.
    
    # Axial coordinates for 11 hexagons in a hexagonal arrangement with max axial coord 3
    # A hexagon of side n in axial coords has all (q,r) with max(|q|,|r|,|q+r|) <= n
    # For n=3, that's 1+6+12=19 hexagons. Too many.
    # For n=2, that's 1+6=7... no, 1+6+12=19? No.
    # Side 1: 1 hex
    # Side 2: 7 hex
    # Side 3: 19 hex
    # So 11 is between side 2 and 3. We need a subset of 11 from a larger cluster.
    
    # The minimal bounding hexagon side for 11 hexagons is 4.0 in terms of side length
    # of the outer hexagon. Let's verify by placing them and computing the bounding box.
    
    # Place 11 hexagons at these axial coordinates (q, r):
    coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),  # 7 core
        (2, -1), (2, -2), (-2, 2), (-2, 1)  # 4 more
    ]
    
    # Convert to Cartesian (flat-top hexagons, side length 1)
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    inner = []
    for q, r in coords:
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        inner.append([x, y, 0.0])
    
    # Center of mass or just use (0,0) as center
    center = (0.0, 0.0)
    
    # Compute the required side length of the outer hexagon
    # The outer hexagon must contain all 11 inner hexagons.
    # A regular hexagon with side S centered at (cx, cy) contains a point (x, y) if
    # the point is within the hexagonal boundary.
    
    # For a regular hexagon with flat-top orientation, side S, centered at origin:
    # The width is 2*S, height is sqrt(3)*S
    # Actually, let's compute the bounding hexagon side length directly.
    
    # The minimal side length of a regular hexagon containing all points can be
    # computed by finding the maximum extent in 3 directions (for flat-top).
    
    # For flat-top hexagon with side S, the constraints are:
    # |x| <= S
    # |sqrt(3)*x/2 + y| <= sqrt(3)*S/2  (upper right edge)
    # |-sqrt(3)*x/2 + y| <= sqrt(3)*S/2  (lower right edge)
    
    # Wait, let me reconsider. For a regular hexagon with side length S, flat-top:
    # Vertices at (S,0), (S/2, sqrt(3)S/2), (-S/2, sqrt(3)S/2), (-S,0), (-S/2,-sqrt(3)S/2), (S/2,-sqrt(3)S/2)
    
    # The support function in direction theta gives the maximum extent.
    # For the hexagon to contain all points, we need S such that all points are inside.
    
    # Let's compute the required S by checking each point against the hexagon boundaries.
    # For a point (x,y) to be inside a flat-top hexagon of side S centered at origin:
    # 1. |x| <= S
    # 2. |y| <= sqrt(3)/2 * S  (not quite, this is for the width)
    # Actually, the three pair of parallel edges:
    # Edge 1 (right): x = S
    # Edge 2 (upper right): sqrt(3)*x + 2*y = sqrt(3)*S
    # Edge 3 (lower right): sqrt(3)*x - 2*y = sqrt(3)*S
    
    # Let me compute the minimal S numerically.
    
    def min_hex_side(points, cx=0.0, cy=0.0):
        """Find the minimal side length of a flat-top regular hexagon centered at (cx,cy) containing all points."""
        S = 0.0
        for x, y, _ in points:
            dx = x - cx
            dy = y - cy
            # For flat-top hexagon, the constraints are:
            # |dx| <= S
            # |sqrt(3)*dx + 2*dy| <= sqrt(3)*S
            # |sqrt(3)*dx - 2*dy| <= sqrt(3)*S
            s1 = abs(dx)
            s2 = abs(sqrt3 * dx + 2 * dy) / sqrt3
            s3 = abs(sqrt3 * dx - 2 * dy) / sqrt3
            S = max(S, s1, s2, s3)
        return S
    
    side = min_hex_side(inner)
    
    # Round up to be safe (add small epsilon for floating point)
    side = math.ceil(side * 1000) / 1000.0 + 1e-6
    
    return inner, center, side, 0.0
