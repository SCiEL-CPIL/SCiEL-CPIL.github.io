"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Place unit hexagons in a hexagonal lattice arrangement inside a larger hexagon.
    # The inner hexagons are arranged in a centered hexagonal pattern of side length n_inner.
    # For a hexagonal packing of unit hexagons (side 1), the centers form a triangular lattice
    # with spacing sqrt(3) in the horizontal direction and appropriate vertical offsets.
    
    # Let's use a hexagonal arrangement of the inner hexagons.
    # For a hexagonal cluster of "radius" r (number of hexagons from center to edge),
    # the total number of hexagons is 1 + 3*r*(r+1).
    
    # We need exactly N=11 hexagons.
    # r=1 gives 7, r=2 gives 19. So we can't get exactly 11 with a perfect hexagonal cluster.
    # Let's use a different approach: place them in a grid-like pattern or optimize positions.
    
    # Actually, let's think about this differently. We want to minimize the outer hexagon side length.
    # The optimal packing for hexagons inside a hexagon is typically a hexagonal lattice.
    
    # Let's try a simple approach: place the hexagons in a compact arrangement.
    # For N=11, we can use a pattern like:
    #   Row 0: 3 hexagons
    #   Row 1: 4 hexagons  
    #   Row 2: 4 hexagons
    # That's 11 total.
    
    # In a hexagonal lattice, adjacent hexagons have centers separated by sqrt(3) * side_length.
    # For unit hexagons (side=1), the center-to-center distance is sqrt(3).
    
    # Let's define the positions in a hexagonal grid coordinate system.
    # Using axial coordinates (q, r) for a hexagonal grid.
    
    # For a compact arrangement of 11 hexagons:
    # Let's use a pattern that minimizes the bounding hexagon.
    
    # A good arrangement for 11 hexagons:
    # Center + 6 neighbors + 4 more to make 11.
    # Or we can use a more systematic approach.
    
    # Let's place them in a hexagonal lattice pattern.
    # The key insight: if we place unit hexagons in a hexagonal lattice with all aligned (angle=0),
    # the centers form a triangular lattice with spacing sqrt(3).
    
    # For a hexagonal cluster of "radius" r, the outer hexagon has side length:
    # L = (2*r + 1) * sqrt(3) / 2 + 1  (approximately, needs careful calculation)
    
    # Actually, let's compute this properly.
    # A regular hexagon with side length s has width 2*s and height sqrt(3)*s.
    # When we pack unit hexagons in a hexagonal arrangement, the centers are at distance sqrt(3) apart.
    
    # Let's try a specific arrangement and compute the required outer hexagon size.
    
    # Arrangement: use a hexagonal grid with centers at:
    # For row i (0-indexed from top), the y-coordinate is -i * sqrt(3)/2
    # The x-coordinates depend on the row parity.
    
    # Let's just place them in a compact cluster and compute the minimum enclosing hexagon.
    
    # Simple approach: place N hexagons in a hexagonal lattice pattern.
    # For N=11, use the following positions (in units of sqrt(3) for x-spacing):
    
    # Let's use axial coordinates and convert to Cartesian.
    # Axial to Cartesian: x = sqrt(3) * (q + r/2), y = 3/2 * r
    
    # For a compact cluster of 11:
    # Let's use a pattern like:
    # q=0,r=0 (center)
    # q=1,r=0; q=0,r=1; q=-1,r=1; q=-1,r=0; q=0,r=-1; q=1,r=-1 (6 neighbors)
    # That's 7. Need 4 more.
    # Add: q=2,r=0; q=1,r=1; q=-1,r=2; q=-2,r=1 (or similar)
    
    # Let's just use a simple grid and compute the answer.
    
    # Actually, let's use a more direct approach:
    # Place the hexagons in a hexagonal lattice and find the minimum outer hexagon.
    
    # For simplicity, let's use a known good arrangement.
    # Place 11 hexagons in a pattern that fits in a hexagon of side ~3.5 to 4.0.
    
    # Let's compute: if we have a hexagonal cluster of radius 2 (19 hexagons), 
    # the outer hexagon side would be about (2*2+1)*sqrt(3)/2 + 1 = 5*sqrt(3)/2 + 1 ≈ 4.33 + 1 = 5.33
    # That's too many. For 11, we need something smaller.
    
    # Let's try placing them in a 3-4-4 pattern (rows of 3, 4, 4 hexagons).
    
    # Row 0 (top): 3 hexagons at y = 0
    # Row 1: 4 hexagons at y = -sqrt(3)/2
    # Row 2: 4 hexagons at y = -sqrt(3)
    
    # x-coordinates:
    # Row 0: x = -sqrt(3), 0, sqrt(3)
    # Row 1: x = -1.5*sqrt(3), -0.5*sqrt(3), 0.5*sqrt(3), 1.5*sqrt(3)
    # Row 2: x = -sqrt(3), 0, sqrt(3), 2*sqrt(3)  -- wait, this doesn't look right.
    
    # Let me use a proper hexagonal grid.
    # In a hexagonal grid with pointy-top orientation:
    # The horizontal spacing between adjacent columns is sqrt(3).
    # The vertical spacing between rows is 1.5.
    # Odd rows are offset by sqrt(3)/2.
    
    # Let's use flat-top orientation for simplicity.
    # Flat-top hexagon: width = 2*side, height = sqrt(3)*side.
    # For side=1: width=2, height=sqrt(3).
    
    # Centers of adjacent flat-top hexagons:
    # Horizontal neighbors: distance 1.5*side = 1.5
    # Diagonal neighbors: distance sqrt(3)*side = sqrt(3)
    
    # Let's just place them in a simple pattern and compute.
    
    # I'll use a hexagonal lattice with centers at:
    # (i * 1.5, j * sqrt(3) + (i % 2) * sqrt(3)/2) for integer i, j
    
    # Let's pick 11 positions that form a compact cluster.
    
    positions = []
    # Center
    positions.append((0, 0))
    # 6 immediate neighbors in hexagonal lattice
    neighbors = [
        (1.5, 0),          # right
        (0.75, sqrt3/2),   # upper right
        (-0.75, sqrt3/2),  # upper left
        (-1.5, 0),         # left
        (-0.75, -sqrt3/2), # lower left
        (0.75, -sqrt3/2),  # lower right
    ]
    for p in neighbors:
        positions.append(p)
    # That's 7. Need 4 more.
    # Add next ring partially
    more = [
        (3.0, 0),
        (2.25, sqrt3/2),
        (-2.25, sqrt3/2),
        (-3.0, 0),
    ]
    for p in more:
        positions.append(p)
    
    # That's 11 positions.
    
    # Now compute the minimum outer hexagon that contains all these unit hexagons.
    # For simplicity, assume the outer hexagon is axis-aligned (angle=0) and centered at origin.
    # We need to find the minimum side length s such that all unit hexagons fit inside.
    
    # A unit hexagon centered at (cx, cy) with angle 0 (flat-top) has vertices at:
    # (cx + 1, cy), (cx + 0.5, cy + sqrt(3)/2), (cx - 0.5, cy + sqrt(3)/2),
    # (cx - 1, cy), (cx - 0.5, cy - sqrt(3)/2), (cx + 0.5, cy - sqrt(3)/2)
    
    # The outer hexagon with side length s, centered at origin, angle 0, has vertices at:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    
    # To check if a point is inside the outer hexagon, we can use the fact that a regular hexagon
    # with side s centered at origin contains all points (x,y) such that:
    # |x| <= s and |y| <= s*sqrt(3)/2 and |x + y/sqrt(3)| <= s and |x - y/sqrt(3)| <= s
    
    # Actually, the half-width in x-direction is s, and half-height in y-direction is s*sqrt(3)/2.
    # But it's more complex due to the hexagonal shape.
    
    # A regular hexagon with side s, flat-top, centered at origin:
    # The constraints are:
    # -s <= x <= s
    # -s*sqrt(3)/2 <= y <= s*sqrt(3)/2
    # And the six sides create additional constraints.
    
    # For a point (x,y) to be inside a flat-top regular hexagon with side s centered at origin:
    # |x| <= s
    # |y| <= s * sqrt(3) / 2
    # |x + y/sqrt(3)| <= s  -- this isn't quite right.
    
    # Let me think again. The six edges of the hexagon are:
    # 1. x = s (right vertical edge, from y=-s*sqrt(3)/2 to y=s*sqrt(3)/2)
    # 2. x/2 + y*sqrt(3)/2 = s (upper right edge)
    # 3. -x/2 + y*sqrt(3)/2 = s (upper left edge)
    # 4. x = -s (left vertical edge)
    # 5. -x/2 - y*sqrt(3)/2 = s (lower left edge)
    # 6. x/2 - y*sqrt(3)/2 = s (lower right edge)
    
    # So the constraints are:
    # x <= s
    # -x <= s
    # x/2 + y*sqrt(3)/2 <= s
    # -x/2 + y*sqrt(3)/2 <= s
    # -x/2 - y*sqrt(3)/2 <= s
    # x/2 - y*sqrt(3)/2 <= s
    
    # Equivalently:
    # |x| <= s
    # |x/2 + y*sqrt(3)/2| <= s
    # |-x/2 + y*sqrt(3)/2| <= s
    
    # Or:
    # |x| <= s
    # |x + y*sqrt(3)| <= 2*s
    # |-x + y*sqrt(3)| <= 2*s
    
    # Let's verify: for the point (s, 0): |s| = s <= s OK, |s + 0| = s <= 2s OK, |-s + 0| = s <= 2s OK.
    # For (s/2, s*sqrt(3)/2): |s/2| = s/2 <= s OK, |s/2 + s*sqrt(3)/2*sqrt(3)| = |s/2 + 3s/2| = 2s <= 2s OK, |-s/2 + s*sqrt(3)/2*sqrt(3)| = |-s/2 + 3s/2| = s <= 2s OK.
    
    # Good. So to check if all vertices of all inner hexagons are inside the outer hexagon,
    # we need: for each inner hexagon vertex (vx, vy):
    # |vx| <= s
    # |vx + vy*sqrt(3)| <= 2*s
    # |-vx + vy*sqrt(3)| <= 2*s
    
    # This is equivalent to: s >= max over all vertices of max(|vx|, |vx + vy*sqrt(3)|/2, |-vx + vy*sqrt(3)|/2)
    
    sqrt3 = np.sqrt(3)
    
    min_s = 0
    for (cx, cy) in positions:
        # Vertices of unit flat-top hexagon centered at (cx, cy)
        vertices = [
            (cx + 1, cy),
            (cx + 0.5, cy + sqrt3/2),
            (cx - 0.5, cy + sqrt3/2),
            (cx - 1, cy),
            (cx - 0.5, cy - sqrt3/2),
            (cx + 0.5, cy - sqrt3/2),
        ]
        for (vx, vy) in vertices:
            s_req = max(abs(vx), abs(vx + vy*sqrt3)/2, abs(-vx + vy*sqrt3)/2)
            min_s = max(min_s, s_req)
    
    # The outer hexagon side length is min_s.
    # Center at (0,0), angle 0.
    
    inner = [[x, y, 0.0] for (x, y) in positions]
    center = (0.0, 0.0)
    side = min_s
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
