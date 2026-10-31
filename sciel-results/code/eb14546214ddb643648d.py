import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 120.0
    
    # Try to pack 11 unit hexagons in a minimal regular hexagon container
    # Using a hexagonal lattice approach
    
    # A unit regular hexagon has:
    # - circumradius = 1
    # - inradius (apothem) = sqrt(3)/2
    # - width (flat-to-flat) = sqrt(3)
    # - distance between parallel sides = sqrt(3)
    
    # For hexagons on a triangular lattice with spacing sqrt(3) (touching),
    # we can arrange them in a hexagonal pattern.
    
    # The 19-hex cluster (3 rings) has centers at distance 0, sqrt(3), 2*sqrt(3) from center
    # But that's 1+6+12 = 19 hexagons.
    
    # For 11 hexagons, we can use a compact arrangement.
    # Let's try a pattern: center + 6 around + 4 more in a ring.
    
    # Actually, let's think about this differently.
    # If we place hexagons on a triangular grid with spacing sqrt(3),
    # the centers form a triangular lattice.
    
    # For a hexagonal container aligned with the same orientation:
    # The container's apothem (distance from center to side) must be >= 
    # max over all hexagons of (distance from container center to hex center + hex apothem)
    # for the side-aligned case, or we need to check all 6 sides.
    
    # Let's use a simpler approach: place hexagons in a compact hexagonal arrangement
    # and compute the minimal enclosing hexagon.
    
    # Hexagonal coordinates for 11 hexagons:
    # Ring 0: 1 hex at center
    # Ring 1: 6 hexes
    # Ring 2: 4 hexes (partial ring)
    # Total: 1 + 6 + 4 = 11
    
    # In axial coordinates (q, r) for a hex grid:
    # Ring 0: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: select 4 from the 12 positions in ring 2
    
    # Converting axial to pixel coordinates for a pointy-top hex grid:
    # x = size * (sqrt(3) * q + sqrt(3)/2 * r)
    # y = size * (3/2 * r)
    # where size = 1 for unit hexagons
    
    # But we want flat-top hexagons aligned with a flat-top container.
    # For flat-top:
    # x = size * (3/2 * q)
    # y = size * (sqrt(3) * r + sqrt(3)/2 * q)
    
    # Let's use flat-top orientation.
    # Center of each hexagon at (cx, cy).
    
    # For flat-top hexagons on a hex grid:
    # Spacing between centers in same row: 1.5 * 1 = 1.5
    # Spacing between rows: sqrt(3) * 1 = sqrt(3)
    # Odd rows offset by 0.75
    
    # Actually, let me just place them explicitly.
    
    # Compact arrangement of 11 hexagons:
    # Row 0 (y=0): 3 hexes at x = -1.5, 0, 1.5
    # Row 1 (y=sqrt(3)): 4 hexes at x = -2.25, -0.75, 0.75, 2.25
    # Row 2 (y=2*sqrt(3)): 3 hexes at x = -1.5, 0, 1.5
    # Row 3 (y=3*sqrt(3)): 1 hex at x = 0
    # Total: 3+4+3+1 = 11
    
    # Wait, that's not quite right for touching hexagons.
    # For flat-top hexagons with side length 1:
    # Horizontal distance between adjacent hexes in same row: 3/2 = 1.5
    # Vertical distance between rows: sqrt(3)
    # Odd rows are offset by 3/4 = 0.75
    
    # Let's place them:
    # Row 0 (y=0): x = -1.5, 0, 1.5 (3 hexes)
    # Row 1 (y=sqrt(3)): x = -0.75, 0.75 (2 hexes) -- offset
    # Hmm, this is getting complex. Let me just use the known compact packing.
    
    # Simple approach: use a hexagonal number pattern.
    # For 11 hexagons, the optimal is likely a 4-3-4 arrangement or similar.
    
    # Let's just place them in a rough hexagonal pattern and compute the container.
    
    sqrt3 = math.sqrt(3)
    
    # Centers of 11 unit hexagons in a compact arrangement
    # Using flat-top orientation, centers on a triangular lattice
    # with nearest-neighbor distance = sqrt(3)
    
    # Axial coordinates (col, row) -> Cartesian:
    # x = sqrt(3) * (col + row/2)
    # y = 1.5 * row
    # This is for pointy-top. For flat-top, swap appropriately.
    
    # Let's just use a known compact arrangement.
    # Place hexagons at these centers (flat-top, side=1):
    centers = [
        (0.0, 0.0),
        (sqrt3, 0.0),
        (-sqrt3, 0.0),
        (0.5*sqrt3, 1.5),
        (-0.5*sqrt3, 1.5),
        (0.5*sqrt3, -1.5),
        (-0.5*sqrt3, -1.5),
        (1.5*sqrt3, 1.5),
        (-1.5*sqrt3, 1.5),
        (1.5*sqrt3, -1.5),
        (-1.5*sqrt3, -1.5),
    ]
    
    # All hexagons have angle 0 (aligned with container)
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Now find the minimal regular hexagon container (aligned, angle=0) that contains all.
    # For a flat-top regular hexagon container with side length S and center (0,0):
    # The boundary consists of 6 sides. A point (px, py) is inside if:
    # |px| <= S * sqrt(3)/2  (for the left/right "vertical" sides... wait)
    
    # For a flat-top regular hexagon with side S centered at origin:
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees at distance S.
    # The six sides are at distances (apothem) = S * sqrt(3)/2 from center.
    
    # A point (px, py) is inside the flat-top hexagon with side S iff:
    # |py| <= S * sqrt(3)/2 - |px| / 2  ... no that's not right.
    
    # Actually, for a flat-top hexagon, the constraints are:
    # |py| <= S * sqrt(3)/2  (top and bottom flat sides)
    # AND |px| <= S  (left and right points) -- no, that's not right either.
    
    # Let me think again. A regular hexagon with side S, flat-top, centered at origin:
    # Vertices: (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
    
    # The six edges:
    # 1. Right point to upper-right: from (S,0) to (S/2, S*sqrt(3)/2)
    # 2. Upper-right to upper-left: from (S/2, S*sqrt(3)/2) to (-S/2, S*sqrt(3)/2) -- top flat side
    # 3. etc.
    
    # The top flat side is at y = S*sqrt(3)/2.
    # The right slanted side: line from (S,0) to (S/2, S*sqrt(3)/2).
    #   This line has equation: x + y/sqrt(3) = S
    #   So constraint: x + y/sqrt(3) <= S
    # Similarly for other sides.
    
    # The six constraints for a point (px, py) to be inside:
    # 1. py <= S*sqrt(3)/2  (top)
    # 2. py >= -S*sqrt(3)/2  (bottom)
    # 3. px + py/sqrt(3) <= S  (upper-right slant)
    # 4. px - py/sqrt(3) <= S  (lower-right slant)
    # 5. -px + py/sqrt(3) <= S  (upper-left slant)
    # 6. -px - py/sqrt(3) <= S  (lower-left slant)
    
    # So the minimal S such that all hexagon vertices are inside:
    # For each hexagon center (cx, cy), we need all 6 vertices of that hexagon to satisfy the constraints.
    # But actually, since the hexagons are also flat-top with side 1, the farthest point of each hexagon
    # from the container center determines the required S.
    
    # For a unit flat-top hexagon centered at (cx, cy), its vertices are at:
    # (cx + 1, cy), (cx + 0.5, cy + sqrt(3)/2), (cx - 0.5, cy + sqrt(3)/2),
    # (cx - 1, cy), (cx - 0.5, cy - sqrt(3)/2), (cx + 0.5, cy - sqrt(3)/2)
    
    # We need to find S such that for all hexagons and all their vertices, the constraints hold.
    # The required S for a given vertex (vx, vy) is:
    # S >= max(vy/sqrt(3), -vy/sqrt(3), vx + vy/sqrt(3), vx - vy/sqrt(3), -vx + vy/sqrt(3), -vx - vy/sqrt(3))
    # Simplifying: S >= max(|vx| + |vy|/sqrt(3), |vy|/sqrt(3) * something)
    
    # Actually: S >= max(vy/sqrt(3), -vy/sqrt(3), vx + vy/sqrt(3), vx - vy/sqrt(3), -vx + vy/sqrt(3), -vx - vy/sqrt(3))
    # = max(|vy|/sqrt(3), |vx| + |vy|/sqrt(3))
    # Because vx + vy/sqrt(3) and its negation give |vx + vy/sqrt(3)|, etc.
    # The maximum is actually: max over the 6 expressions.
    # Note that vx + vy/sqrt(3) and -vx - vy/sqrt(3) give |vx + vy/sqrt(3)|
    # vx - vy/sqrt(3) and -vx + vy/sqrt(3) give |vx - vy/sqrt(3)|
    # vy/sqrt(3) and -vy/sqrt(3) give |vy|/sqrt(3)
    
    # So S >= max(|vy|/sqrt(3), |vx + vy/sqrt(3)|, |vx - vy/sqrt(3)|)
    # But |vx + vy/sqrt(3)| >= |vy|/sqrt(3) when vx is positive enough, etc.
    # Actually the max of these three is what we need.
    
    def required_S(vx, vy):
        return max(abs(vy)/sqrt3, abs(vx + vy/sqrt3), abs(vx - vy/sqrt3))
    
    S = 0.0
    for cx, cy in centers:
        # Vertices of unit flat-top hexagon at (cx, cy)
        verts = [
            (cx + 1.0, cy),
            (cx + 0.5, cy + sqrt3/2),
            (cx - 0.5, cy + sqrt3/2),
            (cx - 1.0, cy),
            (cx - 0.5, cy - sqrt3/2),
            (cx + 0.5, cy - sqrt3/2),
        ]
        for vx, vy in verts:
            S = max(S, required_S(vx, vy))
    
    # Add a tiny epsilon to avoid floating point issues
    S += 1e-9
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
