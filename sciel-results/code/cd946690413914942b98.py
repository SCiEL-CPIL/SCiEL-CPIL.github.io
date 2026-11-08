"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin
    
    # Hexagonal lattice arrangement for 11 unit hexagons
    # Using a 3-2-3-2-3 staggered pattern (like a hexagonal cluster)
    # Centers on triangular lattice with spacing 2 (for unit hexagons, non-overlap distance)
    
    # For axis-aligned unit hexagons, non-overlap requires center distance >= 2
    # in the hexagonal norm sense. On triangular lattice with spacing 2:
    # neighbors at (2,0), (1, sqrt(3)), (-1, sqrt(3)), (-2,0), (-1,-sqrt(3)), (1,-sqrt(3))
    
    sqrt3 = math.sqrt(3)
    
    # Generate 11 centers in a compact hexagonal arrangement
    # Row 0: 3 hexagons
    # Row 1: 2 hexagons (staggered)
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons (staggered)
    # Row 4: 3 hexagons
    # This gives 3+2+3+2+3 = 13, too many. Let's use a different arrangement.
    
    # Better: use a central hexagon with 6 around it, plus 4 more
    # Or: 2-3-2-3-1 or similar
    
    # Let me try: arrange in rows of 3, 2, 3, 2, 1 = 11
    # Or more symmetric: 3, 2, 3, 2, 1
    
    # Actually, let's use the standard hexagonal number arrangement:
    # Center at (0,0), then 6 at distance 2, then more at distance 4
    # That's 1 + 6 + 12 = 19 for full hex, too many.
    
    # For 11: center + 6 ring + 4 from second ring
    # Or simply: rows of 3, 2, 3, 2, 1 staggered
    
    # Let me just use a simple grid-like arrangement and optimize
    # Row-based: y positions at 0, sqrt(3), 2*sqrt(3), 3*sqrt(3), 4*sqrt(3)
    # x positions staggered
    
    centers = []
    # Row 0 (y=0): 3 hexagons at x = -2, 0, 2
    for x in [-2.0, 0.0, 2.0]:
        centers.append((x, 0.0))
    # Row 1 (y=sqrt(3)): 2 hexagons at x = -1, 1
    for x in [-1.0, 1.0]:
        centers.append((x, sqrt3))
    # Row 2 (y=2*sqrt(3)): 3 hexagons at x = -2, 0, 2
    for x in [-2.0, 0.0, 2.0]:
        centers.append((x, 2*sqrt3))
    # Row 3 (y=3*sqrt(3)): 2 hexagons at x = -1, 1
    for x in [-1.0, 1.0]:
        centers.append((x, 3*sqrt3))
    # Row 4 (y=4*sqrt(3)): 1 hexagon at x = 0
    centers.append((0.0, 4*sqrt3))
    
    # That's 3+2+3+2+1 = 11 centers
    
    # Now compute the minimum side length S of outer hexagon (centered at origin, axis-aligned)
    # that contains all vertices of all inner hexagons.
    
    # Vertices of a unit hexagon centered at (cx, cy) with angle 0:
    # (cx + cos(k*pi/3), cy + sin(k*pi/3)) for k=0..5
    
    def hex_vertices(cx, cy):
        verts = []
        for k in range(6):
            angle = k * math.pi / 3
            verts.append((cx + math.cos(angle), cy + math.sin(angle)))
        return verts
    
    # For outer hexagon centered at origin with side S, axis-aligned:
    # A point (x, y) is inside if:
    #   |x| <= S
    #   |y| <= S * sqrt(3) / 2
    #   |x/2 + y*sqrt(3)/2| <= S
    
    def required_S_for_point(x, y):
        # Find minimum S such that point is inside outer hexagon
        # Constraint 1: |x| <= S  =>  S >= |x|
        # Constraint 2: |y| <= S*sqrt(3)/2  =>  S >= |y| * 2/sqrt(3)
        # Constraint 3: |x/2 + y*sqrt(3)/2| <= S  =>  S >= |x/2 + y*sqrt(3)/2|
        s1 = abs(x)
        s2 = abs(y) * 2.0 / sqrt3
        s3 = abs(x/2.0 + y * sqrt3/2.0)
        return max(s1, s2, s3)
    
    max_S = 0.0
    for (cx, cy) in centers:
        for (vx, vy) in hex_vertices(cx, cy):
            S_req = required_S_for_point(vx, vy)
            if S_req > max_S:
                max_S = S_req
    
    # The outer hexagon center is at (0,0), side length is max_S, angle 0
    inner = [[cx, cy, 0.0] for (cx, cy) in centers]
    outer_center = (0.0, 0.0)
    outer_side = max_S
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
