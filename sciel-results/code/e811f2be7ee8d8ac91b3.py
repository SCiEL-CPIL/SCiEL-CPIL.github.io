"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import itertools

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place N=11 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    
    Strategy: Use a 3x3 hexagonal lattice block (9 positions) plus 2 additional positions
    to fit 11 hexagons. The minimal enclosing hexagon for a tight cluster of 11 unit hexagons
    can be found by arranging them in a compact hexagonal pattern.
    
    For 11 hexagons, the optimal arrangement is a hexagonal number pattern:
    - A hexagon of side length 2 has 1+6+12 = 19 positions (too many)
    - A hexagon of side length 1 has 1 position
    - We need 11 positions
    
    The minimal enclosing hexagon side length for 11 unit hexagons:
    If we arrange them in a "flower" pattern (1 center + 6 surrounding + 4 more),
    the enclosing hexagon needs to be large enough to contain all of them.
    
    Let's think about this differently. The unit hexagons have side length 1, so their
    circumradius (center to vertex) is 1, and their inradius (center to edge) is sqrt(3)/2.
    
    For a tight packing of hexagons on a hexagonal lattice, the centers are at distance
    sqrt(3) apart (since the inradius is sqrt(3)/2, and two adjacent hexagons touch edge-to-edge,
    so center-to-center distance is 2 * sqrt(3)/2 = sqrt(3)).
    
    Axial coordinates for hexagonal grid:
    - (0,0): center
    - The 6 neighbors are at axial coords: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    
    For 11 hexagons, a good arrangement is:
    Row -1: positions at y=-1
    Row 0: positions at y=0
    Row 1: positions at y=1
    
    Let's use axial coordinates (q, r) where the actual position is:
    x = sqrt(3) * (q + r/2)
    y = 1.5 * r
    
    A compact arrangement of 11 hexagons:
    - Center: (0,0)
    - 6 neighbors: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    - 4 more: (2,0), (2,-1), (0,-2), (-2,1) or some other combination
    
    Actually, let me just compute the minimal enclosing hexagon for a given set of centers.
    
    The outer hexagon must contain all 11 unit hexagons. A unit hexagon centered at (cx, cy)
    with rotation angle theta occupies a region. The outer hexagon centered at (ox, oy) with
    side length S and rotation phi must contain all these regions.
    
    For a regular hexagon with side length S, the distance from center to each edge (inradius)
    is S * sqrt(3) / 2, and the distance from center to each vertex (circumradius) is S.
    
    For the outer hexagon to contain a unit hexagon centered at (cx, cy), we need the outer
    hexagon to be large enough. If both hexagons have the same orientation, the condition is
    that the distance between centers plus the inradius of the inner hexagon is <= the inradius
    of the outer hexagon... no, that's not quite right.
    
    Actually, for two concentric hexagons with the same orientation, the inner one fits if its
    circumradius <= the outer one's inradius... no.
    
    Let me think more carefully. A regular hexagon with side length s has:
    - Circumradius R = s
    - Inradius r = s * sqrt(3) / 2
    
    For the outer hexagon (side S, center O) to contain the inner hexagon (side 1, center C),
    we need every point of the inner hexagon to be inside the outer hexagon.
    
    The farthest point of the inner hexagon from O is at distance |OC| + R_inner = |OC| + 1.
    But the outer hexagon's boundary is not a circle; it's a hexagon.
    
    For hexagons with the same orientation, the condition for containment is:
    The inner hexagon's vertices must all be inside or on the outer hexagon.
    
    A simpler approach: if we align all hexagons (same rotation), the problem reduces to
    finding the minimal S such that the outer hexagon contains all inner hexagons.
    
    For a regular hexagon with side length S and center at origin, oriented with a flat top,
    the six edges are at distance S*sqrt(3)/2 from the center, in directions 0, 60, 120, 180, 240, 300 degrees.
    
    A point (x, y) is inside this hexagon if and only if:
    |x| <= S
    |y| <= S * sqrt(3) / 2
    |x/2 + y*sqrt(3)/2| <= S * sqrt(3) / 2  ... this is getting complicated.
    
    Let me use a different formulation. A regular hexagon centered at origin with circumradius R
    (side length R) and oriented with vertices at angles 0, 60, 120, 180, 240, 300 has the property
    that a point (x, y) is inside if and only if for each of the 6 edge normals n_i, the projection
    of the point onto n_i is <= R * sqrt(3)/2.
    
    Actually, let me just use a numerical approach. I'll try a few compact arrangements and
    compute the minimal enclosing hexagon side length for each.
    
    For now, let me use a known result: the minimal side length for packing 11 unit hexagons
    in a regular hexagon is approximately 4.0 or so. Let me just return a reasonable arrangement.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Hexagonal lattice positions for 11 hexagons
    # Using axial coordinates (q, r), convert to Cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    sqrt3 = math.sqrt(3)
    
    # Define 11 positions in a compact hexagonal arrangement
    # A "super-hexagon" of radius 1 has 7 positions (center + 6 neighbors)
    # Add 4 more positions to make 11
    
    axial_positions = [
        (0, 0),      # center
        (1, 0),      # right
        (1, -1),     # upper right
        (0, -1),     # up
        (-1, 0),     # left
        (-1, 1),     # lower left
        (0, 1),      # down
        (2, 0),      # far right
        (2, -1),     # far upper right
        (-2, 0),     # far left
        (0, 2),      # far down
    ]
    
    # Convert to Cartesian coordinates
    centers = []
    for q, r in axial_positions:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    # Compute the bounding box to center the arrangement
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    
    # Center the arrangement at origin
    centers = [(x - cx, y - cy) for x, y in centers]
    
    # Now compute the minimal enclosing hexagon side length
    # For a hexagon centered at origin with side length S and flat-top orientation,
    # a point (x, y) is inside if:
    # |x| <= S
    # |y| <= S * sqrt(3) / 2
    # |x/2 + y*sqrt(3)/2| <= S * sqrt(3) / 2
    # |-x/2 + y*sqrt(3)/2| <= S * sqrt(3) / 2
    # etc.
    
    # Actually, for a regular hexagon with circumradius S (side length S) centered at origin,
    # oriented with a vertex at angle 0 (pointing right), the six edge normals are at
    # angles 30, 90, 150, 210, 270, 330 degrees, and the inradius is S * sqrt(3)/2.
    
    # A point (x, y) is inside if for each edge normal n_i, dot((x,y), n_i) <= S * sqrt(3)/2
    
    # Edge normals for a hexagon with vertex at angle 0:
    # The edges are perpendicular to directions 30, 90, 150, 210, 270, 330 degrees
    edge_normals = []
    for i in range(6):
        angle = math.radians(30 + i * 60)
        edge_normals.append((math.cos(angle), math.sin(angle)))
    
    # For each inner hexagon centered at (cx_i, cy_i) with side length 1,
    # the farthest point in direction n is at distance (cx_i, cy_i) dot n + 1 * sqrt(3)/2
    # (since the inradius of the unit hexagon is sqrt(3)/2)
    
    # Wait, the unit hexagon's extent in direction n is its inradius if n is perpendicular to an edge,
    # or its circumradius if n points to a vertex. For a regular hexagon, the support function in
    # direction n is:
    # h(n) = max over vertices of dot(vertex, n)
    # For a unit hexagon centered at origin with vertex at angle 0, vertices are at angles 0, 60, ..., 300
    # with radius 1.
    
    # The support function of a regular hexagon with circumradius R and vertex at angle 0 in direction
    # at angle theta is:
    # h(theta) = R * cos(theta mod 60 - 30) / cos(30) ... this is getting complex.
    
    # Simpler: for the outer hexagon to contain the inner hexagon, we need the outer hexagon's
    # inradius in each of the 6 edge-normal directions to be >= the inner hexagon's support
    # in that direction plus the center offset.
    
    # For a unit hexagon with the same orientation as the outer hexagon, the support in the
    # direction of the i-th edge normal is exactly the inradius = sqrt(3)/2.
    
    # So the condition is: for each edge normal n_i,
    # dot(center_i, n_i) + sqrt(3)/2 <= S * sqrt(3) / 2
    
    # Therefore: S >= (dot(center_i, n_i) + sqrt(3)/2) / (sqrt(3)/2)
    
    # Taking the max over all centers and all edge normals gives the minimal S.
    
    min_S = 0.0
    for cx_i, cy_i in centers:
        for nx, ny in edge_normals:
            dot = cx_i * nx + cy_i * ny
            required_S = (dot + sqrt3 / 2.0) / (sqrt3 / 2.0)
            if required_S > min_S:
                min_S = required_S
    
    # Ensure S is at least 1 (to contain a single unit hexagon)
    min_S = max(min_S, 1.0)
    
    # Build the inner hexagon positions
    inner = [[cx_i, cy_i, 0.0] for cx_i, cy_i in centers]
    
    # Outer hexagon center is at origin
    outer_center = (0.0, 0.0)
    outer_side = min_S
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
