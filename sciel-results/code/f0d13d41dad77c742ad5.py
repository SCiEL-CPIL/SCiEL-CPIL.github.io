"""Optimized hexagon packing: unit regular hexagons inside a larger regular hexagon."""
import numpy as np
from scipy.spatial.distance import cdist

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place N unit regular hexagons inside a regular hexagon, minimizing outer side length.
    
    Strategy:
    - Use a hexagonal lattice arrangement for inner hexagons.
    - The optimal packing for regular hexagons of side 1 is on a hexagonal grid.
    - For N hexagons arranged in a hexagonal pattern, the bounding hexagon side length
      can be computed geometrically.
    
    For N hexagons on a hexagonal lattice:
    - The minimal bounding hexagon for a hexagonal cluster of side k (containing 1 + 3k(k-1) hexagons)
      has side length 1 + 2k * (sqrt(3)/2 + 1/2) ... let me think more carefully.
    
    Actually, let's think about this more carefully.
    
    A regular hexagon of side s has:
    - Width (flat-to-flat): s * sqrt(3)
    - Height (point-to-point): 2 * s
    - Distance from center to vertex: s
    - Distance from center to edge: s * sqrt(3) / 2
    
    For unit hexagons (side 1) on a hexagonal lattice:
    - The lattice vectors are at 60 degrees to each other.
    - If we place hexagons at positions on a hexagonal grid, the distance between adjacent
      hexagon centers in the "flat" direction is sqrt(3) (flat-to-flat distance).
    
    For a hexagonal cluster of "radius" r (where r=0 is just 1 hexagon, r=1 is 7 hexagons, etc.):
    - Total hexagons = 1 + 3*r*(r+1)
    - The outer hexagon that contains all these unit hexagons needs to be large enough.
    
    For r hexagons radius, the centers extend to distance r * sqrt(3) from center in the
    flat directions. The unit hexagons themselves extend sqrt(3)/2 from their centers in
    the flat direction and 1 from their centers in the point direction.
    
    So the outer hexagon side length S needs:
    - In the flat direction: r * sqrt(3) + sqrt(3)/2 <= S * sqrt(3) / 2
      => S >= 2r + 1
    - In the point direction: r * 1 + 1 <= S
      => S >= r + 1
    
    Wait, I need to be more careful. Let me reconsider.
    
    A regular hexagon of side S has:
    - Inradius (center to edge midpoint): S * sqrt(3) / 2
    - Circumradius (center to vertex): S
    
    The unit hexagons have:
    - Inradius: sqrt(3)/2
    - Circumradius: 1
    
    On a hexagonal lattice with spacing d (center-to-center distance for non-overlapping hexagons):
    For unit hexagons on a hexagonal lattice, the minimum center-to-center distance is sqrt(3)
    (this is the flat-to-flat distance, which is the minimum distance to avoid overlap when
    aligned on the lattice).
    
    Actually, for regular hexagons of side 1, the minimum center-to-center distance to avoid
    overlap is 2 * inradius = sqrt(3) when they are edge-to-edge.
    
    So the lattice spacing is sqrt(3).
    
    For a hexagonal cluster of radius r (r layers out from center):
    - Number of hexagons: 1 + 3*r*(r+1)
    - The farthest centers are at distance r * sqrt(3) from the center (in the flat directions)
    - The unit hexagons at those positions extend sqrt(3)/2 further in the flat direction
    - So the total extent in the flat direction is r * sqrt(3) + sqrt(3)/2 = sqrt(3) * (r + 0.5)
    - The outer hexagon's inradius must be >= this: S * sqrt(3) / 2 >= sqrt(3) * (r + 0.5)
      => S >= 2r + 1
    
    In the point directions (vertex directions of the lattice):
    - The farthest centers are at distance r * (3/2) from center? No...
    
    Let me reconsider the hexagonal lattice geometry.
    
    For a hexagonal lattice with spacing a = sqrt(3):
    - The lattice points in a hexagonal arrangement of radius r have maximum distance
      from center of r * a in the "axial" directions.
    
    Actually, for a hexagonal number arrangement:
    - Layer 0: 1 hexagon at center
    - Layer 1: 6 hexagons at distance a from center
    - Layer 2: 12 hexagons, farthest at distance 2a
    - Layer r: 6r hexagons, farthest at distance r*a
    
    The farthest point of any unit hexagon from the center is:
    - Distance from center to hexagon center: r * a = r * sqrt(3)
    - Plus the circumradius of the unit hexagon: 1
    - Total: r * sqrt(3) + 1
    
    But this is the radial distance. The outer hexagon must contain all these points.
    
    For a regular hexagon of side S, the maximum distance from center to any point inside
    is S (the circumradius, at the vertices).
    
    But we need the entire unit hexagons to be inside, not just their centers.
    
    The worst case is when a unit hexagon's vertex points outward. The maximum distance
    from the outer hexagon's center to any point of a unit hexagon is:
    - Distance from outer center to unit hexagon center + circumradius of unit hexagon
    - = r * sqrt(3) + 1
    
    For this to fit inside the outer hexagon of side S, we need:
    S >= r * sqrt(3) + 1
    
    But wait, this isn't quite right either, because the outer hexagon isn't a circle.
    The constraint is that all points of all unit hexagons must be inside the outer hexagon.
    
    Let me use a different approach: for a given arrangement, compute the minimum enclosing
    regular hexagon.
    
    For simplicity, let's use the hexagonal lattice arrangement and compute the required
    outer hexagon side length numerically.
    
    For N=11, we can use a hexagonal arrangement with r=1 (7 hexagons) plus 4 more,
    or we can try other arrangements.
    
    Actually, let me just implement a numerical approach:
    1. Generate candidate positions on a hexagonal lattice.
    2. For N hexagons, find the best subset or arrangement.
    3. Compute the minimum enclosing regular hexagon side length.
    
    For efficiency, let's use the known optimal hexagonal packing.
    
    For N=11, a good arrangement is a hexagonal cluster of radius 1 (7 hexagons) 
    plus 4 additional hexagons placed optimally.
    
    Alternatively, we can use a linear or 2D grid arrangement.
    
    Let me just implement a simple but effective approach:
    - Place hexagons on a hexagonal lattice.
    - For N=11, use a compact arrangement.
    - Compute the bounding hexagon side length.
    """
    
    def hex_to_xy(q, r, size=1.0):
        """Convert axial coordinates to pixel coordinates for a hexagonal grid."""
        x = size * 3/2 * q
        y = size * (np.sqrt(3)/2 * q + np.sqrt(3) * r)
        return x, y
    
    def get_hexagon_vertices(cx, cy, side, angle_deg=0.0):
        """Get vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            vx = cx + side * np.cos(theta)
            vy = cy + side * np.sin(theta)
            vertices.append([vx, vy])
        return np.array(vertices)
    
    def point_in_hexagon(p, center, side, angle_deg=0.0):
        """Check if point p is inside a regular hexagon."""
        # Transform to hexagon's local coordinate system
        dx = p[0] - center[0]
        dy = p[1] - center[1]
        angle = np.radians(angle_deg)
        # Rotate point by -angle
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        # Check if inside regular hexagon of side 'side'
        # For a regular hexagon with flat top, the constraints are:
        # |lx| <= side
        # |ly| <= side * sqrt(3) / 2
        # |lx| + |ly| * 2 / sqrt(3) <= side  (for the slanted edges)
        # Actually, for a regular hexagon, the inradius is side * sqrt(3) / 2
        # and the circumradius is side.
        # The hexagon can be defined by 6 half-planes.
        inradius = side * np.sqrt(3) / 2
        for i in range(6):
            theta = i * np.pi / 3
            nx = np.cos(theta)
            ny = np.sin(theta)
            if lx * nx + ly * ny > inradius:
                return False
        return True
    
    def min_enclosing_hexagon_side(vertices, center=(0.0, 0.0)):
        """
        Find the minimum side length of a regular hexagon (with given center and orientation)
        that contains all given vertices.
        
        For a regular hexagon of side S centered at origin with angle 0:
        A point (x, y) is inside if for all 6 directions theta_i = i*pi/3:
        x*cos(theta_i) + y*sin(theta_i) <= S * sqrt(3) / 2
        
        So S >= max over all points and all 6 directions of (x*cos(theta_i) + y*sin(theta_i)) / (sqrt(3)/2)
        """
        inradius_factor = np.sqrt(3) / 2
        max_val = 0
        for i in range(6):
            theta = i * np.pi / 3
            nx = np.cos(theta)
            ny = np.sin(theta)
            for v in vertices:
                val = v[0] * nx + v[1] * ny
                if val > max_val:
                    max_val = val
        return max_val / inradius_factor
    
    # Generate hexagonal lattice positions
    # For N=11, use a hexagonal cluster
    # Hexagonal numbers: 1, 7, 19, 37, ...
    # 11 is between 7 and 19, so use radius 1 (7 hexagons) + 4 extra
    
    # Let's try a few arrangements and pick the best
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Arrangement 1: Hexagonal cluster of radius 1 (7 hexagons) + 4 extra
    # The 7 hexagons are at axial coords: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Extra 4 can be placed at (2,0), (1,1), (-1,2), (-2,1) or similar
    
    lattice_spacing = np.sqrt(3)  # Center-to-center distance for non-overlapping unit hexagons
    
    # Generate positions on hexagonal lattice
    positions = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = lattice_spacing * 3/2 * q
            y = lattice_spacing * (np.sqrt(3)/2 * q + np.sqrt(3) * r)
            positions.append((x, y))
    
    # Select N positions that minimize the bounding hexagon
    # For simplicity, use a greedy or precomputed arrangement
    
    # Known good arrangement for 11 hexagons:
    # Use axial coordinates:
    # Layer 0: (0,0)
    # Layer 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Layer 2 (partial): (2,0), (2,-1), (1,1), (0,2)
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, 1), (0, 2)
    ]
    
    inner_positions = []
    for q, r in axial_coords[:N]:
        x = lattice_spacing * 3/2 * q
        y = lattice_spacing * (np.sqrt(3)/2 * q + np.sqrt(3) * r)
        inner_positions.append((x, y))
    
    # Compute all vertices of all inner hexagons
    all_vertices = []
    for cx, cy in inner_positions:
        verts = get_hexagon_vertices(cx, cy, 1.0, 0.0)
        all_vertices.extend(verts.tolist())
    all_vertices = np.array(all_vertices)
    
    # Find minimum enclosing hexagon side length
    side = min_enclosing_hexagon_side(all_vertices)
    
    # The outer hexagon is centered at (0,0) with this side length
    inner = [[x, y, 0.0] for x, y in inner_positions]
    outer_center = (0.0, 0.0)
    outer_side = side
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
