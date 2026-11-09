import numpy as np
import math
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Unit regular hexagon with side 1, flat-top orientation
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees from center
    # For flat-top: vertices at 0, 60, 120, 180, 240, 300
    # Inradius (center to edge) = sqrt(3)/2
    # Circumradius (center to vertex) = 1
    
    # Two unit hexagons with same orientation are disjoint iff center distance >= sqrt(3)
    # (minimum when they share an edge)
    
    # Place 11 unit hexagons in a honeycomb pattern using axial coordinates
    # Axial to Cartesian for flat-top hexagons:
    # x = size * (3/2 * q)
    # y = size * (sqrt(3)/2 * q + sqrt(3) * r)
    # where size = 1 (side length), and center spacing is sqrt(3)
    
    # Actually, for unit hexagons in a honeycomb, the center-to-center distance
    # in adjacent directions is sqrt(3).
    # Let me use a coordinate system where adjacent hex centers are sqrt(3) apart.
    
    # Axial coordinates (q, r) -> Cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # Wait, let me be more careful.
    
    # For a regular hexagon with side length s, flat-top orientation:
    # The distance between centers of adjacent hexagons (sharing an edge) is 2 * inradius = 2 * (sqrt(3)/2) * s = sqrt(3) * s
    # For s=1, this is sqrt(3).
    
    # Axial to pixel for flat-top hexagons with side length 1:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # This gives center spacing of sqrt(3) horizontally and sqrt(3) in the 60-degree direction.
    
    # Let me verify: (q,r) and (q+1,r): dx = sqrt(3), dy = 0. Distance = sqrt(3). Good.
    # (q,r) and (q,r+1): dx = sqrt(3)/2, dy = 3/2. Distance = sqrt(3/4 + 9/4) = sqrt(12/4) = sqrt(3). Good.
    
    sqrt3 = math.sqrt(3)
    
    def axial_to_cart(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    # Generate positions for 11 hexagons in a compact arrangement
    # Start with center (0,0), then ring 1 (6 hexagons), then 4 from ring 2
    
    # Ring 0: (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2: pick 4 that make a compact shape
    
    # Let me try a different approach: use a "hexagon of hexagons" shape
    # For 11 hexagons, a good shape is:
    # Row -2: 2 hexagons
    # Row -1: 3 hexagons  
    # Row  0: 3 hexagons
    # Row  1: 2 hexagons
    # Row  2: 1 hexagon
    # Total: 11
    
    # Actually, let me just use axial coordinates and pick a compact set.
    
    # Let me try: a hexagonal cluster with some extra
    # Positions in axial coords:
    positions_axial = [
        (0, 0),    # center
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),  # ring 1
        (2, 0), (2, -1), (1, -2), (0, -2),  # 4 from ring 2
    ]
    
    # Convert to Cartesian
    positions = [axial_to_cart(q, r) for q, r in positions_axial]
    
    # Center the configuration
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    positions = [(x - cx, y - cy) for x, y in positions]
    
    # Now compute the minimal enclosing regular hexagon
    # A regular hexagon with side length S, centered at (0,0), with flat-top orientation
    # has the property that a point (x,y) is inside if:
    # |x| <= S * sqrt(3) / 2  (for the horizontal extent... wait, no)
    
    # For a regular hexagon with side length S, flat-top:
    # The six sides are at distances S*sqrt(3)/2 from the center.
    # The hexagon is the intersection of 6 half-planes.
    # For flat-top orientation, the normals to the edges are at angles 0, 60, 120, 180, 240, 300 degrees.
    # Wait, for flat-top, the edges are horizontal at top and bottom, so normals are vertical.
    
    # Let me reconsider. Flat-top hexagon with side S:
    # Vertices at angles 0, 60, 120, 180, 240, 300 from center, at distance S.
    # Edge midpoints at angles 30, 90, 150, 210, 270, 330 from center, at distance S*sqrt(3)/2.
    
    # A point (x,y) is inside if for all 6 edge normals, the dot product with the point
    # is <= S*sqrt(3)/2.
    
    # Edge normals (unit vectors) for flat-top hexagon:
    # The edges are perpendicular to directions 30, 90, 150, 210, 270, 330 degrees.
    # So normals are at those angles.
    
    def min_hexagon_side(positions, center_angle_deg=0.0):
        """Find minimum side length S such that all points are inside a regular hexagon
        of side S centered at origin, with the given rotation."""
        theta = math.radians(center_angle_deg)
        # 6 edge normals
        normals = []
        for i in range(6):
            angle = theta + math.radians(60 * i + 30)  # for flat-top, normals at 30,90,150,...
            normals.append((math.cos(angle), math.sin(angle)))
        
        max_proj = 0.0
        for (x, y) in positions:
            for (nx, ny) in normals:
                proj = abs(x * nx + y * ny)
                if proj > max_proj:
                    max_proj = proj
        
        # The inradius must be at least max_proj, so S * sqrt(3)/2 >= max_proj
        # S >= 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / sqrt3
    
    # Try different rotations of the outer hexagon
    best_side = float('inf')
    best_angle = 0.0
    
    # Try angles from 0 to 60 degrees (due to 6-fold symmetry)
    for angle_deg in range(0, 61, 1):
        side = min_hexagon_side(positions, angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    # Fine-tune around the best angle
    if best_side < float('inf'):
        for angle_deg in np.linspace(best_angle - 1, best_angle + 1, 41):
            side = min_hexagon_side(positions, angle_deg)
            if side < best_side:
                best_side = side
                best_angle = angle_deg
    
    # Check deadline
    if time.time() > deadline:
        # Return current best
        inner = [[pos[0], pos[1], 0.0] for pos in positions]
        return inner, (0.0, 0.0), best_side, best_angle
    
    # Try to improve by also optimizing positions slightly
    # For now, return the lattice-based solution
    inner = [[pos[0], pos[1], 0.0] for pos in positions]
    return inner, (0.0, 0.0), best_side, best_angle

