import time
import numpy as np

N = 11
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def construct():
    """
    Place N unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Returns: inner, center, side, angle
    - inner: list of [x, y, angle_degrees] for each inner hexagon
    - center: (x, y) of outer hexagon
    - side: side length of outer hexagon
    - angle: rotation of outer hexagon (0.0)
    """
    # For N unit hexagons, we can arrange them in a hexagonal lattice.
    # The optimal arrangement for small N is a centered hexagonal cluster.
    # For N=11, this is a hexagon of radius 1 (1 center + 6 ring + 4 more? No, 1+6=7, next is 19).
    # Actually, 11 is 1 + 6 + 4. But a linear arrangement or a compact cluster is better.
    # Let's try to pack them in a hexagonal grid pattern.
    
    # The width of a hexagon of side 1 is 2 (flat-to-flat is sqrt(3)*1, point-to-point is 2).
    # Wait, regular hexagon side 1:
    # Width (point to point) = 2
    # Height (flat to flat) = sqrt(3)
    
    # If we place them in a row, the length is 2*N. The outer hexagon would need to be large.
    # A hexagonal cluster is more compact.
    
    # Let's use a hexagonal coordinate system for the centers of the unit hexagons.
    # The centers of adjacent hexagons in a honeycomb are distance sqrt(3) apart?
    # No, if they are touching, the distance between centers is 2 * (apothem) = 2 * (sqrt(3)/2) = sqrt(3).
    # So the lattice spacing is sqrt(3).
    
    # Let's generate positions for N hexagons in a compact hexagonal shape.
    # For N=11, we can do:
    # Row 0: 3 hexagons
    # Row 1: 5 hexagons
    # Row 2: 3 hexagons
    # Total: 11. This forms a hexagonal-like shape.
    
    # Or simply:
    # Center one, 6 around it, and 4 more attached to the outer ring.
    
    # Let's try a simple spiral or row packing first to get a valid score, then optimize.
    # For now, let's place them in a hexagonal grid pattern.
    
    # Hexagonal grid coordinates:
    # x = sqrt(3) * (col + row/2)
    # y = 1.5 * row  (if flat-top, but let's use pointy-top for simplicity with standard angles)
    
    # Actually, let's use the standard "axial coordinates" for hex grids.
    # Distance between centers = sqrt(3) for touching flat sides.
    
    # Let's define the positions of N hexagons in a compact cluster.
    # We'll use a list of axial coordinates (q, r).
    
    def hex_neighbors(q, r):
        return [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]
    
    # Generate a compact cluster of N hexagons
    # Start with (0,0), add neighbors in order of distance from origin
    used = set()
    queue = [(0, 0)]
    used.add((0, 0))
    positions = []
    
    while len(positions) < N and queue:
        q, r = queue.pop(0)
        positions.append((q, r))
        if len(positions) < N:
            # Add neighbors in a specific order to make it compact
            neighbors = hex_neighbors(q, r)
            # Sort by distance to origin (q^2 + q*r + r^2)
            neighbors.sort(key=lambda x: x[0]**2 + x[0]*x[1] + x[1]**2)
            for nq, nr in neighbors:
                if (nq, nr) not in used:
                    used.add((nq, nr))
                    queue.append((nq, nr))
    
    # Convert axial coordinates to Cartesian
    # For flat-top hexagons:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # But we want them to be touching, so the distance between centers should be sqrt(3).
    # Let's check: distance between (0,0) and (1,0) is sqrt(3) * 1 = sqrt(3). Correct.
    # Distance between (0,0) and (0,1) is sqrt( (sqrt(3)*0.5)^2 + (1.5*1)^2 ) = sqrt(0.75 + 2.25) = sqrt(3). Correct.
    
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in positions:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    # All inner hexagons have angle 0.0 (same orientation)
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Now, find the minimum side length of the outer hexagon that contains all these.
    # The outer hexagon is regular, centered at (cx, cy), with side length S and rotation theta.
    # We can fix the outer hexagon's rotation to 0.0 for simplicity (since the inner ones are also 0.0,
    # and the optimal outer hexagon will likely align with them).
    # The outer hexagon's vertices are at distance S from the center.
    # The apothem (distance from center to side) is S * sqrt(3) / 2.
    
    # To find the minimum S, we can compute the maximum distance from the origin to any point of the inner hexagons,
    # but since the outer hexagon is rotated, we need to check the support function.
    
    # Simpler approach: The outer hexagon of side S contains a point (x, y) if and only if
    # max over all 6 sides of the hexagon of (dot product of (x, y) with the normal of the side) <= apothem.
    
    # For a regular hexagon centered at origin with side S and rotation 0:
    # The 6 normals are at angles 0, 60, 120, 180, 240, 300 degrees.
    # Apothem = S * sqrt(3) / 2.
    
    # So, for each inner hexagon, we need to check all its vertices against the outer hexagon's constraints.
    # Or, more efficiently, we can compute the maximum "hexagonal norm" of any point in the inner hexagons.
    
    # The hexagonal norm of a point (x, y) is max(|x|, |y| * 2/sqrt(3), |(x*sqrt(3) - y)/2|) ... no, that's not right.
    
    # Let's just compute the required S by checking all vertices of all inner hexagons.
    
    # Vertices of a unit hexagon centered at (cx, cy) with angle 0:
    # (cx + 1, cy), (cx + 0.5, cy + sqrt(3)/2), (cx - 0.5, cy + sqrt(3)/2),
    # (cx - 1, cy), (cx - 0.5, cy - sqrt(3)/2), (cx + 0.5, cy - sqrt(3)/2)
    
    max_norm = 0.0
    for x, y, angle in inner:
        # For angle 0, the vertices are:
        verts = [
            (x + 1.0, y),
            (x + 0.5, y + sqrt3 / 2.0),
            (x - 0.5, y + sqrt3 / 2.0),
            (x - 1.0, y),
            (x - 0.5, y - sqrt3 / 2.0),
            (x + 0.5, y - sqrt3 / 2.0)
        ]
        for vx, vy in verts:
            # Compute the hexagonal norm: max of the 6 linear forms
            # The 6 normals are at angles 0, 60, 120, 180, 240, 300
            # n0 = (1, 0) -> dot = vx
            # n1 = (0.5, sqrt(3)/2) -> dot = 0.5*vx + sqrt(3)/2*vy
            # n2 = (-0.5, sqrt(3)/2) -> dot = -0.5*vx + sqrt(3)/2*vy
            # n3 = (-1, 0) -> dot = -vx
            # n4 = (-0.5, -sqrt(3)/2) -> dot = -0.5*vx - sqrt(3)/2*vy
            # n5 = (0.5, -sqrt(3)/2) -> dot = 0.5*vx - sqrt(3)/2*vy
            
            norm_val = max(
                abs(vx),
                abs(0.5 * vx + sqrt3 / 2.0 * vy),
                abs(-0.5 * vx + sqrt3 / 2.0 * vy)
            )
            if norm_val > max_norm:
                max_norm = norm_val
    
    # The apothem must be at least max_norm.
    # Apothem = S * sqrt(3) / 2
    # So S = max_norm * 2 / sqrt(3)
    
    side = max_norm * 2.0 / sqrt3
    
    # The center of the outer hexagon should be at the centroid of the inner hexagons for optimality.
    # For now, let's use (0, 0) as the center.
    center = (0.0, 0.0)
    
    return inner, center, side, 0.0

# Test the construct function
if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side length: {side}")
    print(f"Number of inner hexagons: {len(inner)}")
