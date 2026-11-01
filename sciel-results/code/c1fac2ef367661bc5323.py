"""Program for packing 11 unit regular hexagons inside a regular hexagon."""
import numpy as np
from itertools import combinations

N = 11

def construct():
    # Directions in the hexagonal lattice (axial coordinates)
    # Neighbors of (q, r): (q+1, r), (q, r+1), (q-1, r-1), (q-1, r), (q, r-1), (q+1, r-1)
    dirs = [(1, 0), (0, 1), (-1, -1), (-1, 0), (0, -1), (1, -1)]
    
    # Generate lattice points by distance from origin
    # dist((q, r)) = (|q| + |r| + |q+r|) / 2
    # We need 11 points: 1 at dist 0, 6 at dist 1, 4 at dist 2
    # Let's generate all points with dist <= 2 and sort them
    
    points = []
    # Generate a sufficient range
    for q in range(-3, 4):
        for r in range(-3, 4):
            s = -q - r
            d = (abs(q) + abs(r) + abs(s)) // 2
            if d <= 2:
                points.append((d, q, r))
    
    points.sort()
    
    # Group by distance
    from collections import defaultdict
    by_dist = defaultdict(list)
    for d, q, r in points:
        by_dist[d].append((q, r))
    
    # dist 0: 1 point
    # dist 1: 6 points
    # dist 2: 12 points
    
    # We need 11 points total.
    # Take all dist 0 (1) and all dist 1 (6) = 7 points.
    # Need 4 more from dist 2 (12 points).
    
    base_points = by_dist[0] + by_dist[1]
    candidates = by_dist[2]
    
    # Helper: convert axial (q, r) to Cartesian (x, y) for a unit hexagon
    # For a pointy-top hexagon with side length 1:
    # x = 3/2 * q
    # y = sqrt(3) * (r + q/2)
    # But wait, standard axial to Cartesian for pointy-top:
    # x = 3/2 * q
    # y = sqrt(3) * (r + q/2)
    # Let's verify the distance between neighbors.
    # (1,0) -> (1.5, sqrt(3)/2). (0,1) -> (0, sqrt(3)).
    # Dist between (0,0) and (1,0): sqrt(2.25 + 0.75) = sqrt(3). Correct.
    
    sqrt3 = np.sqrt(3)
    
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        return (x, y)
    
    # Helper: compute the minimum side length S of a regular hexagon centered at (cx, cy) 
    # with rotation angle theta that contains all given points (centers of unit hexagons).
    # The unit hexagons have side 1. The container hexagon has side S.
    # A point (x, y) relative to center is (dx, dy).
    # In the container's local frame (rotated by -theta), the point is (dx', dy').
    # The condition for being inside a regular hexagon of side S centered at origin:
    # max(|dx'|, |dy'|, |dx'/2 - dy'*sqrt(3)/2|, |-dx'/2 - dy'*sqrt(3)/2|, |dx' - dy'*sqrt(3)|, |dx' + dy'*sqrt(3)|) ... no, that's not right.
    
    # Let's use the distance to the sides.
    # A regular hexagon of side S can be defined as the intersection of 6 half-planes.
    # The apothem (distance from center to side) is a = S * sqrt(3) / 2.
    # The normals to the sides are at angles 0, 60, 120, 180, 240, 300 degrees (for flat-top) 
    # or 30, 90, 150, 210, 270, 330 (for pointy-top).
    # Let's assume the container is pointy-top as well, but rotated by theta.
    # The normals are at angles theta + 30 + 60*k for k=0..5.
    # For a point (x, y) to be inside, the projection onto each normal must be <= a + 1?
    # Wait, the inner hexagons have side 1, so they extend 1/2 away from their centers in all directions?
    # No, the maximum distance from the center of a unit hexagon to its boundary is the circumradius R = 1.
    # So the condition is that the distance from the container center to the point (center of inner hex) 
    # plus the circumradius of the inner hex (which is 1) must be <= the apothem of the container?
    # No, that's for circles. For hexagons, it's more complex.
    # However, if the inner hexagons are aligned with the container (same rotation), then the condition is simpler.
    # Let's assume the inner hexagons are aligned with the container (angle 0).
    # Then the "radius" of the inner hexagon in the direction of the container's side normal is the apothem of the inner hex, which is sqrt(3)/2.
    # So the condition is: projection of (center - container_center) onto normal + sqrt(3)/2 <= S * sqrt(3) / 2.
    # This means projection <= (S-1) * sqrt(3) / 2.
    # This is true IF the inner hexagons are aligned.
    # If they are rotated, the effective radius in the normal direction is the support function of the hexagon.
    # For a regular hexagon of side 1, the support function h(u) in direction u is:
    # h(u) = 1 if u is in the direction of a vertex, and sqrt(3)/2 if u is in the direction of a side.
    # The maximum support is 1 (circumradius), minimum is sqrt(3)/2 (apothem).
    # To be safe, we can use the circumradius 1. Then the condition is:
    # projection <= S * sqrt(3)/2 - 1.
    # This is conservative but valid.
    
    # Let's try to optimize for aligned hexagons first.
    # If aligned, the condition is:
    # |proj| <= (S-1) * sqrt(3) / 2
    # So S >= 1 + 2 * max|proj| / sqrt(3)
    
    def min_side_length(points_cart, cx, cy, theta):
        # theta is rotation of the container.
        # Points are in global frame.
        # Transform points to container local frame.
        # Rotate by -theta.
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        
        max_proj = 0
        # Normals for a pointy-top hexagon are at 30, 90, 150, 210, 270, 330 degrees.
        # In radians: pi/6 + k*pi/3
        normals = [(np.cos(np.pi/6 + k*np.pi/3), np.sin(np.pi/6 + k*np.pi/3)) for k in range(6)]
        
        for (x, y) in points_cart:
            dx = x - cx
            dy = y - cy
            # Rotate
            dx_local = dx * cos_t - dy * sin_t
            dy_local = dx * sin_t + dy * cos_t
            
            for nx, ny in normals:
                proj = abs(dx_local * nx + dy_local * ny)
                if proj > max_proj:
                    max_proj = proj
        
        # S >= 1 + 2 * max_proj / sqrt(3)
        S = 1 + 2 * max_proj / sqrt3
        return S
    
    # We will search over the 4 points to choose from dist 2.
    # We also need to choose the container center (cx, cy) and rotation theta.
    # To minimize S, we should center the container at the centroid of the inner hexagons?
    # Or solve for the optimal center and rotation.
    # For a fixed set of points, the optimal center and rotation can be found by optimization.
    # However, since the number of points is small (11), we can use a simple grid search or optimization.
    
    best_S = float('inf')
    best_config = None
    
    # Iterate over combinations of 4 points from dist 2
    for combo in combinations(range(len(candidates)), 4):
        selected_indices = [0, 1, 2, 3, 4, 5, 6] + [7+i for i in combo] # This indexing is wrong
        
        # Let's just construct the list of points
        points_axial = by_dist[0] + by_dist[1]
        for i in combo:
            points_axial.append(candidates[i])
        
        points_cart = [axial_to_cartesian(q, r) for (q, r) in points_axial]
        
        # Optimize center and rotation
        # The optimal center is likely the centroid.
        cx = np.mean([p[0] for p in points_cart])
        cy = np.mean([p[1] for p in points_cart])
        
        # The optimal rotation is likely 0 or pi/6 (to align with the lattice symmetry).
        # Let's try a few rotations: 0, pi/6, pi/3, pi/2, 2pi/3, 5pi/6
        for theta in [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]:
            S = min_side_length(points_cart, cx, cy, theta)
            if S < best_S:
                best_S = S
                best_config = (points_axial, cx, cy, theta)
    
    # Extract best config
    points_axial, cx, cy, theta = best_config
    
    # Construct the output
    inner = []
    for (q, r) in points_axial:
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, 0.0]) # Angle 0.0 for inner hexagons (aligned with lattice)
    
    # Convert theta to degrees
    theta_deg = np.degrees(theta)
    
    # Return inner, (cx, cy), best_S, theta_deg
    return inner, (cx, cy), best_S, theta_deg

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
