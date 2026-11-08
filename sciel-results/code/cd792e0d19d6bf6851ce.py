import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    
    # Generate 11 closest lattice points in axial coordinates
    # Axial coordinates (q, r)
    # Distance d = (|q| + |r| + |q+r|) / 2
    points_axial = []
    q_range = range(-5, 6)
    for q in q_range:
        for r in q_range:
            d = (abs(q) + abs(r) + abs(q + r)) / 2.0
            points_axial.append((d, q, r))
    
    points_axial.sort(key=lambda x: x[0])
    # Take the first 11
    selected = points_axial[:N]
    
    # Convert to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    sqrt3 = np.sqrt(3)
    pts = np.array([[sqrt3 * (q + r/2.0), 1.5 * r] for _, q, r in selected])
    
    # Center the points
    pts -= pts.mean(axis=0)
    
    best_L = 100.0
    
    # Optimize rotation
    # The hexagon symmetry is 60 degrees. We can search 0 to 60.
    # However, the lattice has 60 degree symmetry too.
    # So we might just need to check a few angles or a fine grid.
    
    def min_enclosing_hexagon_side(points, theta):
        # Rotate points by -theta
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        rotated_pts = points @ R.T
        
        # For an aligned hexagon of side K, the condition for a point (x,y) to be inside
        # is equivalent to checking the hexagonal distance.
        # Or simply, the max of the 3 linear forms.
        # In the frame of the hexagon, the boundaries are:
        # x <= K * sqrt(3)/2 ? No.
        
        # Let's use the property that a point (x,y) is in a hexagon of side K (aligned)
        # if it is in the intersection of 3 strips.
        # The width of the hexagon in the x-direction is K * sqrt(3).
        # The width in the y-direction is K * 1.5 * 2 = 3K? No.
        
        # Easier: Convert Cartesian to Axial coordinates in the rotated frame?
        # If we rotate the points, we are effectively asking: what is the minimal K
        # such that the points fit in a hexagon of side K, aligned with the axes?
        
        # A hexagon of side K (aligned with x-axis flat-top?)
        # Let's assume pointy-top or flat-top.
        # Standard hexagon with side 1, flat-top:
        # Vertices at angles 0, 60, 120...
        # Apothem a = sqrt(3)/2.
        # Width (distance between parallel sides) = 2 * apothem = sqrt(3).
        # For side K, width = K * sqrt(3).
        
        # The condition for containment in a hexagon of side K centered at origin:
        # max( |x|, |x/2 + y*sqrt(3)/2|, |-x/2 + y*sqrt(3)/2| ) <= K * sqrt(3)/2 ?
        # Let's verify with K=1.
        # Center (0,0) -> 0.
        # Vertex (sqrt(3)/2, 1/2)? No, vertex is at distance 1.
        # Flat-top hexagon vertices: (1,0), (1/2, sqrt(3)/2)...
        # If side is 1, apothem is sqrt(3)/2.
        # The constraint is that the projection on the 3 normal vectors is <= apothem.
        # Normals are at 0, 60, 120 degrees.
        # n1 = (1, 0)
        # n2 = (1/2, sqrt(3)/2)
        # n3 = (-1/2, sqrt(3)/2)
        
        # For a hexagon of side K, the apothem is K * sqrt(3)/2.
        # So we need max projection <= K * sqrt(3)/2.
        
        n1 = np.array([1.0, 0.0])
        n2 = np.array([0.5, sqrt3/2.0])
        n3 = np.array([-0.5, sqrt3/2.0])
        
        proj1 = np.abs(rotated_pts @ n1)
        proj2 = np.abs(rotated_pts @ n2)
        proj3 = np.abs(rotated_pts @ n3)
        
        max_proj = np.max(np.maximum(proj1, np.maximum(proj2, proj3)))
        
        # K * sqrt(3)/2 >= max_proj
        # K >= max_proj / (sqrt(3)/2)
        K = max_proj / (sqrt3 / 2.0)
        return K

    # Search angles
    # Due to symmetry of the 11-point cluster (which is roughly hexagonal),
    # the optimal angle might be near 0 or 30.
    # Let's scan 0 to 60 with step 1 degree.
    
    for deg in range(0, 61, 1):
        if time.time() > deadline - 5.0:
            break
        theta = np.radians(deg)
        K = min_enclosing_hexagon_side(pts, theta)
        L = K + 1.0
        if L < best_L:
            best_L = L
            best_theta = theta

    # Fine tune around best angle if time permits
    if time.time() < deadline - 10.0:
        for deg10 in range(int(best_theta * 10) - 5, int(best_theta * 10) + 6):
            theta = np.radians(deg10 / 10.0)
            K = min_enclosing_hexagon_side(pts, theta)
            L = K + 1.0
            if L < best_L:
                best_L = L

    # Return inner hexagons
    # The inner hexagons are at the original lattice positions (before rotation of outer)
    # But the interface asks for (x, y, angle_degrees) of inner hexagons.
    # And outer center, side, angle.
    
    # We assumed inner hexagons are aligned with the lattice (angle 0).
    # The outer hexagon is rotated by best_theta.
    # Wait, if we rotated the points by -theta to check containment in aligned hex,
    # it means the outer hexagon is rotated by +theta relative to the lattice.
    
    inner = []
    for _, q, r in selected:
        x = sqrt3 * (q + r/2.0)
        y = 1.5 * r
        # Center the inner hexagons as well?
        # The outer center is (0,0) in our calculation.
        # We subtracted the mean of the points.
        # So we must shift the inner points by -mean to match the outer center (0,0).
        # But the construct function returns absolute coordinates.
        # We can return relative to (0,0) and set outer center to (0,0).
        
        # The points were centered.
        # So the coordinates returned should be the centered coordinates.
        # But we calculated pts as centered.
        # So we just use the centered coordinates.
        
        # We need to map back to the specific point.
        # Easier: Recalculate centered coordinates.
        pass

    # Let's just return the centered coordinates calculated in 'pts'
    # And set outer center to (0,0)
    
    # However, 'pts' is an array. We need a list of lists.
    # And angle is 0.0 for all inner hexagons.
    
    inner_list = []
    for p in pts:
        inner_list.append([p[0], p[1], 0.0])
        
    # Outer hexagon
    outer_center = (0.0, 0.0)
    outer_side = best_L
    outer_angle = best_theta * 180.0 / np.pi

    return inner_list, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
