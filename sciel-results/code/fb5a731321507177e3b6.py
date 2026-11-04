import numpy as np
from scipy.spatial import ConvexHull

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the rigid cluster of 11 hexagons
    # 7-hexagon flower + 4 hexagons at side positions
    sqrt3 = np.sqrt(3)
    centers = [
        (0.0, 0.0),
        (sqrt3, 0.0), (-sqrt3, 0.0),
        (sqrt3/2, 1.5), (-sqrt3/2, 1.5),
        (sqrt3/2, -1.5), (-sqrt3/2, -1.5),
        (3*sqrt3/2, 1.5), (-3*sqrt3/2, 1.5),
        (3*sqrt3/2, -1.5), (-3*sqrt3/2, -1.5)
    ]
    
    # Generate vertices of all 11 hexagons
    # A unit regular hexagon centered at (cx, cy) with rotation 0 has vertices at:
    # (cx + cos(k*60), cy + sin(k*60)) for k=0..5
    vertices = []
    for cx, cy in centers:
        for k in range(6):
            angle = k * np.pi / 3
            x = cx + np.cos(angle)
            y = cy + np.sin(angle)
            vertices.append((x, y))
            
    vertices = np.array(vertices)
    
    # Compute convex hull
    hull = ConvexHull(vertices)
    hull_points = vertices[hull.vertices]
    
    # Function to compute minimal enclosing hexagon side length for a given rotation
    def calc_side_length(theta):
        # Rotate hull points by theta
        c = np.cos(theta)
        s = np.sin(theta)
        rot_matrix = np.array([[c, -s], [s, c]])
        rotated_points = hull_points @ rot_matrix.T
        
        # Calculate width in 6 directions
        # Directions are theta + k * 60 degrees
        # But since we rotated the points, we can just use fixed directions 0, 60, ...
        # Wait, if we rotate the points by theta, the "fixed" directions of the enclosing hexagon
        # should be 0, 60, ... relative to the original frame?
        # Yes. The enclosing hexagon is fixed at 0, and we rotate the cluster.
        # So we project rotated_points onto vectors at 0, 60, 120, 180, 240, 300.
        
        min_width = np.inf
        for k in range(6):
            angle = k * np.pi / 3
            u = np.array([np.cos(angle), np.sin(angle)])
            projections = rotated_points @ u
            width = np.max(projections) - np.min(projections)
            if width > min_width:
                min_width = width
                
        # Side length S is width / sqrt(3)
        return min_width / np.sqrt(3)

    # Ternary search for optimal theta in [0, pi/3]
    lo = 0.0
    hi = np.pi / 3
    for _ in range(100):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if calc_side_length(m1) < calc_side_length(m2):
            hi = m2
        else:
            lo = m1
            
    optimal_theta = (lo + hi) / 2
    optimal_side = calc_side_length(optimal_theta)
    
    # Construct the inner list
    # Rotate centers by optimal_theta
    c = np.cos(optimal_theta)
    s = np.sin(optimal_theta)
    inner = []
    angle_deg = np.degrees(optimal_theta)
    for cx, cy in centers:
        rx = cx * c - cy * s
        ry = cx * s + cy * c
        inner.append([rx, ry, angle_deg])
        
    return inner, (0.0, 0.0), optimal_side, 0.0
# EVOLVE-BLOCK-END
