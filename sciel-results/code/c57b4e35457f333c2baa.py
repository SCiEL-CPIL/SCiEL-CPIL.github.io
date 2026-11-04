import numpy as np
from scipy.spatial import ConvexHull

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the compact 11-hexagon cluster in axial coordinates
    # (0,0) center
    # 6 neighbors
    # 4 "corner" neighbors from the next ring (distance squared 3)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-1, -1), (1, -2)
    ]
    
    # Convert axial to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r  (Wait, standard hex grid:
    # If pointy top:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # If flat top:
    # x = 3/2 * q
    # y = sqrt(3) * (r + q/2)
    # Let's use flat top, so sides are horizontal/vertical-ish?
    # Regular hexagon side 1.
    # Distance between centers is sqrt(3).
    # Let's use the standard "pointy top" orientation for the hexagons themselves.
    # Center (0,0). Vertices at angles 30, 90, ...
    
    d = np.sqrt(3)
    centers = []
    for q, r in axial_coords:
        x = d * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    # Generate vertices for all hexagons
    # A regular hexagon of side 1, pointy top, centered at (cx, cy)
    # Vertices are at distance 1 from center, angles 30 + k*60
    all_vertices = []
    for cx, cy in centers:
        for k in range(6):
            angle = np.radians(30 + k * 60)
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            all_vertices.append((vx, vy))
    
    all_vertices = np.array(all_vertices)
    
    # Compute convex hull of the union of hexagons
    hull = ConvexHull(all_vertices)
    hull_vertices = all_vertices[hull.vertices]
    
    # Function to calculate minimal enclosing hexagon side length for a given rotation theta
    def min_side_length(theta):
        # Rotate hull vertices by theta
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated_verts = hull_vertices @ R.T
        
        # The target hexagon is fixed with "flat top" or "pointy top"?
        # Let's assume the target hexagon has normals at angles 0, 60, 120, 180, 240, 300.
        # This corresponds to a hexagon with a flat side on top/bottom?
        # If normals are at 0, 60..., the sides are perpendicular to these.
        # Side 1 normal 0 (right). Side is vertical? No, normal is x-axis, side is y-axis.
        # So the hexagon has vertical sides on left and right.
        # This is a "pointy top" hexagon?
        # Vertices of pointy top: 90, 150, 210, 270, 330, 30.
        # Normals: 0, 60, 120, 180, 240, 300.
        # Yes.
        
        normals = []
        for k in range(6):
            angle = np.radians(k * 60)
            normals.append((np.cos(angle), np.sin(angle)))
        normals = np.array(normals)
        
        # Project rotated vertices onto normals
        projections = rotated_verts @ normals.T
        widths = np.max(projections, axis=0) - np.min(projections, axis=0)
        
        # Side length S such that width <= S * sqrt(3)
        # So S = width / sqrt(3)
        max_width = np.max(widths)
        return max_width / np.sqrt(3)
    
    # Ternary search for optimal theta in [0, pi/3]
    lo = 0.0
    hi = np.pi / 3.0
    for _ in range(50):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = min_side_length(m1)
        f2 = min_side_length(m2)
        if f1 < f2:
            hi = m2
        else:
            lo = m1
            
    theta_opt = (lo + hi) / 2.0
    side_len = min_side_length(theta_opt)
    
    # Rotate the centers back to get final positions
    cos_t = np.cos(theta_opt)
    sin_t = np.sin(theta_opt)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    inner = []
    for cx, cy in centers:
        pos = R @ np.array([cx, cy])
        x, y = pos
        angle_deg = np.degrees(theta_opt)
        inner.append([float(x), float(y), float(angle_deg)])
        
    return inner, (0.0, 0.0), float(side_len), 0.0
# EVOLVE-BLOCK-END
