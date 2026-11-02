import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # 1. Define the 12 hexagon centers in axial coordinates
    # Using a hexagonal spiral order for compactness
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    # 2. Convert axial to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    centers = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append([x, y])
    centers = np.array(centers)

    # 3. Pre-compute vertices for a unit hexagon at angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    angles_hex = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    hex_vertices = np.column_stack([np.cos(angles_hex), np.sin(angles_hex)])
    
    # 4. Define the normals of the outer hexagon (angle 0)
    # Outer hexagon side L, "pointy top" or "flat top"?
    # The return value `angle` for outer hexagon is 0.0 in the seed.
    # Standard regular hexagon with angle 0 usually means a vertex is at (L, 0) or side is horizontal?
    # Let's assume the standard mathematical definition where the polygon is defined by its vertices.
    # If angle=0, and it's a regular hexagon, we need to know the orientation.
    # Usually, "angle" rotates the shape.
    # Let's assume the outer hexagon has vertices at angles 0, 60, ... relative to its center.
    # In that case, the normals (perpendicular to sides) are at 30, 90, 150, 210, 270, 330.
    # Distance from center to side (apothem) is L * cos(pi/6) = L * sqrt(3)/2.
    # Condition: |p . n_k| <= L * sqrt(3)/2 for all k.
    
    outer_normals_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
    outer_normals = np.column_stack([np.cos(outer_normals_angles), np.sin(outer_normals_angles)])

    def calculate_min_side(theta):
        # Rotate centers by theta
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        rotated_centers = centers @ rotation_matrix.T
        
        # We need to check all vertices of all hexagons.
        # A hexagon at center c, rotated by theta, has vertices c + R(theta) @ v_k
        # Since the hexagon itself is rotated by theta, its local vertices are v_k.
        # The global vertex is c_global + R(theta) @ v_k.
        # Wait, if we rotate the centers by theta, and the hexagons were originally at angle 0,
        # then the new hexagons are rotated by theta.
        # So vertex = rotated_center + R(theta) @ hex_vertex_local.
        
        # We can compute the max projection for all vertices.
        # There are 12 hexagons * 6 vertices = 72 points.
        # For each point, project onto 6 normals.
        
        max_proj = 0.0
        for i in range(N):
            c = rotated_centers[i]
            # Local vertices rotated by theta
            # R(theta) @ hex_vertices
            rotated_vertices = hex_vertices @ rotation_matrix.T
            global_vertices = c + rotated_vertices
            
            # Project onto outer normals
            projections = np.abs(global_vertices @ outer_normals.T)
            current_max = np.max(projections)
            if current_max > max_proj:
                max_proj = current_max
                
        # L = 2 / sqrt(3) * max_proj
        return (2.0 / np.sqrt(3)) * max_proj

    # 5. Ternary search for optimal theta in [0, pi/6]
    # The function is periodic and symmetric.
    lo = 0.0
    hi = np.pi / 6.0
    # Perform ternary search
    for _ in range(50):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = calculate_min_side(m1)
        f2 = calculate_min_side(m2)
        if f1 < f2:
            hi = m2
        else:
            lo = m1
            
    theta_opt = (lo + hi) / 2.0
    L_opt = calculate_min_side(theta_opt)
    
    # 6. Construct the output
    # Inner hexagons: centers rotated by theta_opt, angle = theta_opt in degrees
    cos_t = np.cos(theta_opt)
    sin_t = np.sin(theta_opt)
    rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    final_centers = centers @ rotation_matrix.T
    
    inner = []
    angle_deg = np.degrees(theta_opt)
    for i in range(N):
        inner.append([final_centers[i][0], final_centers[i][1], angle_deg])
        
    center = [0.0, 0.0]
    side = L_opt
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
