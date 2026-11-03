import numpy as np
import math

N = 12

def get_hex_vertices(cx, cy, angle_deg=0.0):
    """Return 6 vertices of a unit regular hexagon centered at (cx, cy)."""
    angle_rad = math.radians(angle_deg)
    # Vertices are at angles 0, 60, 120... relative to center
    # For a "pointy top" hex, vertex 0 is at (1, 0)?
    # Standard math: cos(0)=1, sin(0)=0.
    # Let's assume standard orientation where one vertex is on the x-axis.
    # Width (vertex to vertex) = 2. Height (side to side) = sqrt(3).
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + math.cos(theta)
        vy = cy + math.sin(theta)
        verts.append((vx, vy))
    return verts

def min_hex_side(points, outer_rotation_deg):
    """
    Calculate the minimum side length S of a regular hexagon 
    centered at (0,0) with given rotation that contains all points.
    """
    theta = math.radians(outer_rotation_deg)
    # Normals to the sides of the outer hexagon
    # The sides are perpendicular to vectors at angles theta, theta+60, ...
    # Distance from center to side is d = S * sqrt(3) / 2
    # Condition: dot(p, n) <= d for all points p and normals n
    # S >= max(dot(p, n)) / (sqrt(3)/2)
    
    max_proj = 0.0
    for i in range(6):
        n_angle = theta + i * math.pi / 3.0
        nx = math.cos(n_angle)
        ny = math.sin(n_angle)
        for px, py in points:
            proj = px * nx + py * ny
            if proj > max_proj:
                max_proj = proj
    
    # S * sqrt(3)/2 >= max_proj
    # S >= max_proj * 2 / sqrt(3)
    return max_proj * 2.0 / math.sqrt(3.0)

def evaluate_configuration(centers):
    """
    Given a list of centers for N unit hexagons, find the optimal outer hexagon.
    Returns (side, angle).
    """
    # Collect all vertices
    all_points = []
    for cx, cy in centers:
        verts = get_hex_vertices(cx, cy, 0.0)
        all_points.extend(verts)
    
    # Center the configuration at (0,0)
    # Calculate centroid of centers
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # Shift points to center
    centered_points = [(px - cx_avg, py - cy_avg) for px, py in all_points]
    
    best_side = float('inf')
    best_angle = 0.0
    
    # Scan outer rotation from 0 to 30 degrees
    # Due to symmetry, 0-30 is sufficient
    for angle_deg in np.linspace(0, 30, 31):
        s = min_hex_side(centered_points, angle_deg)
        if s < best_side:
            best_side = s
            best_angle = angle_deg
            
    return best_side, best_angle

def construct():
    # Generate candidate configurations
    
    candidates = []
    
    # Candidate 1: 3x4 Grid
    # 3 rows, 4 columns
    # Horizontal spacing: 1.5
    # Vertical spacing: sqrt(3)
    centers_3x4 = []
    for r in range(3):
        for c in range(4):
            x = (c - 1.5) * 1.5
            y = (r - 1.0) * math.sqrt(3)
            centers_3x4.append((x, y))
    candidates.append(centers_3x4)
    
    # Candidate 2: 4x3 Grid
    centers_4x3 = []
    for r in range(4):
        for c in range(3):
            x = (c - 1.0) * 1.5
            y = (r - 1.5) * math.sqrt(3)
            centers_4x3.append((x, y))
    candidates.append(centers_4x3)
            
    # Candidate 3: Hex Cluster 2-3-4-3
    # Row 0: 2
    # Row 1: 3
    # Row 2: 4
    # Row 3: 3
    centers_cluster = []
    # Row 0 (y=0)
    for c in range(2):
        x = (c - 0.5) * 1.5
        centers_cluster.append((x, 0.0))
    # Row 1 (y=sqrt(3))
    for c in range(3):
        x = (c - 1.0) * 1.5
        centers_cluster.append((x, math.sqrt(3)))
    # Row 2 (y=2*sqrt(3))
    for c in range(4):
        x = (c - 1.5) * 1.5
        centers_cluster.append((x, 2.0 * math.sqrt(3)))
    # Row 3 (y=3*sqrt(3))
    for c in range(3):
        x = (c - 1.0) * 1.5
        centers_cluster.append((x, 3.0 * math.sqrt(3)))
    candidates.append(centers_cluster)
    
    # Candidate 4: Hex Cluster 1-3-4-3-1
    centers_cluster2 = []
    # Row 0 (y=0)
    centers_cluster2.append((0.0, 0.0))
    # Row 1 (y=sqrt(3))
    for c in range(3):
        x = (c - 1.0) * 1.5
        centers_cluster2.append((x, math.sqrt(3)))
    # Row 2 (y=2*sqrt(3))
    for c in range(4):
        x = (c - 1.5) * 1.5
        centers_cluster2.append((x, 2.0 * math.sqrt(3)))
    # Row 3 (y=3*sqrt(3))
    for c in range(3):
        x = (c - 1.0) * 1.5
        centers_cluster2.append((x, 3.0 * math.sqrt(3)))
    # Row 4 (y=4*sqrt(3))
    centers_cluster2.append((0.0, 4.0 * math.sqrt(3)))
    candidates.append(centers_cluster2)
    
    # Evaluate all candidates
    best_side = float('inf')
    best_angle = 0.0
    best_centers = None
    
    for centers in candidates:
        s, a = evaluate_configuration(centers)
        if s < best_side:
            best_side = s
            best_angle = a
            best_centers = centers
            
    # Construct the inner list
    inner = []
    for cx, cy in best_centers:
        # The angle of the inner hexagon is 0.0
        inner.append([cx, cy, 0.0])
        
    # The center of the outer hexagon is (0,0) in our local frame
    # But we shifted the points to center them. 
    # The problem asks for outer center. 
    # If we shifted points by (-cx_avg, -cy_avg), the outer center in original coords is (cx_avg, cy_avg).
    # However, it's simpler to just return the centered coordinates for inner and (0,0) for outer.
    # The evaluator will place inner hexagons at (x,y) and outer at (cx,cy).
    # So we should return the ORIGINAL coordinates for inner, and the centroid for outer.
    
    cx_avg = sum(c[0] for c in best_centers) / len(best_centers)
    cy_avg = sum(c[1] for c in best_centers) / len(best_centers)
    
    # Re-construct inner with original coords
    inner = []
    for cx, cy in best_centers:
        inner.append([cx, cy, 0.0])
        
    return inner, (cx_avg, cy_avg), best_side, best_angle

# EVOLVE-BLOCK-START
def construct():
    # We can just call the helper logic inside
    # But to keep it self-contained in the block if required, 
    # I will inline the best logic.
    
    # For efficiency and clarity, I will use the logic developed above.
    
    import numpy as np
    import math
    
    def get_hex_vertices(cx, cy, angle_deg=0.0):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            vx = cx + math.cos(theta)
            vy = cy + math.sin(theta)
            verts.append((vx, vy))
        return verts

    def min_hex_side(points, outer_rotation_deg):
        theta = math.radians(outer_rotation_deg)
        max_proj = 0.0
        for i in range(6):
            n_angle = theta + i * math.pi / 3.0
            nx = math.cos(n_angle)
            ny = math.sin(n_angle)
            for px, py in points:
                proj = px * nx + py * ny
                if proj > max_proj:
                    max_proj = proj
        return max_proj * 2.0 / math.sqrt(3.0)

    def evaluate_configuration(centers):
        all_points = []
        for cx, cy in centers:
            verts = get_hex_vertices(cx, cy, 0.0)
            all_points.extend(verts)
        
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        
        centered_points = [(px - cx_avg, py - cy_avg) for px, py in all_points]
        
        best_side = float('inf')
        best_angle = 0.0
        
        for angle_deg in np.linspace(0, 30, 31):
            s = min_hex_side(centered_points, angle_deg)
            if s < best_side:
                best_side = s
                best_angle = angle_deg
                
        return best_side, best_angle, cx_avg, cy_avg

    candidates = []
    
    # 3x4 Grid
    centers_3x4 = []
    for r in range(3):
        for c in range(4):
            x = (c - 1.5) * 1.5
            y = (r - 1.0) * math.sqrt(3)
            centers_3x4.append((x, y))
    candidates.append(centers_3x4)
    
    # 4x3 Grid
    centers_4x3 = []
    for r in range(4):
        for c in range(3):
            x = (c - 1.0) * 1.5
            y = (r - 1.5) * math.sqrt(3)
            centers_4x3.append((x, y))
    candidates.append(centers_4x3)
            
    # Cluster 2-3-4-3
    centers_cluster = []
    for c in range(2):
        x = (c - 0.5) * 1.5
        centers_cluster.append((x, 0.0))
    for c in range(3):
        x = (c - 1.0) * 1.5
        centers_cluster.append((x, math.sqrt(3)))
    for c in range(4):
        x = (c - 1.5) * 1.5
        centers_cluster.append((x, 2.0 * math.sqrt(3)))
    for c in range(3):
        x = (c - 1.0) * 1.5
        centers_cluster.append((x, 3.0 * math.sqrt(3)))
    candidates.append(centers_cluster)
    
    # Cluster 1-3-4-3-1
    centers_cluster2 = []
    centers_cluster2.append((0.0, 0.0))
    for c in range(3):
        x = (c - 1.0) * 1.5
        centers_cluster2.append((x, math.sqrt(3)))
    for c in range(4):
        x = (c - 1.5) * 1.5
        centers_cluster2.append((x, 2.0 * math.sqrt(3)))
    for c in range(3):
        x = (c - 1.0) * 1.5
        centers_cluster2.append((x, 3.0 * math.sqrt(3)))
    centers_cluster2.append((0.0, 4.0 * math.sqrt(3)))
    candidates.append(centers_cluster2)
    
    best_side = float('inf')
    best_angle = 0.0
    best_centers = None
    best_cx = 0.0
    best_cy = 0.0
    
    for centers in candidates:
        s, a, cx, cy = evaluate_configuration(centers)
        if s < best_side:
            best_side = s
            best_angle = a
            best_centers = centers
            best_cx = cx
            best_cy = cy
            
    inner = []
    for cx, cy in best_centers:
        inner.append([cx, cy, 0.0])
        
    return inner, (best_cx, best_cy), best_side, best_angle
# EVOLVE-BLOCK-END
