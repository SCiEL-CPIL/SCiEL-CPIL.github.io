import numpy as np
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Packs 12 unit hexagons inside a regular hexagon.
    Strategy: Arrange the 12 inner hexagons in a compact 3x4 rectangular grid of hexagon centers.
    This forms a large parallelogram/hexagon-like cluster.
    We calculate the minimum enclosing regular hexagon for this cluster.
    
    1. Generate centers for a 3x4 grid (offset rows).
    2. Calculate the vertices of all 12 unit hexagons.
    3. Determine the minimal side length S of a regular hexagon (centered at origin, aligned with axes)
       that contains all these vertices.
    4. Return the configuration.
    """
    
    # 1. Generate centers for a compact 3x4 arrangement.
    # A standard hexagonal grid has row offset sqrt(3)/2.
    # Let's use 4 columns and 3 rows.
    # Row y-coordinates: 0, sqrt(3)/2, sqrt(3)
    # Col x-coordinates: 0, 1.5, 3.0, 4.5 (for even rows)
    # Odd rows are shifted by 0.75 in x? No, standard offset is half the spacing.
    # Spacing in x is 1.5 (distance between centers of touching hexes in a row).
    # Distance between rows is sqrt(3)/2.
    # For a "brick wall" pattern:
    # Row 0: x = 0, 1.5, 3.0, 4.5
    # Row 1: x = 0.75, 2.25, 3.75, 5.25 (shifted by 0.75)
    # Row 2: x = 0, 1.5, 3.0, 4.5
    
    # Let's verify if this is compact.
    # The cluster width is roughly 5.25 + radius.
    # The cluster height is roughly sqrt(3) + radius.
    
    # Alternative: 2x6? Width ~ 9. Height ~ 1.732.
    # 3x4 seems more balanced.
    
    # Let's center the cluster at (0,0) to make the enclosing hexagon calculation simpler.
    centers = []
    # 3 rows, 4 cols
    for r in range(3):
        y = r * (math.sqrt(3) / 2.0)
        x_offset = 0.75 if r % 2 == 1 else 0.0
        for c in range(4):
            x = c * 1.5 + x_offset
            centers.append((x, y))
            
    # Center the cluster
    cx = sum(p[0] for p in centers) / N
    cy = sum(p[1] for p in centers) / N
    centers = [(x - cx, y - cy) for x, y in centers]
    
    # 2. Calculate vertices of all inner hexagons.
    # A unit hexagon (side 1) has vertices at angles 0, 60, 120, 180, 240, 300 degrees from its center.
    # Radius (center to vertex) = 1.
    # Wait, side length = 1 implies circumradius = 1.
    # The vertices of a unit hexagon centered at (cx, cy) with 0 rotation are:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    
    all_vertices = []
    angles = [0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3]
    for (hx, hy) in centers:
        for a in angles:
            vx = hx + math.cos(a)
            vy = hy + math.sin(a)
            all_vertices.append((vx, vy))
            
    all_vertices = np.array(all_vertices)
    
    # 3. Find minimal enclosing regular hexagon.
    # The outer hexagon is regular, centered at (0,0).
    # Its sides are aligned with directions 0, 60, 120, 180, 240, 300? 
    # A regular hexagon with "flat top" has normals at 30, 90, 150, 210, 270, 330?
    # Or "pointy top" with normals at 0, 60, 120...?
    # Let's define the outer hexagon by its 6 face normals.
    # The distance from center to each face must be at least the maximum projection of any vertex onto that normal.
    # For a regular hexagon of side S, the distance from center to face (apothem) is S * sqrt(3) / 2.
    # So S >= 2 * max_projection / sqrt(3).
    
    # We can rotate the outer hexagon to minimize S.
    # However, the inner cluster has a specific symmetry.
    # The 3x4 grid has a horizontal axis of symmetry and a vertical axis?
    # The bounding box of the cluster is rectangular-ish.
    # Let's try aligning the outer hexagon with the cluster's primary axes.
    # The cluster is wider than it is tall.
    # A "flat-top" hexagon has horizontal top/bottom faces. Normals are vertical (90, 270) and angled.
    # A "pointy-top" hexagon has vertical left/right faces? No, pointy top means vertices at 90, 270. Faces are at 30, 90... wait.
    
    # Let's just iterate over a few angles for the outer hexagon to find the best fit.
    # The outer hexagon rotation angle theta.
    # The 6 normals are at theta, theta+60, ..., theta+300.
    
    best_S = float('inf')
    best_theta = 0
    
    # Try angles from 0 to 60 degrees in steps of 5 degrees
    for theta_deg in range(0, 61, 5):
        theta = math.radians(theta_deg)
        max_proj = 0
        for i in range(6):
            normal_angle = theta + i * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            projections = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
            # We need the max positive projection and min negative projection?
            # The hexagon contains the origin. The vertices must be within the half-planes defined by the faces.
            # The constraint is |projection| <= apothem.
            # So we need max(abs(projections))
            max_proj = max(max_proj, np.max(np.abs(projections)))
        
        # Apothem = S * sqrt(3) / 2
        # S = 2 * apothem / sqrt(3)
        S = 2.0 * max_proj / math.sqrt(3)
        if S < best_S:
            best_S = S
            best_theta = theta_deg
            
    # Add a tiny epsilon to ensure no floating point errors cause rejection
    best_S += 1e-6
    
    # Construct the return value
    # inner: list of [x, y, angle_degrees]
    inner = [[x, y, 0.0] for x, y in centers]
    # outer center: (0, 0)
    # side: best_S
    # angle: best_theta (rotation of the outer hexagon)
    
    return inner, (0.0, 0.0), best_S, best_theta

# EVOLVE-BLOCK-END
