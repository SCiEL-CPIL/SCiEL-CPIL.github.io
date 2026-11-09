import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the 4x3 grid of inner hexagons
    # Using a honeycomb lattice structure
    # 3 rows, 4 columns
    hexes = []
    sqrt3 = np.sqrt(3)
    
    # Generate raw coordinates
    raw_coords = []
    for i in range(3):  # rows
        for j in range(4):  # cols
            # Standard hex grid spacing
            x = j * sqrt3 + (i % 2) * (sqrt3 / 2)
            y = i * 1.5
            raw_coords.append((x, y))
    
    # Center the cluster at the origin
    xs = [p[0] for p in raw_coords]
    ys = [p[1] for p in raw_coords]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    
    for x, y in raw_coords:
        hexes.append([x - mean_x, y - mean_y, 0.0])

    # Determine the minimum side length S of the outer hexagon
    # We test a "Flat-Top" outer hexagon (rotation 30 degrees relative to pointy-top)
    # A flat-top hexagon with side S has vertices at angles 0, 60, 120, 180, 240, 300?
    # No, flat-top means one side is horizontal.
    # Vertices at 30, 90, 150, 210, 270, 330 degrees?
    # Let's use the distance to boundary formula for a regular hexagon.
    
    # For a regular hexagon centered at origin with "radius" (distance to vertex) R, side S = R.
    # The distance from center to the midpoint of a side (apothem) is R * sqrt(3)/2.
    # The boundary is defined by 6 lines.
    
    # Let's assume the outer hexagon is rotated by 30 degrees (Flat-Top)
    # The axes of symmetry for the boundary constraints are at angles 0, 60, 120... for a pointy top?
    # Let's just iterate over possible rotations of the outer hexagon to find the tightest fit.
    # However, the analytical solution suggests S = 8/sqrt(3) ~ 4.618 with 30 deg rotation.
    
    def min_hex_side(points, outer_angle_deg):
        # outer_angle_deg is the rotation of the outer hexagon.
        # 0 degrees usually means Pointy-Top (vertex at top).
        # 30 degrees means Flat-Top (side at top).
        
        # We want to find the minimum S such that all points are inside.
        # For a hexagon with side S and rotation theta:
        # The support function h(n) = S * cos(30 - (angle(n) - theta))? 
        # Simpler: transform points into the hexagon's local frame.
        
        theta = np.deg2rad(outer_angle_deg)
        
        max_required_s = 0.0
        for px, py, _ in points:
            # Rotate point by -theta to align hexagon with standard orientation
            # Standard orientation: Flat-top or Pointy-top?
            # Let's define standard as Pointy-Top (vertex at 90 deg).
            # Vertices at 90, 30, -30, -90, -150, 150.
            # Sides are between these.
            # Distance to side constraints:
            # The 6 normal vectors to the sides of a pointy-top hexagon are at:
            # 0, 60, 120, 180, 240, 300 degrees.
            # Wait.
            # Pointy Top (Vertex at Top, 90 deg):
            # Top side is horizontal? No, Vertex is at top.
            # So top boundary is a vertex.
            # The "flat" sides are at angles 30, 90, 150, 210, 270, 330? No.
            
            # Let's use the property that a regular hexagon is the intersection of 6 strips.
            # For a hexagon with side S, the apothem is A = S * sqrt(3) / 2.
            # The distance from center to any boundary point in direction phi is:
            # r(phi) = A / cos( (phi - alpha) % 60 - 30 )?
            
            # Let's just compute the max of the "hexagonal norm".
            # For a point (x,y), the minimum S required to contain it in a hexagon of rotation theta is:
            # max over 3 axes of |dot(p, axis)| / (A/S factor)
            
            # Axes for Pointy-Top (0 deg rotation):
            # The 3 pairs of opposite sides have normals at 0, 60, 120 degrees.
            # Wait, if vertex is at 90, sides are horizontal?
            # No, if vertex is at 90, the top vertex is (0, S).
            # The side connecting (S/2, S*sqrt(3)/2) and (-S/2, S*sqrt(3)/2) is horizontal?
            # No, those are vertices.
            
            # Let's stick to the robust method:
            # A hexagon with side S and rotation theta contains point p if:
            # for k in 0..5:
            #   dot(p_rotated, n_k) <= S * sqrt(3) / 2
            # where n_k are the unit normals to the sides.
            
            # For Pointy-Top (Vertex at 90 deg), the sides are:
            # 1. Top-Right side. Normal angle: 30 deg.
            # 2. Right side? No, vertex at 30?
            # Let's list vertices of Pointy-Top S=1:
            # (0, 1), (sqrt(3)/2, 0.5), (sqrt(3)/2, -0.5), (0, -1), (-sqrt(3)/2, -0.5), (-sqrt(3)/2, 0.5)
            # Side 1: (0,1) to (sqrt(3)/2, 0.5). Midpoint (sqrt(3)/4, 0.75). Normal is perpendicular to side.
            # Vector: (sqrt(3)/2, -0.5). Normal: (0.5, sqrt(3)/2). Angle 60 deg.
            # Side 2: (sqrt(3)/2, 0.5) to (sqrt(3)/2, -0.5). Vertical. Normal: (1, 0). Angle 0 deg.
            # Side 3: (sqrt(3)/2, -0.5) to (0, -1). Normal: (-0.5, -sqrt(3)/2)? Angle 240?
            
            # So normals are at 0, 60, 120, 180, 240, 300.
            # Condition: dot(p, n_k) <= S * sqrt(3) / 2.
            
            p_rot = np.array([px * np.cos(-theta) - py * np.sin(-theta),
                              px * np.sin(-theta) + py * np.cos(-theta)])
            
            # Normals at 0, 60, 120, 180, 240, 300
            normals = []
            for k in range(6):
                angle = np.deg2rad(k * 60)
                normals.append(np.array([np.cos(angle), np.sin(angle)]))
            
            for n in normals:
                dist = np.dot(p_rot, n)
                # Max dist determines required apothem
                apothem_needed = abs(dist)
                # S = apothem * 2 / sqrt(3)
                s_req = apothem_needed * 2.0 / np.sqrt(3)
                if s_req > max_required_s:
                    max_required_s = s_req
        
        return max_required_s

    # Try rotation 0 (Pointy-Top) and 30 (Flat-Top) for the outer hexagon
    s0 = min_hex_side(hexes, 0.0)
    s30 = min_hex_side(hexes, 30.0)
    
    if s0 <= s30:
        side = s0
        outer_angle = 0.0
    else:
        side = s30
        outer_angle = 30.0

    # Add a tiny epsilon to ensure strict containment due to floating point
    side += 1e-6

    return hexes, (0.0, 0.0), side, outer_angle
# EVOLVE-BLOCK-END
