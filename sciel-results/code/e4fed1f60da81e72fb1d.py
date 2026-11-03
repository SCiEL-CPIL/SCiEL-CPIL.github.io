import numpy as np

N = 11

def construct():
    # Unit hexagon geometry (side length 1, flat-top orientation at angle 0)
    # Center-to-center distances in hex grid:
    #   same row: sqrt(3)
    #   adjacent rows: 1.5 vertical, sqrt(3)/2 horizontal offset
    
    sqrt3 = np.sqrt(3.0)
    
    # Compact hexagonal cluster: center + 6 neighbors + 4 more
    # Pattern: 
    #   Row -1: positions at x = -sqrt3, 0, sqrt3 (3 hexagons)
    #   Row 0:  positions at x = -sqrt3/2, sqrt3/2 (2 hexagons) - offset
    #   Row 1:  positions at x = -sqrt3, 0, sqrt3 (3 hexagons)
    #   Plus 3 more to make 11... 
    # 
    # Better: use a 1-2-3-2-3 pattern or similar
    # 
    # Let me use a known compact arrangement:
    # Center at (0,0)
    # Ring 1: 6 hexagons at distance 2*sqrt(3)/sqrt(3) = 2? No...
    
    # For flat-top hexagons, adjacent center distance = sqrt(3) * (1/sin(60)) = 2? 
    # Actually: the distance between centers of adjacent unit hexagons is 2*apothem = 2*(sqrt(3)/2) = sqrt(3)
    
    # So hex grid with flat-top:
    #   dx_same_row = sqrt(3)
    #   dy_adj_row = 1.5
    #   dx_adj_row_offset = sqrt(3)/2
    
    # Build a cluster:
    # Layer 0 (y=0): x = 0
    # Layer 1 (y=1.5): x = -sqrt(3)/2, sqrt(3)/2
    # Layer 2 (y=3.0): x = -sqrt(3), 0, sqrt(3)
    # Layer -1 (y=-1.5): x = -sqrt(3)/2, sqrt(3)/2
    # Layer -2 (y=-3.0): x = -sqrt(3), 0, sqrt(3)
    # Total: 1+2+3+2+3 = 11
    
    positions = []
    angles = []
    
    # Layer 0
    positions.append((0.0, 0.0))
    angles.append(0.0)
    
    # Layer 1 (y = 1.5)
    positions.append((-sqrt3/2, 1.5))
    angles.append(0.0)
    positions.append((sqrt3/2, 1.5))
    angles.append(0.0)
    
    # Layer 2 (y = 3.0)
    positions.append((-sqrt3, 3.0))
    angles.append(0.0)
    positions.append((0.0, 3.0))
    angles.append(0.0)
    positions.append((sqrt3, 3.0))
    angles.append(0.0)
    
    # Layer -1 (y = -1.5)
    positions.append((-sqrt3/2, -1.5))
    angles.append(0.0)
    positions.append((sqrt3/2, -1.5))
    angles.append(0.0)
    
    # Layer -2 (y = -3.0)
    positions.append((-sqrt3, -3.0))
    angles.append(0.0)
    positions.append((0.0, -3.0))
    angles.append(0.0)
    positions.append((sqrt3, -3.0))
    angles.append(0.0)
    
    inner = [[pos[0], pos[1], ang] for pos, ang in zip(positions, angles)]
    
    # Compute bounding hexagon
    # The cluster spans:
    #   x: from -sqrt3 to sqrt3 -> width 2*sqrt3
    #   y: from -3 to 3 -> height 6
    # 
    # For a regular hexagon (flat-top) to contain this, we need:
    #   The hexagon must contain all 11 hexagons.
    # 
    # A regular hexagon with side S, flat-top, has:
    #   Width (horizontal, flat-to-flat): S * sqrt(3)
    #   Height (vertical, point-to-point): 2*S
    # 
    # But our cluster is taller than wide. Let's use point-top orientation for outer.
    # For point-top hexagon with side S:
    #   Width (horizontal, point-to-point): 2*S
    #   Height (vertical, flat-to-flat): S * sqrt(3)
    
    # Our cluster: x range ~ [-sqrt3, sqrt3] = width 2*sqrt3 ≈ 3.464
    #             y range [-3, 3] = height 6
    # 
    # Plus each inner hexagon has extent: for flat-top unit hex:
    #   x extent: ±sqrt(3)/2
    #   y extent: ±1
    # 
    # So total bounding box of all inner hexagons:
    #   x: [-sqrt3 - sqrt(3)/2, sqrt3 + sqrt(3)/2] = [-3*sqrt(3)/2, 3*sqrt(3)/2]
    #   y: [-3-1, 3+1] = [-4, 4]
    # 
    # Width: 3*sqrt(3) ≈ 5.196
    # Height: 8
    # 
    # For a point-top regular hexagon with side S:
    #   Width = 2*S, Height = S*sqrt(3)
    # We need 2*S >= 5.196 and S*sqrt(3) >= 8
    # S >= 2.598 and S >= 4.619
    # So S >= 4.619
    # 
    # But the hexagon shape isn't just a bounding box - it's hexagonal.
    # Let me compute the minimum S more carefully.
    
    # Get all vertices of all inner hexagons
    def hex_vertices(cx, cy, side=1.0, angle_deg=0.0):
        # Flat-top hexagon vertices
        rad = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = rad + i * np.pi / 3
            # Flat-top: first vertex at angle 0
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append((x, y))
        return vertices
    
    all_vertices = []
    for (px, py, pa) in inner:
        verts = hex_vertices(px, py, 1.0, pa)
        all_vertices.extend(verts)
    
    all_vertices = np.array(all_vertices)
    cx, cy = np.mean(all_vertices[:, 0]), np.mean(all_vertices[:, 1])
    
    # Find minimum S for outer hexagon (point-top) containing all vertices
    # A point-top hexagon with center (0,0) and side S has vertices at:
    #   (S*cos(30+60k), S*sin(30+60k)) for k=0..5, i.e., angles 30,90,150,210,270,330
    # Support function: for direction theta, h(theta) = S * max(cos(theta-30-cos(30+60k)),...)
    # Simpler: check if each vertex is inside.
    
    # For a regular hexagon (point-top) centered at origin with side S:
    # It's the intersection of 6 half-planes.
    # The 6 normals are at angles 0, 60, 120, 180, 240, 300 degrees (pointing to edge midpoints)
    # Edge midpoint distance = S * sqrt(3)/2
    # So: for each normal direction n_k = (cos(k*pi/3), sin(k*pi/3)),
    #     dot(p, n_k) <= S * sqrt(3)/2
    
    def min_side_for_hex(vertices, orientation='point_top'):
        vertices = np.array(vertices)
        cx = np.mean(vertices[:, 0])
        cy = np.mean(vertices[:, 1])
        rel = vertices - np.array([cx, cy])
        
        if orientation == 'point_top':
            # Normals at 0, 60, 120, 180, 240, 300
            normals = np.array([[np.cos(np.pi*k/3), np.sin(np.pi*k/3)] for k in range(6)])
            # h = S * sqrt(3)/2
            S_vals = []
            for n in normals:
                dots = rel @ n
                max_dot = np.max(dots)
                if max_dot > 0:
                    S_vals.append(max_dot / (np.sqrt(3)/2))
            return max(S_vals) if S_vals else 0.0
        else:  # flat_top
            # Normals at 30, 90, 150, 210, 270, 330
            normals = np.array([[np.cos(np.pi*(2*k+1)/6), np.sin(np.pi*(2*k+1)/6)] for k in range(6)])
            S_vals = []
            for n in normals:
                dots = rel @ n
                max_dot = np.max(dots)
                if max_dot > 0:
                    S_vals.append(max_dot / (np.sqrt(3)/2))
            return max(S_vals) if S_vals else 0.0
    
    S_pt = min_side_for_hex(all_vertices, 'point_top')
    S_ft = min_side_for_hex(all_vertices, 'flat_top')
    
    # Try both orientations, pick the smaller
    if S_pt <= S_ft:
        outer_side = S_pt
        outer_angle = 30.0  # point-top means first vertex at 30 deg, so rotation = 30
        outer_center = (cx, cy)
    else:
        outer_side = S_ft
        outer_angle = 0.0
        outer_center = (cx, cy)
    
    # Return format: inner rows (x, y, angle_degrees), outer (x,y), side, angle
    return inner, outer_center, outer_side, outer_angle
