import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    sqrt3 = math.sqrt(3.0)

    # Helper to convert axial (q, r) to cartesian (x, y) for pointy-top hexagons
    # This orientation aligns flat edges with the outer hexagon's flat edges when outer angle is 0.
    # For pointy-top:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Generate all 19 centers for a hexagon of radius 2 (side length 3)
    # Axial coordinates (q, r) such that max(|q|, |r|, |q+r|) <= 2
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x, y = axial_to_cartesian(q, r)
                centers.append((x, y))
                
    # We have 19 centers. We need 12.
    # To minimize the bounding box (and thus L), we should pick the most central ones.
    # Sort by distance from origin
    centers.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the first 12
    selected_centers = centers[:12]
    
    # Construct inner list
    # Format: [x, y, angle_degrees]
    # Inner hexagons are pointy-top (angle 0 in our definition corresponds to pointy-top if we define vertices accordingly,
    # but the 'angle' parameter in the output usually rotates the hexagon.
    # Let's stick to the definition:
    # If angle=0 is pointy-top, vertices are at angles 30, 90, 150...
    # Wait, the current program used flat-top logic for vertices but pointy-top logic for grid?
    # Let's be consistent.
    # The grid generated above is pointy-top (neighbors at 30, 90, 150... relative to center? No.)
    # In axial coords for pointy-top:
    # (1,0) -> (sqrt3, 0) -> Right
    # (0,1) -> (sqrt3/2, 1.5) -> Top-Right
    # Distance between (0,0) and (1,0) is sqrt3.
    # Distance between (0,0) and (0,1) is sqrt(3/4 + 2.25) = sqrt(3)
    # So they are touching.
    
    # We need to ensure strict disjointness.
    # The distance between centers is exactly sqrt(3).
    # The sum of inradii is 1 + 1 = 2? No.
    # For two regular hexagons of side 1 to be disjoint, the distance between centers must be > 2 * inradius = sqrt(3).
    # Wait. Inradius = sqrt(3)/2 * side = sqrt(3)/2.
    # If they touch, distance = sqrt(3).
    # So we need distance > sqrt(3).
    # Scaling the positions by a factor > 1 increases the distance.
    
    scale = 1.001
    
    inner = []
    for x, y in selected_centers:
        inner.append([x * scale, y * scale, 0.0])
        
    # Now, calculate the required outer side length L.
    # Outer hexagon: center (0,0), angle 0 (pointy-top? or flat-top?)
    # The task says "regular hexagon n=12" and "outer side length".
    # The score is the outer side length.
    # We need to find the minimum L such that all inner hexagons fit inside.
    
    # Let's assume the outer hexagon is pointy-top (vertices at 90, 150, ...).
    # Or flat-top?
    # It doesn't matter for the minimum L if we can rotate the inner cluster?
    # The inner cluster is fixed relative to each other (rigid body).
    # We can rotate the entire inner cluster to minimize the bounding hexagon size.
    
    # Let's try rotating the inner cluster by an angle theta.
    # We want to minimize the maximum distance from the origin to any vertex of the inner hexagons,
    # projected onto the 6 normal directions of the outer hexagon.
    
    # Outer hexagon constraints:
    # For a point P to be inside a regular hexagon of side L, center (0,0), pointy-top:
    # The vertices are at angles 30, 90, 150, 210, 270, 330 degrees?
    # No, pointy-top has a vertex at 90 degrees.
    # Vertices: (0, L), (L*sqrt3/2, L/2), (L*sqrt3/2, -L/2), (0, -L), (-L*sqrt3/2, -L/2), (-L*sqrt3/2, L/2).
    # The edges are defined by lines.
    # The distance from center to edge (apothem) is L * sqrt3 / 2.
    # The normals to the edges are at angles 0, 60, 120, 180, 240, 300.
    # Wait.
    # Vertex at 90 deg. Edge to 30 deg vertex.
    # Midpoint of edge between 90 and 30: angle 60.
    # Normal is perpendicular to edge.
    # Edge direction is 120 deg (from 30 to 90 is 60 deg arc, chord direction is 90+30=120? No.)
    # Let's just use the dot product with normals.
    # For a regular polygon, the constraint is dot(P, n_k) <= Apothem.
    # For a pointy-top hexagon, the edges are vertical on the left and right? No.
    # Pointy-top: top vertex (0, L).
    # Edge from (0, L) to (L*sqrt3/2, L/2).
    # Vector: (L*sqrt3/2, -L/2). Direction -30 deg.
    # Normal: 60 deg.
    # So normals are at 60, 120, 180, 240, 300, 0.
    # Apothem = L * sqrt3 / 2.
    # So constraint: |dot(P, n_k)| <= L * sqrt3 / 2 for all k.
    # This implies L >= 2 * max_k |dot(P, n_k)| / sqrt3.
    
    # The inner hexagons are also pointy-top (angle 0).
    # Their vertices relative to their center (cx, cy) are at angles 30, 90, 150, 210, 270, 330.
    # Vertex distance from center is 1.
    # Vertices: (cos(30), sin(30)), (cos(90), sin(90)), etc.
    # i.e., (sqrt3/2, 1/2), (0, 1), (-sqrt3/2, 1/2), (-sqrt3/2, -1/2), (0, -1), (sqrt3/2, -1/2).
    
    hex_verts = []
    for k in range(6):
        angle = math.pi / 6 + k * math.pi / 3
        vx = math.cos(angle)
        vy = math.sin(angle)
        hex_verts.append((vx, vy))
    
    normals = []
    for k in range(6):
        angle = k * math.pi / 3
        nx = math.cos(angle)
        ny = math.sin(angle)
        normals.append((nx, ny))
    
    # We can rotate the entire inner cluster by angle phi.
    # We want to minimize L(phi) = 2 * max_{i, v, k} |dot(R_phi * (cx_i + vx_v), n_k)| / sqrt3
    
    best_L = float('inf')
    best_phi = 0.0
    
    # Search for optimal rotation phi
    # The symmetry of the hexagon suggests we only need to search 0 to 30 degrees (pi/6).
    # Let's do a fine grid search.
    
    num_steps = 360
    for i in range(num_steps):
        phi = (i / num_steps) * (math.pi / 6)
        
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        
        max_proj = 0.0
        
        for cx, cy, _ in inner:
            # Rotate center
            rcx = cx * cos_phi - cy * sin_phi
            rcy = cx * sin_phi + cy * cos_phi
            
            for vx, vy in hex_verts:
                # Rotate vertex
                rvx = vx * cos_phi - vy * sin_phi
                rvy = vx * sin_phi + vy * cos_phi
                
                px = rcx + rvx
                py = rcy + rvy
                
                for nx, ny in normals:
                    val = abs(px * nx + py * ny)
                    if val > max_proj:
                        max_proj = val
        
        L_req = 2.0 * max_proj / sqrt3
        if L_req < best_L:
            best_L = L_req
            best_phi = phi
            
    # Apply the best rotation to the inner hexagons
    cos_phi = math.cos(best_phi)
    sin_phi = math.sin(best_phi)
    
    final_inner = []
    for cx, cy, ang in inner:
        rx = cx * cos_phi - cy * sin_phi
        ry = cx * sin_phi + cy * cos_phi
        # The angle of the hexagon itself also rotates?
        # If we rotate the position, we should also rotate the orientation to keep it consistent?
        # Yes, if we rotate the whole cluster, the hexagons rotate too.
        # Original angle was 0. New angle is best_phi in degrees.
        final_inner.append([rx, ry, best_phi * 180.0 / math.pi])
        
    # Add a tiny margin to ensure strict containment and disjointness
    side = best_L * 1.001
    
    center = (0.0, 0.0)
    angle = 0.0 # Outer hexagon orientation is fixed at 0 (pointy-top)
    
    return final_inner, center, side, angle
# EVOLVE-BLOCK-END
