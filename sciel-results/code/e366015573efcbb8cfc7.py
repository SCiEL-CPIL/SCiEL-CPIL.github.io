import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3.0)

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), rotated by angle_deg, side length side."""
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    # Unit hexagon vertices (flat-to-flat, i.e., vertices at angles 0,60,...,300)
    base_angles = np.arange(6) * np.pi / 3.0
    base = side * np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    # Rotate
    R = np.array([[ca, -sa], [sa, ca]])
    verts = base @ R.T + np.array([cx, cy])
    return verts

def _hexagon_in_hexagon(vertices, container_side, container_angle_deg):
    """Check if all vertices are inside the container hexagon. Return True if inside."""
    a = np.radians(container_angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    # Container hexagon: flat-to-flat, side length container_side
    # Vertices at angles 0,60,...,300 from center, radius = container_side
    # The 6 inward normals point at angles 30,90,150,210,270,330 (for flat-to-flat)
    # Actually for flat-to-flat (flats horizontal), the normals to the edges are at angles 0,60,120,180,240,300
    # Wait: flat-to-flat means the flats are horizontal. So top and bottom edges are horizontal.
    # The normals to the 6 edges point at angles 0, 60, 120, 180, 240, 300 degrees.
    # Each edge is at distance container_side * cos(pi/6) = container_side * sqrt(3)/2 from center? No.
    # For a regular hexagon with side s, the apothem (distance from center to edge) is s * sqrt(3)/2.
    # The vertices are at distance s from center.
    # Edge normals at angles 0,60,...,300. Each edge: n·x <= apothem where apothem = s*sqrt(3)/2.
    # Actually, for flat-to-flat hexagon with side s:
    #   Vertices at (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2,-s*sqrt(3)/2), (s/2,-s*sqrt(3)/2)
    #   Edges: 
    #     Right edge: from (s/2, s*sqrt(3)/2) to (s/2, -s*sqrt(3)/2)? No, that's not right.
    #   Let me reconsider. Flat-to-flat: the flat sides are on top and bottom.
    #   Vertices: (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    #   Top edge: from (-s/2, s*sqrt(3)/2) to (s/2, s*sqrt(3)/2) -- horizontal, y = s*sqrt(3)/2
    #   Bottom edge: y = -s*sqrt(3)/2
    #   Upper-right edge: from (s, 0) to (s/2, s*sqrt(3)/2)
    #   Lower-right edge: from (s, 0) to (s/2, -s*sqrt(3)/2)
    #   etc.
    #   The 6 half-plane constraints (for centered hexagon):
    #     |y| <= s*sqrt(3)/2
    #     x + y/sqrt(3) <= s  (upper-right and upper-left combined? No)
    #   Let me use the standard form. For a regular hexagon with flat top/bottom, side s:
    #     -s*sqrt(3)/2 <= y <= s*sqrt(3)/2
    #     -s + y/sqrt(3) <= x <= s + y/sqrt(3)  ... no
    #   
    #   Better: the 6 edge normals (unit vectors) for flat-to-flat hexagon:
    #     n0 = (1, 0)  -> right edge? No, right edge is slanted.
    #   
    #   Let me just compute the 6 vertices and use the half-plane test directly.
    #   Container vertices (centered at origin, side s, flat-to-flat):
    #     v0 = (s, 0)
    #     v1 = (s/2, s*SQRT3/2)
    #     v2 = (-s/2, s*SQRT3/2)
    #     v3 = (-s, 0)
    #     v4 = (-s/2, -s*SQRT3/2)
    #     v5 = (s/2, -s*SQRT3/2)
    #   
    #   Edge from v0 to v1: direction = (-s/2, s*SQRT3/2), normal (inward) = (SQRT3/2, 1/2) normalized? 
    #   Actually, for a convex polygon, point p is inside if it's on the correct side of each edge.
    #   For edge from vi to vj, the inward normal is perpendicular to (vj-vi), pointing inward.
    
    s = container_side
    cv = np.array([
        [s, 0],
        [s/2, s*SQRT3/2],
        [-s/2, s*SQRT3/2],
        [-s, 0],
        [-s/2, -s*SQRT3/2],
        [s/2, -s*SQRT3/2]
    ])
    # Rotate container
    R = np.array([[ca, -sa], [sa, ca]])
    cv = cv @ R.T
    
    # For each edge, check that all vertices are on the inward side
    for i in range(6):
        p1 = cv[i]
        p2 = cv[(i+1) % 6]
        edge = p2 - p1
        # Inward normal: rotate edge by -90 degrees (clockwise) and check sign
        # For a CCW polygon, inward normal is (edge_y, -edge_x) if polygon is CCW
        # Let's check orientation: v0=(s,0), v1=(s/2, s*sqrt3/2). edge = (-s/2, s*sqrt3/2)
        # Cross product z: (-s/2)(s*sqrt3/2) - (s*sqrt3/2)(-s/2) = -s^2*sqrt3/4 + s^2*sqrt3/4 = 0? No.
        # The polygon is CCW. Inward normal for CCW polygon edge (dx,dy) is (dy, -dx).
        nx, ny = edge[1], -edge[0]
        # Check: for edge v0->v1, n = (s*sqrt3/2, s/2). This should point toward center (0,0).
        # Dot with (0- s, 0-0) = (-s, 0): (s*sqrt3/2)(-s) + (s/2)(0) = -s^2*sqrt3/2 < 0. 
        # Hmm, that means n points away from center. So inward is (-ny, nx) = (-edge[1], edge[0])? 
        # Let me just use: for CCW polygon, the interior is to the left of each directed edge.
        # So for edge p1->p2, a point p is inside if cross(p2-p1, p-p1) >= 0.
        for v in vertices:
            cross = edge[0]*(v[1]-p1[1]) - edge[1]*(v[0]-p1[0])
            if cross < -1e-9:
                return False
    return True

def _overlap_penalty(verts_list, penalty_coeff=1000.0):
    """Compute penalty for overlapping hexagons. Returns 0 if no overlap."""
    # Simple check: for each pair, check if any vertex of one is inside the other
    # This is not perfect but works as a penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            # Check if any vertex of hex i is inside hex j, or vice versa
            for v in verts_list[i]:
                # Point-in-hexagon test for hex j
                # Use the same edge-crossing test
                pass  # Too slow, skip detailed check
    return penalty

def _max_side_needed(positions, angles, container_angle_deg):
    """Find the minimum side length of a flat-to-flat hexagon (rotated by container_angle_deg) 
    that contains all hexagon vertices."""
    all_verts = []
    for i in range(N):
        v = _hex_vertices(positions[i][0], positions[i][1], angles[i])
        all_verts.append(v)
    
    # For a given container side s, the constraint is that all vertices satisfy the 6 half-plane inequalities
    # For flat-to-flat hexagon with side s, centered at origin, rotated by angle a:
    # The 6 inward normals (before rotation) are at angles 0, 60, 120, 180, 240, 300 degrees
    # Wait, for flat-to-flat (flats horizontal), the edge normals point at 0, 60, 120, 180, 240, 300? 
    # No. The flats are on top and bottom, so the top edge has normal (0,1) and bottom (0,-1).
    # The other 4 edges have normals at 60, 120, 240, 300 degrees.
    # So normals are at 0, 60, 120, 180, 240, 300? No:
    #   Top edge: normal (0, 1) -> angle 90
    #   Bottom edge: normal (0, -1) -> angle 270
    #   Upper-right edge: normal at angle 30
    #   Upper-left edge: normal at angle 150
    #   Lower-right edge: normal at angle 330
    #   Lower-left edge: normal at angle 210
    # So normals at 30, 90, 150, 210, 270, 330 degrees.
    # The apothem (distance from center to each edge) is s * sqrt(3)/2.
    
    a = np.radians(container_angle_deg)
    normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0 + a
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_proj = 0.0
    for v in all_verts:
        for n in normals:
            proj = v[0]*n[0] + v[1]*n[1]
            if proj > max_proj:
                max_proj = proj
    
    # apothem = s * sqrt(3)/2, so s = max_proj / (sqrt(3)/2) = 2*max_proj/sqrt(3)
    return 2.0 * max_proj / SQRT3

def _objective(params):
    """Objective: maximize side length needed + penalty for overlaps."""
    positions = params[:2*N].reshape(N, 2)
    angles = params[2*N:2*N+N]
    container_angle = params[2*N+N]
    
    side = _max_side_needed(positions, angles, container_angle)
    
    # Penalty for overlapping hexagons (approximate)
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            # Minimum distance between centers of non-overlapping unit hexagons is 2.0
            if dist < 2.0:
                penalty += (2.0 - dist) ** 2
    
    return side + 100.0 * penalty

def construct():
    # Initial positions: hexagonal lattice pattern
    # 3 rows of 4, with offset
    pos = []
    for row in range(3):
        y = (row - 1) * SQRT3
        for col in range(4):
            x = (col - 1.5) * 2.0
            if row % 2 == 1:
                x += 1.0  # Offset middle row
            pos.append([x, y])
    pos = np.array(pos)
    
    # Center
    pos -= pos.mean(axis=0)
    
    angles_init = np.zeros(N)
    container_angle_init = 0.0
    
    x0 = np.concatenate([pos.flatten(), angles_init, [container_angle_init]])
    
    result = minimize(_objective, x0, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
    
    params = result.x
    positions = params[:2*N].reshape(N, 2)
    angles = params[2*N:2*N+N]
    container_angle = params[2*N+N]
    
    side = _max_side_needed(positions, angles, container_angle)
    
    inner = []
    for i in range(N):
        inner.append([float(positions[i][0]), float(positions[i][1]), float(angles[i])])
    
    center = (0.0, 0.0)
    
    return inner, center, float(side), float(container_angle)
