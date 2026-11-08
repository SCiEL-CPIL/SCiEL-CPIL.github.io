"""New program for the suite task; distinct from previous designs."""
import numpy as np
from scipy.optimize import minimize

N = 12

def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and angle (degrees)."""
    angles = np.deg2rad(angle_deg + np.arange(6) * 60.0)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _container_hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of the container regular hexagon."""
    angles = np.deg2rad(angle_deg + np.arange(6) * 60.0)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _point_in_hexagon(pt, vertices):
    """Check if a point is inside the hexagon defined by vertices (CCW order)."""
    # Use cross product test for convex polygon
    n = len(vertices)
    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        edge = v2 - v1
        to_pt = pt - v1
        cross = edge[0] * to_pt[1] - edge[1] * to_pt[0]
        if cross < -1e-10:  # tolerance
            return False
    return True

def _max_violation(inner, center, side, angle):
    """Return the maximum penetration depth of inner hexagons beyond the container.
    If all are inside, returns a negative number."""
    cx, cy = center
    container_verts = _container_hexagon_vertices(cx, cy, side, angle)
    
    max_violation = -np.inf
    
    for (x, y, ang) in inner:
        inner_verts = _hexagon_vertices(x, y, 1.0, ang)
        for v in inner_verts:
            # Check if vertex v is outside the container
            if not _point_in_hexagon(v, container_verts):
                # Compute distance from v to the container boundary
                # Approximate by checking distance to each edge
                min_dist = np.inf
                n = len(container_verts)
                for i in range(n):
                    v1 = container_verts[i]
                    v2 = container_verts[(i + 1) % n]
                    edge = v2 - v1
                    edge_len = np.linalg.norm(edge)
                    if edge_len == 0:
                        continue
                    # Project v onto the line containing the edge
                    t = np.dot(v - v1, edge) / (edge_len * edge_len)
                    t = max(0.0, min(1.0, t))
                    proj = v1 + t * edge
                    dist = np.linalg.norm(v - proj)
                    # Check if the projection is on the segment (it is by construction)
                    # The sign of violation depends on which side of the edge v is on
                    # If v is outside, the distance is the penetration depth
                    # We need to check if v is on the "outside" side of this edge
                    # For a convex polygon, if v is outside, it's outside at least one edge
                    # The distance to the edge line gives the penetration
                    cross = np.cross(edge, v - v1)
                    # For CCW polygon, outside means cross < 0 for edges
                    # Actually, let's just use the distance and rely on the point-in-poly test
                    min_dist = min(min_dist, dist)
                
                # The violation is the distance from the vertex to the boundary
                # But we need to be careful: if the vertex is far outside, the distance to the nearest edge
                # might not be the penetration depth along the normal.
                # For a better measure, let's use the distance to the container center minus the inradius?
                # Actually, for a hexagon, the distance from center to any point inside is at most side.
                # A simpler metric: compute the minimum side length S' such that the point is inside a hexagon
                # centered at (cx, cy) with rotation angle.
                # The distance from center to a point (x, y) in a hexagon with rotation theta is:
                # Let's transform the point to the hexagon's local frame
                rel_x, rel_y = v[0] - cx, v[1] - cy
                # Rotate back by -angle
                theta = np.deg2rad(-angle)
                lx = rel_x * np.cos(theta) - rel_y * np.sin(theta)
                ly = rel_x * np.sin(theta) + rel_y * np.cos(theta)
                
                # In a hexagon with side S and rotation 0, the boundary is defined by:
                # |x| <= S (along the x-axis for flat-top? No, for pointy-top)
                # Let's use the standard: for a hexagon centered at origin with one vertex on the x-axis (pointy-top)
                # The inequalities are:
                # x <= S
                # x >= -S
                # sqrt(3)*x + y <= sqrt(3)*S  ... wait, let's use the apothem form
                # The apothem (inradius) is r = S * sqrt(3)/2
                # The six sides are at distance r from the center, with normals at angles 0, 60, 120, 180, 240, 300
                # For a point (lx, ly), the distance to the i-th side is:
                # d_i = lx * cos(alpha_i) + ly * sin(alpha_i)
                # where alpha_i are the outward normals.
                # For pointy-top (vertex at angle 0), the sides have normals at 30, 90, 150, 210, 270, 330
                # The point is inside if lx*cos(a) + ly*sin(a) <= r for all a.
                # Violation = max_i (lx*cos(a_i) + ly*sin(a_i) - r)
                
                r = side * np.sqrt(3) / 2.0
                normals = np.deg2rad(30 + np.arange(6) * 60)
                dists = lx * np.cos(normals) + ly * np.sin(normals)
                violation = np.max(dists) - r
                max_violation = max(max_violation, violation)
    
    return max_violation

def construct():
    # Start with a hexagonal arrangement of 12 unit hexagons
    # Center one, 6 around it, 5 more? 
    # A standard hexagonal cluster of 12: 
    # Layer 0: 1 hex
    # Layer 1: 6 hexes
    # Layer 2: 5 hexes (completing a partial hexagonal number)
    # Or use a 3x4 grid rotated.
    
    # Let's use a 3x4 grid for simplicity, then optimize.
    # Positions in a 3x4 grid of hexagons (side 1):
    # Horizontal spacing: 2 * 1 * cos(30) = sqrt(3)
    # Vertical spacing: 1.5
    # Staggered rows
    
    positions = []
    for i in range(3):  # rows
        for j in range(4):  # columns
            x = j * np.sqrt(3)
            y = i * 1.5
            if i % 2 == 1:
                x += np.sqrt(3) / 2
            positions.append([x, y, 0.0])
    
    # Center the grid
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    # Initial guess for container
    initial_side = 4.5
    initial_center = [0.0, 0.0]
    initial_angle = 0.0
    
    # Objective: minimize container side, subject to all inner hexagons being inside
    # We'll use a penalty method: minimize side + penalty * max_violation
    
    def objective(params):
        side = params[0]
        cx, cy = params[1], params[2]
        angle = params[3]
        
        # Compute max violation
        violation = _max_violation(positions, (cx, cy), side, angle)
        
        # Penalty if violation > 0
        penalty = 0.0
        if violation > 0:
            penalty = 100.0 * violation**2
        
        return side + penalty
    
    # Initial parameters: [side, cx, cy, angle]
    x0 = np.array([initial_side, initial_center[0], initial_center[1], initial_angle])
    
    # Bounds
    bounds = [(3.0, 10.0), (-5.0, 5.0), (-5.0, 5.0), (0.0, 60.0)]
    
    result = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    best_side = result.x[0]
    best_center = (result.x[1], result.x[2])
    best_angle = result.x[3]
    
    # Verify and adjust if needed
    final_violation = _max_violation(positions, best_center, best_side, best_angle)
    if final_violation > 1e-4:
        # Increase side slightly
        best_side += final_violation + 0.01
    
    return positions, best_center, best_side, best_angle
