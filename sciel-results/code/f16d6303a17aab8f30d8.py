"""Optimized hexagon packing using gradient-based optimization."""
import numpy as np
from scipy.optimize import minimize
import itertools

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return the 6 vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    angles = angle + np.arange(6) * np.pi / 3
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack([x, y])

def point_to_hexagon_distance(px, py, cx, cy, side, angle_deg):
    """
    Compute the distance from point (px, py) to the boundary of a regular hexagon.
    If the point is outside, returns positive distance; if inside, returns negative distance.
    We want the minimum distance from inner hex vertices to the outer hex boundary.
    """
    # Transform point into the hexagon's local coordinate system
    angle = np.radians(angle_deg)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    dx = px - cx
    dy = py - cy
    lx = cos_a * dx - sin_a * dy
    ly = sin_a * dx + cos_a * dy
    
    # Regular hexagon with side length 'side' (which equals circumradius for regular hex)
    # The hexagon has flat-top orientation when angle=0: vertices at 0, 60, 120, 180, 240, 300
    # In local coords, the hexagon is defined by |x| <= side and |y| <= sqrt(3)/2 * side for the "pointy-top" 
    # but let's use the standard: vertices at (R*cos(k*pi/3), R*sin(k*pi/3)) for k=0..5 where R=side
    
    # For a regular hexagon with circumradius R=side, the inradius is R*sqrt(3)/2
    # The hexagon can be described as the intersection of 6 half-planes.
    # For a pointy-top hexagon (vertex at top), the constraints are:
    #   |x| <= R * sin(60) = R * sqrt(3)/2
    #   |y| <= R
    #   |x| + |y|/sqrt(3) ... this gets complicated.
    
    # Better approach: compute distance to each edge and take the minimum (with sign)
    R = side
    angles_local = np.arange(6) * np.pi / 3
    vx = R * np.cos(angles_local)
    vy = R * np.sin(angles_local)
    
    min_dist = np.inf
    for i in range(6):
        x1, y1 = vx[i], vy[i]
        x2, y2 = vx[(i+1) % 6], vy[(i+1) % 6]
        # Edge from (x1,y1) to (x2,y2)
        ex = x2 - x1
        ey = y2 - y1
        # Project point onto edge
        t = (lx * ex + ly * ey) / (ex*ex + ey*ey)
        t = np.clip(t, 0, 1)
        qx = x1 + t * ex
        qy = y1 + t * ey
        dist = np.sqrt((lx - qx)**2 + (ly - qy)**2)
        
        # Determine if point is inside or outside relative to this edge
        # Normal vector pointing outward
        nx = ey / (ex*ex + ey*ey) * (ex*ex + ey*ey)  # just (ey, -ex) normalized
        # Actually, let's compute signed distance
        # The outward normal for edge i
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        # Outward direction is from center to midpoint
        nlen = np.sqrt(mx*mx + my*my)
        if nlen < 1e-10:
            continue
        nx_out = mx / nlen
        ny_out = my / nlen
        
        # Signed distance: positive if outside
        edge_vec_x = x2 - x1
        edge_vec_y = y2 - y1
        edge_len = np.sqrt(edge_vec_x**2 + edge_vec_y**2)
        # Unit normal outward
        nx = edge_vec_y / edge_len
        ny = -edge_vec_x / edge_len
        # Check which direction is outward
        if nx * mx + ny * my < 0:
            nx = -nx
            ny = -ny
        
        signed_dist = (lx - x1) * nx + (ly - y1) * ny
        if signed_dist > dist:
            # Point is beyond the edge, so distance is signed_dist (outside)
            min_dist = min(min_dist, signed_dist)
        else:
            # Point is on the inside of this edge, use the actual distance but negative if inside
            # We need to determine if the point is inside the hexagon overall
            # For now, use the minimum of signed distances
            min_dist = min(min_dist, signed_dist)
    
    return min_dist

def min_clearance(inner_positions, outer_center, outer_side, outer_angle):
    """
    Compute the minimum clearance between all inner hexagon vertices and the outer hexagon boundary.
    Positive clearance means all inner hexagons are strictly inside.
    """
    ox, oy = outer_center
    min_cl = np.inf
    
    for (ix, iy, iangle) in inner_positions:
        verts = hexagon_vertices(ix, iy, 1.0, iangle)
        for (vx, vy) in verts:
            d = point_to_hexagon_distance(vx, vy, ox, oy, outer_side, outer_angle)
            if d < min_cl:
                min_cl = d
    
    return min_cl

def check_disjointness(inner_positions, tol=1e-6):
    """Check if all inner hexagons are disjoint (no overlap)."""
    n = len(inner_positions)
    for i in range(n):
        xi, yi, ai = inner_positions[i]
        vi = hexagon_vertices(xi, yi, 1.0, ai)
        for j in range(i+1, n):
            xj, yj, aj = inner_positions[j]
            vj = hexagon_vertices(xj, yj, 1.0, aj)
            # Check if any vertex of i is inside j or vice versa
            for (vx, vy) in vi:
                d = point_to_hexagon_distance(vx, vy, xj, yj, 1.0, aj)
                if d < -tol:
                    return False
            for (vx, vy) in vj:
                d = point_to_hexagon_distance(vx, vy, xi, yi, 1.0, ai)
                if d < -tol:
                    return False
    return True

def objective(params, outer_center, outer_side, outer_angle):
    """
    Objective function: minimize negative of minimum clearance (i.e., maximize clearance).
    params: array of length 3*N: [x0, y0, a0, x1, y1, a1, ...]
    Returns negative of min clearance (so minimizing this maximizes clearance).
    """
    n = len(params) // 3
    inner = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner.append((x, y, a))
    
    cl = min_clearance(inner, outer_center, outer_side, outer_angle)
    
    # Add penalty for overlapping inner hexagons
    penalty = 0.0
    for i in range(n):
        xi, yi, ai = inner[i]
        vi = hexagon_vertices(xi, yi, 1.0, ai)
        for j in range(i+1, n):
            xj, yj, aj = inner[j]
            for (vx, vy) in vi:
                d = point_to_hexagon_distance(vx, vy, xj, yj, 1.0, aj)
                if d < 0:
                    penalty += d**2
            for (vx, vy) in hexagon_vertices(xj, yj, 1.0, aj):
                d = point_to_hexagon_distance(vx, vy, xi, yi, 1.0, ai)
                if d < 0:
                    penalty += d**2
    
    # We want to maximize clearance, so minimize -clearance + penalty
    return -cl + penalty

def construct():
    """
    Construct a packing of N unit hexagons inside a regular hexagon,
    minimizing the outer side length.
    Returns: (inner, center, side, angle)
    """
    n = N
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Start with a reasonable initial outer side length
    # For N hexagons in a row, side ~ N * 2 (since each hex has width 2)
    # But we can pack more efficiently with 2D arrangements
    
    best_side = np.inf
    best_inner = None
    
    # Try multiple initial outer side lengths and optimize
    for init_side in [3.0, 4.0, 5.0, 6.0, 7.0]:
        # Initial positions: hexagonal grid pattern
        initial_params = []
        rows = 3
        cols = 4
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= n:
                    break
                # Hex grid spacing
                x = c * 2.0 + (r % 2) * 1.0
                y = r * np.sqrt(3)
                a = 0.0
                initial_params.extend([x, y, a])
                idx += 1
        
        # Pad if needed
        while len(initial_params) < 3 * n:
            initial_params.extend([0.0, 0.0, 0.0])
        
        initial_params = np.array(initial_params[:3*n])
        
        # Optimize
        result = minimize(
            objective,
            initial_params,
            args=(outer_center, init_side, outer_angle),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        if result.success or result.fun < 0:
            # Check if the solution is valid
            params = result.x
            inner = []
            for i in range(n):
                x = params[3*i]
                y = params[3*i+1]
                a = params[3*i+2]
                inner.append((float(x), float(y), float(a)))
            
            cl = min_clearance(inner, outer_center, init_side, outer_angle)
            if cl > 0 and check_disjointness(inner):
                # Try to shrink the outer hexagon
                # Binary search for minimum side length
                lo, hi = 1.0, init_side
                for _ in range(20):
                    mid = (lo + hi) / 2
                    # Check if current inner fits in outer with side=mid
                    cl_mid = min_clearance(inner, outer_center, mid, outer_angle)
                    if cl_mid > 0 and check_disjointness(inner):
                        hi = mid
                    else:
                        lo = mid
                
                if hi < best_side:
                    best_side = hi
                    best_inner = inner
    
    if best_inner is None:
        # Fallback: use the seed solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, (0.0, 0.0), float(best_side), 0.0
