import math
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull # Not needed, we use SAT

N = 12
DEADLINE = 115.0

# Precompute unit hexagon vertices and edge normals
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    
    # Helper functions for SAT and containment
    def get_rotated_verts(pos, angle_rad):
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        # Rotate unit verts
        rx = _UNIT_HEX_VERTS[:, 0] * c - _UNIT_HEX_VERTS[:, 1] * s
        ry = _UNIT_HEX_VERTS[:, 0] * s + _UNIT_HEX_VERTS[:, 1] * c
        return np.column_stack((rx + pos[0], ry + pos[1]))

    def get_rotated_normals(angle_rad):
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        rx = _EDGE_NORMALS[:, 0] * c - _EDGE_NORMALS[:, 1] * s
        ry = _EDGE_NORMALS[:, 0] * s + _EDGE_NORMALS[:, 1] * c
        return np.column_stack((rx, ry))

    def sat_distance(h1_pos, h1_ang, h2_pos, h2_ang):
        """Returns minimum penetration depth (negative if overlapping) or 0 if disjoint.
        Actually, for constraints, we want distance >= 0.
        If overlapping, return -min_distance_to_separate.
        """
        v1 = get_rotated_verts(h1_pos, h1_ang)
        v2 = get_rotated_verts(h2_pos, h2_ang)
        n1 = get_rotated_normals(h1_ang)
        n2 = get_rotated_normals(h2_ang)
        
        min_dist = float('inf')
        
        # Project onto h1 normals
        for i in range(6):
            nx, ny = n1[i]
            p1 = v1[:, 0] * nx + v1[:, 1] * ny
            p2 = v2[:, 0] * nx + v2[:, 1] * ny
            min1, max1 = np.min(p1), np.max(p1)
            min2, max2 = np.min(p2), np.max(p2)
            
            if max1 < min2 or max2 < min1:
                # Disjoint on this axis
                dist = min(max1, max2) - max(min1, min2) # Wait, if max1 < min2, dist = min2 - max1
                # Correct:
                if max1 < min2:
                    dist = min2 - max1
                else:
                    dist = min1 - max2
                if dist < min_dist:
                    min_dist = dist
            else:
                # Overlapping on this axis
                # Penetration depth is min(max1, max2) - max(min1, min2)
                # We want the minimum "gap" which is negative if overlapping.
                # For SAT, if any axis is disjoint, they are disjoint.
                # If all axes overlap, they are overlapping.
                # The "distance" to separate is the min over axes of (gap).
                # If gap < 0, it's penetration.
                pass
        
        # Re-evaluate:
        # If ANY axis is disjoint, return positive distance (min of disjoint gaps).
        # If NO axis is disjoint, return negative penetration (min of penetrations? No, max penetration? No, min separation required).
        # For constraint "no overlap", we need:
        # For ALL axes, gap >= 0? No.
        # If they overlap, there is NO axis where they are disjoint.
        # The condition for disjointness is: EXISTS axis such that gap > 0.
        # So the constraint is: max_gap > 0.
        # Let's compute max_gap.
        
        max_gap = -float('inf')
        
        # Check h1 normals
        for i in range(6):
            nx, ny = n1[i]
            p1 = v1[:, 0] * nx + v1[:, 1] * ny
            p2 = v2[:, 0] * nx + v2[:, 1] * ny
            min1, max1 = np.min(p1), np.max(p1)
            min2, max2 = np.min(p2), np.max(p2)
            
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                gap = 0.0 # Overlapping on this axis
            
            if gap > max_gap:
                max_gap = gap
                
        # Check h2 normals
        for i in range(6):
            nx, ny = n2[i]
            p1 = v1[:, 0] * nx + v1[:, 1] * ny
            p2 = v2[:, 0] * nx + v2[:, 1] * ny
            min1, max1 = np.min(p1), np.max(p1)
            min2, max2 = np.min(p2), np.max(p2)
            
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                gap = 0.0
            
            if gap > max_gap:
                max_gap = gap
                
        return max_gap

    def is_inside_hexagon(point, center, side, angle_deg):
        """Check if point is inside the hexagon defined by center, side, angle."""
        # Transform point to hexagon local frame
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        angle_rad = np.radians(angle_deg)
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        # Rotate point by -angle
        lx = dx * c + dy * s
        ly = -dx * s + dy * c
        
        # Hexagon vertices in local frame (side length S)
        # Vertices at S * (cos(60k), sin(60k))
        # Normals at 60k + 30
        # Distance to edge i: lx * n_x + ly * n_y <= S * cos(30) = S * sqrt(3)/2
        # n_x, n_y are normals of the edges.
        # Edge normals are at angles 30, 90, 150, 210, 270, 330.
        # proj = lx * cos(theta_n) + ly * sin(theta_n)
        # Must be <= S * sqrt(3)/2 for all 6 normals.
        
        max_proj = -float('inf')
        for k in range(6):
            theta_n = np.radians(30 + k * 60)
            nx = math.cos(theta_n)
            ny = math.sin(theta_n)
            proj = lx * nx + ly * ny
            if proj > max_proj:
                max_proj = proj
                
        # The "inradius" of a hexagon with side S is S * sqrt(3) / 2.
        inradius = side * math.sqrt(3) / 2
        
        return max_proj - inradius # Negative if inside, positive if outside

    def containment_penalty(point, center, side, angle_deg):
        """Returns 0 if inside, positive if outside."""
        val = is_inside_hexagon(point, center, side, angle_deg)
        return max(0.0, val)

    # Objective and Constraints for SLSQP
    def objective(params):
        # params: [x1, y1, a1, ..., x12, y12, a12, x_out, y_out, S_out, a_out]
        S_out = params[3*N + 2]
        return S_out

    def constraint_non_overlap(i, j):
        def c(params):
            x1, y1, a1 = params[3*i], params[3*i+1], params[3*i+2]
            x2, y2, a2 = params[3*j], params[3*j+1], params[3*j+2]
            h1_pos = np.array([x1, y1])
            h2_pos = np.array([x2, y2])
            # SAT distance
            dist = sat_distance(h1_pos, np.radians(a1), h2_pos, np.radians(a2))
            return dist
        return c

    def constraint_containment(i):
        def c(params):
            x, y, a = params[3*i], params[3*i+1], params[3*i+2]
            x_out, y_out, S_out, a_out = params[3*N], params[3*N+1], params[3*N+2], params[3*N+3]
            center_out = np.array([x_out, y_out])
            # Check all 6 vertices of inner hex i against outer hex
            # To be inside, all vertices must be inside.
            # Or, max distance to boundary <= 0.
            # Using the vertex check is safer for convex shapes.
            verts = get_rotated_verts(np.array([x, y]), np.radians(a))
            max_pen = -float('inf')
            for v in verts:
                pen = containment_penalty(v, center_out, S_out, a_out)
                if pen > max_pen:
                    max_pen = pen
            # We want max_pen <= 0
            return -max_pen
        return c

    # Initial Guess: Symmetric
    # 1 Center, 6 Ring 1, 5 Ring 2
    init_params = np.zeros(3*N + 4)
    
    # Center
    init_params[0:3] = [0, 0, 0]
    
    # Ring 1
    r1 = 2.0
    for k in range(6):
        ang = k * math.pi / 3
        idx = 1 + k
        init_params[3*idx] = r1 * math.cos(ang)
        init_params[3*idx+1] = r1 * math.sin(ang)
        init_params[3*idx+2] = 0 # Angle
        
    # Ring 2 (5 hexes)
    r2 = 3.5
    # Place them in gaps of Ring 1
    # Gaps at 30, 90, 150, 210, 270 (5 of the 6 gaps? No, 6 gaps. Pick 5?)
    # Let's pick 5 evenly spaced? Or 5 specific ones?
    # To maintain some symmetry, maybe 5 at 30, 90, 150, 210, 270?
    gap_angles = [30, 90, 150, 210, 270]
    for k, ang_deg in enumerate(gap_angles):
        ang = np.radians(ang_deg)
        idx = 7 + k
        init_params[3*idx] = r2 * math.cos(ang)
        init_params[3*idx+1] = r2 * math.sin(ang)
        init_params[3*idx+2] = 0

    # Outer Hex
    init_params[3*N] = 0
    init_params[3*N+1] = 0
    init_params[3*N+2] = 4.0 # Side
    init_params[3*N+3] = 0 # Angle

    # Constraints list
    cons = []
    # Non-overlap
    for i in range(N):
        for j in range(i+1, N):
            cons.append({'type': 'ineq', 'fun': constraint_non_overlap(i, j)})
            
    # Containment
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': constraint_containment(i)})

    # Bounds
    # Positions: -10 to 10
    # Angles: -180 to 180
    # Side: 2.0 to 10.0
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((-180, 180))
    bounds.append((-10, 10)) # x_out
    bounds.append((-10, 10)) # y_out
    bounds.append((2.0, 10.0)) # S_out
    bounds.append((-180, 180)) # a_out

    best_side = float('inf')
    best_params = init_params.copy()

    # Run SLSQP
    try:
        res = minimize(objective, init_params, method='SLSQP', bounds=bounds, 
                       constraints=cons, options={'maxiter': 200, 'ftol': 1e-9})
        if res.success:
            if res.x[3*N+2] < best_side:
                best_side = res.x[3*N+2]
                best_params = res.x
    except Exception:
        pass

    # If SLSQP fails or is slow, fall back to a simple refinement of the initial guess
    # Or just return the initial guess if it's valid?
    # Let's verify the best_params found.
    
    # Extract results
    x_out = best_params[3*N]
    y_out = best_params[3*N+1]
    S_out = best_params[3*N+2]
    a_out = best_params[3*N+3]
    
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([float(x), float(y), float(a)])
        
    center = (float(x_out), float(y_out))
    side = float(S_out)
    angle = float(a_out)
    
    return inner, center, side, angle
