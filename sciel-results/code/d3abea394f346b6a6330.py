"""Unit regular hexagons inside a regular hexagon - optimized packing."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 11
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side s, rotated by angle_deg."""
    a = math.radians(angle_deg)
    # Regular hexagon: vertices at angles 0, 60, 120, 180, 240, 300 degrees (flat-top)
    # For pointy-top: 30, 90, 150, 210, 270, 330
    # Using flat-top: vertex i at angle = a + i*pi/3
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _signed_dist_point_to_segment(px, py, ax, ay, bx, by):
    """Signed distance from point P to line AB. Positive if P is to the left of AB."""
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-20:
        return px - ax, py - ay
    # Project P onto AB
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    projx = ax + t * dx
    projy = ay + t * dy
    # Signed distance: cross product
    cross = dx * (py - ay) - dy * (px - ax)
    dist = cross / math.sqrt(length_sq)
    return dist

def _hexagon_contains_point(cx, cy, s, angle_deg, px, py):
    """Check if point (px,py) is inside or on boundary of hexagon."""
    verts = _hex_vertices(cx, cy, s, angle_deg)
    # For a convex polygon, point is inside if it's on the same side of all edges
    for i in range(6):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % 6]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if cross < -1e-10:
            return False
    return True

def _hexagons_disjoint(c1, c2, s1, s2, a1, a2, tol=1e-9):
    """Check if two hexagons are disjoint (no interior overlap). They may touch."""
    # Quick check: distance between centers
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    center_dist = math.sqrt(dx * dx + dy * dy)
    # Minimum distance between centers for disjoint hexagons depends on orientation
    # For same orientation, min center distance for non-overlap is 2*s (touching)
    # Actually for two unit hexagons with same orientation, they can be closer
    # Let's do proper check
    
    # Check if any vertex of hex1 is inside hex2
    v1 = _hex_vertices(c1[0], c1[1], s1, a1)
    for vx, vy in v1:
        if _hexagon_contains_point(c2[0], c2[1], s2, a2, vx, vy):
            # Check if strictly inside (not just on boundary)
            # If on boundary, they're touching which is OK
            # For strict interior check:
            inside = False
            for i in range(6):
                v2 = _hex_vertices(c2[0], c2[1], s2, a2)
                ax, ay = v2[i]
                bx, by = v2[(i + 1) % 6]
                cross = (bx - ax) * (vy - ay) - (by - ay) * (vx - ax)
                if cross < -1e-8:
                    return False  # Overlapping
            # All cross products >= -tol means on boundary or outside
            # Need to check if strictly inside
            strictly_inside = True
            for i in range(6):
                v2 = _hex_vertices(c2[0], c2[1], s2, a2)
                ax, ay = v2[i]
                bx, by = v2[(i + 1) % 6]
                cross = (bx - ax) * (vy - ay) - (by - ay) * (vx - ax)
                if cross < 1e-8:
                    strictly_inside = False
                    break
            if strictly_inside:
                return False
    
    # Check if any vertex of hex2 is inside hex1
    v2 = _hex_vertices(c2[0], c2[1], s2, a2)
    for vx, vy in v2:
        strictly_inside = True
        for i in range(6):
            v1v = _hex_vertices(c1[0], c1[1], s1, a1)
            ax, ay = v1v[i]
            bx, by = v1v[(i + 1) % 6]
            cross = (bx - ax) * (vy - ay) - (by - ay) * (vx - ax)
            if cross < 1e-8:
                strictly_inside = False
                break
        if strictly_inside:
            return False
    
    # Check edge intersections
    for i in range(6):
        ax, ay = v1[i]
        bx, by = v1[(i + 1) % 6]
        for j in range(6):
            cx2, cy2 = v2[j]
            dx2, dy2 = v2[(j + 1) % 6]
            if _segments_intersect_strict(ax, ay, bx, by, cx2, cy2, dx2, dy2):
                return False
    
    return True

def _segments_intersect_strict(ax, ay, bx, by, cx, cy, dx, dy):
    """Check if segments AB and CD intersect at an interior point (not just touching endpoints)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    A = (ax, ay)
    B = (bx, by)
    C = (cx, cy)
    D = (dx, dy)
    
    d1 = cross(C, D, A)
    d2 = cross(C, D, B)
    d3 = cross(A, B, C)
    d4 = cross(A, B, D)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    return False

def _outer_hexagon_contains_inner(outer_cx, outer_cy, outer_s, outer_angle, 
                                   inner_cx, inner_cy, inner_s, inner_angle, tol=1e-9):
    """Check if inner hexagon is fully inside outer hexagon."""
    inner_verts = _hex_vertices(inner_cx, inner_cy, inner_s, inner_angle)
    for vx, vy in inner_verts:
        if not _hexagon_contains_point(outer_cx, outer_cy, outer_s, outer_angle, vx, vy):
            return False
    return True

def _compute_outer_side_length(inner_positions, inner_angles, outer_cx=0.0, outer_cy=0.0, outer_angle=0.0):
    """Compute minimum outer side length such that all inner hexagons fit inside."""
    # For each inner hexagon, find the maximum "extent" in each of the 6 directions
    # The outer hexagon with center (0,0), side S, angle 0 has supporting lines
    # at distance S*sqrt(3)/2 from center in each of 6 directions
    
    # Actually, let's think about this differently.
    # A regular hexagon with side S centered at origin has inradius = S*sqrt(3)/2
    # The distance from center to each edge is S*sqrt(3)/2
    
    # For an inner hexagon at position (cx, cy) with angle a, we need to find
    # the minimum S such that all vertices are inside the outer hexagon.
    
    # The outer hexagon (flat-top, angle 0) has edges at:
    # Normal directions: 0, 60, 120, 180, 240, 300 degrees
    # Each edge is at distance S*sqrt(3)/2 from center
    
    # For a point (px, py), the constraint is:
    # For each normal direction theta_i, the projection of (px,py) onto that direction
    # must be <= S*sqrt(3)/2
    
    # Actually, the outer hexagon with vertices at angles 0,60,...,300 and side S:
    # The edge between vertices at angle 0 and 60 has normal pointing outward at 30 degrees
    # The distance from center to this edge is S*sqrt(3)/2
    
    # Let me use a different approach: for each inner hexagon vertex, compute the
    # required outer side length for that vertex to be inside, then take the max.
    
    # For a regular hexagon centered at origin with side S and rotation phi:
    # A point (px, py) is inside iff for all 6 edge normals n_i:
    #   dot((px,py) - center, n_i) <= S*sqrt(3)/2
    # where n_i are unit normals to the edges pointing outward
    
    # For outer hexagon at (outer_cx, outer_cy) with angle outer_angle:
    # Edge normals are at angles outer_angle + 30 + i*60 degrees for i=0..5
    
    sqrt3 = math.sqrt(3)
    max_s = 0.0
    
    for idx in range(N):
        cx, cy = inner_positions[idx]
        angle = inner_angles[idx]
        verts = _hex_vertices(cx, cy, 1.0, angle)
        for vx, vy in verts:
            # Relative to outer center
            rx = vx - outer_cx
            ry = vy - outer_cy
            # For each of 6 edge normals
            for i in range(6):
                normal_angle = math.radians(outer_angle) + math.radians(30) + i * math.radians(60)
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                proj = rx * nx + ry * ny
                # Need proj <= S * sqrt(3)/2
                required_s = proj / (sqrt3 / 2.0) if proj > 0 else 0.0
                if required_s > max_s:
                    max_s = required_s
    
    return max_s

def _validate_solution(inner, outer_center, outer_side, outer_angle):
    """Validate that the solution is feasible."""
    ocx, ocy = outer_center
    # Check all inner hexagons are inside outer
    for i in range(N):
        cx, cy, angle = inner[i]
        if not _outer_hexagon_contains_inner(ocx, ocy, outer_side, outer_angle, cx, cy, 1.0, angle, tol=1e-8):
            return False
    
    # Check pairwise disjointness
    for i in range(N):
        for j in range(i + 1, N):
            if not _hexagons_disjoint((inner[i][0], inner[i][1]), (inner[j][0], inner[j][1]), 
                                       1.0, 1.0, inner[i][2], inner[j][2]):
                return False
    return True

def _initial_hexagonal_lattice():
    """Create initial hexagonal lattice arrangement."""
    positions = []
    angles = []
    # Hexagonal lattice: spacing = 2 for unit hexagons (touching)
    # Actually for unit hexagons in hexagonal packing, center-to-center distance is 2
    # Let's use a compact arrangement
    # Rows in hexagonal lattice
    spacing = 2.0  # minimum center distance for non-overlapping unit hexagons
    dy = spacing * math.sqrt(3) / 2  # vertical spacing between rows
    
    # Place in a hexagonal pattern
    # Row 0: 1 hexagon at center
    # Row 1: 2 hexagons
    # etc.
    
    # Simpler: just use a grid pattern
    # For N=11, use a 3-4-3 or similar pattern
    
    # Let's use a hexagonal close-packed arrangement
    # Center at (0,0)
    positions = [(0.0, 0.0)]
    # Ring 1: 6 hexagons at distance 2
    for i in range(6):
        angle = i * math.pi / 3
        x = 2.0 * math.cos(angle)
        y = 2.0 * math.sin(angle)
        positions.append((x, y))
    # Ring 2: need 4 more for total 11
    # Place at distance 4 in some directions
    for i in range(4):
        angle = i * math.pi / 2 + math.pi / 4
        x = 4.0 * math.cos(angle)
        y = 4.0 * math.sin(angle)
        positions.append((x, y))
    
    angles = [0.0] * N
    return positions, angles

def construct():
    start_time = time.time()
    
    # Initial configuration: hexagonal lattice
    positions, angles = _initial_hexagonal_lattice()
    
    # Convert to numpy for optimization
    # Variables: x1,y1,a1, x2,y2,a2, ..., xN,yN,aN, cx, cy, S, theta
    # We'll optimize positions and angles to minimize S
    
    def _objective_and_check(vars_arr):
        """Given variable array, compute S and check feasibility."""
        n_vars = N * 3 + 4  # N*(x,y,angle) + (cx, cy, S, theta)
        if len(vars_arr) < n_vars:
            return 1e10, False
        
        pos = vars_arr[:N*3].reshape(N, 3)
        cx, cy, S, theta = vars_arr[N*3], vars_arr[N*3+1], vars_arr[N*3+2], vars_arr[N*3+3]
        
        # Compute required S for this configuration
        required_S = _compute_outer_side_length(
            [(pos[i][0], pos[i][1]) for i in range(N)],
            [pos[i][2] for i in range(N)],
            cx, cy, theta
        )
        
        # Check pairwise disjointness
        for i in range(N):
            for j in range(i+1, N):
                if not _hexagons_disjoint(
                    (pos[i][0], pos[i][1]), (pos[j][0], pos[j][1]),
                    1.0, 1.0, pos[i][2], pos[j][2]
                ):
                    return 1e10, False
        
        return required_S, True
    
    # Try several initial configurations and optimize
    best_S = 40.0  # Initial bad value
    best_vars = None
    
    # Initial guess
    initial_vars = np.zeros(N * 3 + 4)
    for i in range(N):
        initial_vars[i*3] = positions[i][0]
        initial_vars[i*3+1] = positions[i][1]
        initial_vars[i*3+2] = angles[i]
    initial_vars[N*3] = 0.0    # outer cx
    initial_vars[N*3+1] = 0.0  # outer cy
    initial_vars[N*3+2] = 10.0 # initial S guess
    initial_vars[N*3+3] = 0.0  # outer angle
    
    # Try to optimize using scipy
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30:
        try:
            # Use a penalty-based approach
            def penalty_objective(vars_arr):
                n_vars = N * 3 + 4
                pos = vars_arr[:N*3].reshape(N, 3)
                cx, cy, S, theta = vars_arr[N*3], vars_arr[N*3+1], vars_arr[N*3+2], vars_arr[N*3+3]
                
                # Compute required S
                required_S = _compute_outer_side_length(
                    [(pos[i][0], pos[i][1]) for i in range(N)],
                    [pos[i][2] for i in range(N)],
                    cx, cy, theta
                )
                
                # Penalty for overlaps
                penalty = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        dx = pos[i][0] - pos[j][0]
                        dy = pos[i][1] - pos[j][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        # Minimum distance for non-overlap (rough estimate)
                        min_dist = 2.0
                        if dist < min_dist:
                            penalty += (min_dist - dist) ** 2
                
                return required_S + 100.0 * penalty
            
            result = minimize(penalty_objective, initial_vars, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            elapsed = time.time() - start_time
            if result.success or elapsed < DEADLINE - 20:
                test_S, feasible = _objective_and_check(result.x)
                if feasible and test_S < best_S:
                    best_S = test_S
                    best_vars = result.x.copy()
        except Exception:
            pass
    
    # If we have a feasible solution, construct the output
    if best_vars is not None:
        pos = best_vars[:N*3].reshape(N, 3)
        cx, cy = best_vars[N*3], best_vars[N*3+1]
        theta = best_vars[N*3+3]
        
        inner = [[float(pos[i][0]), float(pos[i][1]), float(pos[i][2])] for i in range(N)]
        
        # Final validation
        if _validate_solution(inner, (cx, cy), best_S, theta):
            return inner, (float(cx), float(cy)), float(best_S), float(theta)
    
    # Fallback: return the initial configuration with a large enough outer hexagon
    inner = [[float(p[0]), float(p[1]), float(a)] for p, a in zip(positions, angles)]
    S = _compute_outer_side_length(positions, angles, 0.0, 0.0, 0.0)
    S = max(S, 10.0)  # Ensure it's large enough
    
    # Validate fallback
    if _validate_solution(inner, (0.0, 0.0), S, 0.0):
        return inner, (0.0, 0.0), float(S), 0.0
    
    # Last resort: spread out more
    inner = [[float(4.0 * k), 0.0, 0.0] for k in range(-(N//2), N - N//2)]
    return inner, (0.0, 0.0), 40.0, 0.0
