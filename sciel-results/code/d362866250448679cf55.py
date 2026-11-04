"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # --- Geometry helpers (vectorized where possible) ---
    def hex_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
        a = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = a + k * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts

    def point_in_convex(p, poly):
        """Check if point p is inside convex polygon poly (CCW)."""
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) < -1e-12:
                return False
        return True

    def polygon_area_signed(verts):
        n = len(verts)
        area = 0.0
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return area / 2.0

    def line_intersect(p1, p2, p3, p4):
        """Intersection of line p1-p2 and line p3-p4."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-15:
            return p1
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    def polygon_overlap_area(v1, v2):
        """Compute overlap area of two convex polygons using Sutherland-Hodgman clipping."""
        def clip_polygon(subject, clip):
            output = list(subject)
            for i in range(len(clip)):
                if not output:
                    break
                inp = output
                output = []
                S = inp[-1]
                for E in inp:
                    if point_in_convex(E, clip):
                        if not point_in_convex(S, clip):
                            output.append(line_intersect(S, E, clip[i], clip[(i + 1) % len(clip)]))
                        output.append(E)
                    elif point_in_convex(S, clip):
                        output.append(line_intersect(S, E, clip[i], clip[(i + 1) % len(clip)]))
                    S = E
            return output

        clipped = clip_polygon(v1, v2)
        if len(clipped) < 3:
            return 0.0
        return abs(polygon_area_signed(clipped))

    def hex_in_hexagon(inner_verts, outer_verts):
        """Check if all vertices of inner hex are inside outer hex."""
        return all(point_in_convex(v, outer_verts) for v in inner_verts)

    # Fast containment check using support functions
    def fast_containment(inner_params, outer_center, outer_side, outer_angle):
        """
        Fast check: for each inner hex, check all 6 vertices against outer hex.
        inner_params: list of (x, y, angle_deg)
        """
        cx, cy = outer_center
        outer_verts = hex_vertices(cx, cy, outer_side, outer_angle)
        for (ix, iy, iang) in inner_params:
            inner_verts = hex_vertices(ix, iy, 1.0, iang)
            if not hex_in_hexagon(inner_verts, outer_verts):
                return False
        return True

    def outer_side_required_fast(inner_params, outer_center, outer_angle):
        """
        Compute minimum outer side length using support function approach.
        For a regular hexagon with side s centered at (cx,cy) with rotation angle,
        the constraint for a point (px,py) to be inside is:
        For each edge normal direction n_k (unit vector), the projection of (px-cx, py-cy) 
        onto n_k must be <= s * sqrt(3)/2 (the apothem).
        So s >= max_k (dot((px-cx,py-cy), n_k)) / (sqrt(3)/2) for all points.
        """
        cx, cy = outer_center
        apothem_factor = math.sqrt(3) / 2.0
        
        # Edge normals: for hexagon with vertex angle offset outer_angle,
        # the edge normals are at angles outer_angle + 30 + k*60 degrees
        normal_angles = [math.radians(outer_angle + 30 + k * 60) for k in range(6)]
        normals = [(math.cos(a), math.sin(a)) for a in normal_angles]
        
        max_val = 0.0
        for (ix, iy, iang) in inner_params:
            inner_verts = hex_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                # Translate to outer center
                dx = vx - cx
                dy = vy - cy
                for nx, ny in normals:
                    proj = dx * nx + dy * ny
                    if proj > max_val:
                        max_val = proj
        
        return max_val / apothem_factor

    def is_valid(inner_params, outer_center, outer_side, outer_angle):
        """Check if configuration is valid: no overlaps, all inside outer hex."""
        # Check containment
        if not fast_containment(inner_params, outer_center, outer_side, outer_angle):
            return False
        # Check non-overlap
        for i in range(N):
            for j in range(i + 1, N):
                vi = hex_vertices(inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2])
                vj = hex_vertices(inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2])
                if polygon_overlap_area(vi, vj) > 1e-8:
                    return False
        return True

    # --- Analytic seed solution ---
    def analytic_solution():
        """
        Construct the concentric ring packing:
        - 1 center hex
        - 6 ring-1 hexagons at distance sqrt(3)
        - 4 ring-2 hexagons at distance 3
        All hexagons rotated by 30 degrees.
        """
        eps = 1e-9
        centers = []
        centers.append((0.0, 0.0))
        
        for k in range(6):
            angle = math.radians(k * 60)
            x = (math.sqrt(3) + eps) * math.cos(angle)
            y = (math.sqrt(3) + eps) * math.sin(angle)
            centers.append((x, y))
            
        for k in range(4):
            angle = math.radians(30 + k * 60)
            x = (3.0 + 2*eps) * math.cos(angle)
            y = (3.0 + 2*eps) * math.sin(angle)
            centers.append((x, y))
        
        inner = []
        for x, y in centers:
            inner.append((x, y, 30.0))
        
        # Calculate required outer side length analytically
        max_x = (2 + eps) * math.sqrt(3)
        max_y = 4 + 2 * eps
        
        s_x = 2 * max_x / math.sqrt(3)
        s_y = max_y
        
        side = max(s_x, s_y) + 1e-9
        
        return inner, (0.0, 0.0), side, 30.0

    # Get the analytic solution as baseline
    inner_seed, outer_center_seed, side_seed, angle_seed = analytic_solution()
    best_score = side_seed
    best_result = (inner_seed, outer_center_seed, side_seed, angle_seed)

    # --- Optimization using gradient-based solver ---
    def objective(params):
        """
        Objective: outer side length + overlap penalty.
        params: 33-dimensional vector (x, y, angle) for each of 11 hexagons.
        Angles are in radians.
        """
        inner_params = []
        for i in range(N):
            x = params[3 * i]
            y = params[3 * i + 1]
            a = params[3 * i + 2] * 180.0 / math.pi
            inner_params.append((x, y, a))

        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        
        outer_side = outer_side_required_fast(inner_params, outer_center, outer_angle)

        # Overlap penalty
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                vi = hex_vertices(inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2])
                vj = hex_vertices(inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2])
                ov = polygon_overlap_area(vi, vj)
                if ov > 1e-10:
                    overlap_penalty += ov ** 2

        return outer_side + 1000.0 * overlap_penalty

    def initial_guess_from_seed():
        """Generate initial positions from the analytic seed."""
        inner, _, _, _ = analytic_solution()
        params = np.zeros(3 * N)
        for i, (x, y, a) in enumerate(inner):
            params[3 * i] = x
            params[3 * i + 1] = y
            params[3 * i + 2] = a * math.pi / 180.0
        return params

    # Try optimization to see if we can improve
    if time.time() < deadline - 20:
        params0 = initial_guess_from_seed()
        
        try:
            from scipy.optimize import minimize
            
            # Try L-BFGS-B with finite differences
            result = minimize(
                objective,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() < deadline - 10:
                best_params = result.x
                inner_positions = []
                for i in range(N):
                    x = best_params[3 * i]
                    y = best_params[3 * i + 1]
                    a = best_params[3 * i + 2] * 180.0 / math.pi
                    inner_positions.append((x, y, a))

                outer_side = outer_side_required_fast(inner_positions, (0.0, 0.0), 0.0)

                if is_valid(inner_positions, (0.0, 0.0), outer_side, 0.0):
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (inner_positions, (0.0, 0.0), outer_side, 0.0)
        except Exception:
            pass

    # Also try Nelder-Mead as a fallback or secondary optimization
    if time.time() < deadline - 15:
        params0 = initial_guess_from_seed()
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective,
                params0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
            )
            
            if time.time() < deadline - 5:
                best_params = result.x
                inner_positions = []
                for i in range(N):
                    x = best_params[3 * i]
                    y = best_params[3 * i + 1]
                    a = best_params[3 * i + 2] * 180.0 / math.pi
                    inner_positions.append((x, y, a))

                outer_side = outer_side_required_fast(inner_positions, (0.0, 0.0), 0.0)

                if is_valid(inner_positions, (0.0, 0.0), outer_side, 0.0):
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (inner_positions, (0.0, 0.0), outer_side, 0.0)
        except Exception:
            pass

    # Return the best result
    inner_positions, outer_center, outer_side, outer_angle = best_result
    inner = [[p[0], p[1], p[2]] for p in inner_positions]
    return inner, (outer_center[0], outer_center[1]), outer_side, outer_angle
# EVOLVE-BLOCK-END
