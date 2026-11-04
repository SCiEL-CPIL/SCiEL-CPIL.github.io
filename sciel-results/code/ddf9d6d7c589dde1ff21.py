"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # --- Geometry helpers ---
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
        # Clip v1 by each edge of v2
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

    def outer_side_required(inner_positions, outer_center, outer_angle):
        """Find minimum outer side length such that all inner hexagons fit inside."""
        cx, cy = outer_center
        # Binary search on outer side length
        lo = 1.0
        hi = 100.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            outer_verts = hex_vertices(cx, cy, mid, outer_angle)
            all_fit = True
            for (ix, iy, iang) in inner_positions:
                inner_verts = hex_vertices(ix, iy, 1.0, iang)
                if not hex_in_hexagon(inner_verts, outer_verts):
                    all_fit = False
                    break
            if all_fit:
                hi = mid
            else:
                lo = mid
        return hi

    def is_valid(inner_positions, outer_center, outer_side, outer_angle):
        """Check if configuration is valid: no overlaps, all inside outer hex."""
        cx, cy = outer_center
        outer_verts = hex_vertices(cx, cy, outer_side, outer_angle)
        # Check containment
        for (ix, iy, iang) in inner_positions:
            inner_verts = hex_vertices(ix, iy, 1.0, iang)
            if not hex_in_hexagon(inner_verts, outer_verts):
                return False
        # Check non-overlap
        for i in range(N):
            for j in range(i + 1, N):
                vi = hex_vertices(*inner_positions[i][:2], 1.0, inner_positions[i][2])
                vj = hex_vertices(*inner_positions[j][:2], 1.0, inner_positions[j][2])
                if polygon_overlap_area(vi, vj) > 1e-8:
                    return False
        return True

    # --- Analytic seed solution based on optimal lattice packing ---
    def analytic_solution():
        """
        Construct the optimal concentric ring packing:
        - 1 center hex
        - 6 ring-1 hexagons at distance sqrt(3) (touching)
        - 4 ring-2 hexagons at distance 3 (touching ring-1)
        All hexagons rotated by 30 degrees (flat-top orientation).
        Outer hexagon also rotated by 30 degrees.
        """
        eps = 1e-9
        centers = []
        centers.append((0.0, 0.0))
        
        # 6 neighbors at dist sqrt(3) + eps
        for k in range(6):
            angle = math.radians(k * 60)
            x = (math.sqrt(3) + eps) * math.cos(angle)
            y = (math.sqrt(3) + eps) * math.sin(angle)
            centers.append((x, y))
            
        # 4 more at dist 3 + 2*eps
        # Angles 30, 90, 150, 210
        for k in range(4):
            angle = math.radians(30 + k * 60)
            x = (3.0 + 2*eps) * math.cos(angle)
            y = (3.0 + 2*eps) * math.sin(angle)
            centers.append((x, y))
        
        inner = []
        for x, y in centers:
            inner.append([x, y, 30.0])
        
        # Calculate required outer side length analytically
        # Outer hexagon orientation: 30 degrees (flat top/bottom, vertical sides)
        # Width = sqrt(3) * S
        # Height = 2 * S
        
        # Max X extent:
        # Ring 2 hexagons at 30 and 150 degrees have centers at x = (3+2eps)*cos(30) = (3+2eps)*sqrt(3)/2
        # Rightmost point of that hex (rotated 30 deg): center_x + (sqrt(3)/2)*1
        # Max X = (3+2eps)*sqrt(3)/2 + sqrt(3)/2 = (4+2eps)*sqrt(3)/2 = (2+eps)*sqrt(3)
        
        # Max Y extent:
        # Ring 2 hexagon at 90 deg has center at y = 3+2eps
        # Topmost point: vertex at 90 deg from center, so y_offset = 1
        # Max Y = 3+2eps + 1 = 4+2eps
        
        max_x = (2 + eps) * math.sqrt(3)
        max_y = 4 + 2 * eps
        
        s_x = 2 * max_x / math.sqrt(3)
        s_y = max_y
        
        side = max(s_x, s_y)
        
        # Add a tiny margin to ensure containment
        side += 1e-9
        
        return inner, (0.0, 0.0), side, 30.0

    # --- Optimization-based refinement ---
    def objective(params, outer_center, outer_angle):
        """Penalty-based objective: outer side length + overlap penalties."""
        # Unpack params: for each inner hex: x, y, angle
        inner_positions = []
        for i in range(N):
            x = params[3 * i]
            y = params[3 * i + 1]
            a = params[3 * i + 2] * 180.0 / math.pi  # convert to degrees
            inner_positions.append((x, y, a))

        # Compute required outer side
        outer_side = outer_side_required(inner_positions, outer_center, outer_angle)

        # Overlap penalty
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                vi = hex_vertices(*inner_positions[i][:2], 1.0, inner_positions[i][2])
                vj = hex_vertices(*inner_positions[j][:2], 1.0, inner_positions[j][2])
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

    # Get the analytic solution as baseline
    inner_seed, outer_center_seed, side_seed, angle_seed = analytic_solution()
    best_score = side_seed
    best_result = (inner_seed, outer_center_seed, side_seed, angle_seed)

    # Try optimization to see if we can improve
    if time.time() < deadline - 30:
        outer_center = (0.0, 0.0)
        outer_angle = 0.0  # Let optimizer find best angle too, but start at 0
        
        params0 = initial_guess_from_seed()
        
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective,
                params0,
                args=(outer_center, outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
            )
            
            if time.time() < deadline - 10:
                best_params = result.x
                inner_positions = []
                for i in range(N):
                    x = best_params[3 * i]
                    y = best_params[3 * i + 1]
                    a = best_params[3 * i + 2] * 180.0 / math.pi
                    inner_positions.append((x, y, a))

                outer_side = outer_side_required(inner_positions, outer_center, outer_angle)

                if is_valid(inner_positions, outer_center, outer_side, outer_angle):
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (inner_positions, outer_center, outer_side, outer_angle)
        except Exception:
            pass

    # Return the best result
    inner_positions, outer_center, outer_side, outer_angle = best_result
    inner = [[p[0], p[1], p[2]] for p in inner_positions]
    return inner, (outer_center[0], outer_center[1]), outer_side, outer_angle
# EVOLVE-BLOCK-END
