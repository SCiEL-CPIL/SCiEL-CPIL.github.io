"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)
    rng = random.Random(42)

    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon in CCW order."""
        vertices = []
        angle_rad = math.radians(angle_deg)
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def point_in_convex(px, py, poly):
        """Check if point is inside convex polygon (CCW order)."""
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            if cross < -1e-10:
                return False
        return True

    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        c1 = cross(p3, p4, p1)
        c2 = cross(p3, p4, p2)
        c3 = cross(p1, p2, p3)
        c4 = cross(p1, p2, p4)

        if ((c1 > 0 and c2 < 0) or (c1 < 0 and c2 > 0)) and \
           ((c3 > 0 and c4 < 0) or (c3 < 0 and c4 > 0)):
            return True
        return False

    def polygons_intersect(poly1, poly2):
        """Check if two convex polygons intersect."""
        for p in poly1:
            if point_in_convex(p[0], p[1], poly2):
                return True
        for p in poly2:
            if point_in_convex(p[0], p[1], poly1):
                return True
        for i in range(len(poly1)):
            p1 = poly1[i]
            p2 = poly1[(i + 1) % len(poly1)]
            for j in range(len(poly2)):
                p3 = poly2[j]
                p4 = poly2[(j + 1) % len(poly2)]
                if segments_intersect(p1, p2, p3, p4):
                    return True
        return False

    def check_overlap(centers, angles):
        """Check if any two inner hexagons overlap."""
        polys = []
        for i in range(N):
            polys.append(get_hex_vertices(centers[i][0], centers[i][1], angles[i]))
        
        for i in range(N):
            for j in range(i + 1, N):
                if polygons_intersect(polys[i], polys[j]):
                    return False
        return True

    def enclosing_side_length(centers, angles, outer_angle=0.0):
        """
        Compute minimal side length of outer hexagon containing all inner hexagons.
        """
        max_extent = 0.0
        for k in range(6):
            angle = math.radians(outer_angle + 30 + k * 60)
            nx = math.cos(angle)
            ny = math.sin(angle)
            
            for i in range(N):
                cx, cy = centers[i]
                inner_angle_rad = math.radians(angles[i])
                n_angle = math.atan2(ny, nx)

                max_dot = -2.0
                for j in range(6):
                    v_angle = inner_angle_rad + j * math.pi / 3.0
                    dot_val = math.cos(v_angle - n_angle)
                    if dot_val > max_dot:
                        max_dot = dot_val

                extent = cx * nx + cy * ny + max_dot
                if extent > max_extent:
                    max_extent = extent

        return 2.0 * max_extent / sqrt3

    def min_distance_between_hexagons(poly1, poly2):
        """
        Compute minimum distance between two convex hexagons.
        Returns negative value if they overlap (penetration depth).
        Uses edge-to-edge and vertex-to-edge distance calculations.
        """
        min_dist = float('inf')
        
        # Check all vertex-to-edge distances
        for i in range(6):
            for j in range(6):
                # Distance from vertex i of poly1 to edge j of poly2
                vx, vy = poly1[i]
                ax, ay = poly2[j]
                bx, by = poly2[(j + 1) % 6]
                
                # Project point onto line segment
                dx = bx - ax
                dy = by - ay
                len_sq = dx * dx + dy * dy
                if len_sq < 1e-12:
                    continue
                t = ((vx - ax) * dx + (vy - ay) * dy) / len_sq
                t = max(0.0, min(1.0, t))
                px = ax + t * dx
                py = ay + t * dy
                dist = math.sqrt((vx - px) ** 2 + (vy - py) ** 2)
                if dist < min_dist:
                    min_dist = dist
                
                # Distance from vertex j of poly2 to edge i of poly1
                vx, vy = poly2[j]
                ax, ay = poly1[i]
                bx, by = poly1[(i + 1) % 6]
                
                dx = bx - ax
                dy = by - ay
                len_sq = dx * dx + dy * dy
                if len_sq < 1e-12:
                    continue
                t = ((vx - ax) * dx + (vy - ay) * dy) / len_sq
                t = max(0.0, min(1.0, t))
                px = ax + t * dx
                py = ay + t * dy
                dist = math.sqrt((vx - px) ** 2 + (vy - py) ** 2)
                if dist < min_dist:
                    min_dist = dist
        
        # If polygons overlap, min_dist will be 0, but we need penetration depth
        # Check if any vertex is inside the other polygon
        for p in poly1:
            if point_in_convex(p[0], p[1], poly2):
                # Estimate penetration depth as distance to nearest edge
                min_pen = float('inf')
                for j in range(6):
                    ax, ay = poly2[j]
                    bx, by = poly2[(j + 1) % 6]
                    dx = bx - ax
                    dy = by - ay
                    len_sq = dx * dx + dy * dy
                    if len_sq < 1e-12:
                        continue
                    # Signed distance (negative if inside)
                    cross = (dx * (p[1] - ay) - dy * (p[0] - ax)) / math.sqrt(len_sq)
                    if cross < 0:
                        if abs(cross) < min_pen:
                            min_pen = abs(cross)
                if min_pen < float('inf'):
                    return -min_pen
        
        for p in poly2:
            if point_in_convex(p[0], p[1], poly1):
                min_pen = float('inf')
                for j in range(6):
                    ax, ay = poly1[j]
                    bx, by = poly1[(j + 1) % 6]
                    dx = bx - ax
                    dy = by - ay
                    len_sq = dx * dx + dy * dy
                    if len_sq < 1e-12:
                        continue
                    cross = (dx * (p[1] - ay) - dy * (p[0] - ax)) / math.sqrt(len_sq)
                    if cross < 0:
                        if abs(cross) < min_pen:
                            min_pen = abs(cross)
                if min_pen < float('inf'):
                    return -min_pen
        
        return min_dist

    def smooth_overlap_penalty(centers, angles):
        """
        Compute a smooth penalty based on minimum distances between hexagons.
        Returns 0 if no overlap, positive value if overlapping.
        """
        penalty = 0.0
        polys = []
        for i in range(N):
            polys.append(get_hex_vertices(centers[i][0], centers[i][1], angles[i]))
        
        for i in range(N):
            for j in range(i + 1, N):
                dist = min_distance_between_hexagons(polys[i], polys[j])
                if dist < 0:
                    # Overlapping: use quadratic penalty
                    penalty += 100.0 * dist * dist
                elif dist < 0.1:
                    # Close but not overlapping: small penalty to encourage separation
                    penalty += 0.1 * (0.1 - dist) ** 2
        
        return penalty

    def grid_to_cart(q, r):
        x = sqrt3 * q + sqrt3 / 2 * r
        y = 1.5 * r
        return (x, y)

    # Initial hexagonal lattice configuration
    init_grid_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]

    centers = [grid_to_cart(q, r) for q, r in init_grid_coords]
    angles = [0.0] * N

    if not check_overlap(centers, angles):
        centers = [(5.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        angles = [0.0] * N

    best_L = enclosing_side_length(centers, angles, 0.0)
    best_centers = [c[:] for c in centers]
    best_angles = angles[:]
    best_outer_angle = 0.0

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        centers_opt = []
        angles_opt = []
        for i in range(N):
            centers_opt.append((params[i * 3], params[i * 3 + 1]))
            angles_opt.append(params[i * 3 + 2])
        outer_angle = params[36]

        L = enclosing_side_length(centers_opt, angles_opt, outer_angle)
        penalty = smooth_overlap_penalty(centers_opt, angles_opt)
        
        return L + penalty

    init_params = []
    for i in range(N):
        init_params.append(centers[i][0])
        init_params.append(centers[i][1])
        init_params.append(angles[i])
    init_params.append(0.0)

    # Multi-start with different strategies
    n_starts = 20
    for s in range(n_starts):
        if time.time() > deadline - 10.0:
            break

        if s == 0:
            current_params = init_params[:]
        elif s == 1:
            current_params = init_params[:]
            for i in range(N):
                current_params[i * 3] += rng.uniform(-0.5, 0.5)
                current_params[i * 3 + 1] += rng.uniform(-0.5, 0.5)
                current_params[i * 3 + 2] += rng.uniform(-15.0, 15.0)
            current_params[36] = rng.uniform(-15.0, 15.0)
        elif s == 2:
            current_params = init_params[:]
            for i in range(N):
                current_params[i * 3] += rng.uniform(-1.5, 1.5)
                current_params[i * 3 + 1] += rng.uniform(-1.5, 1.5)
                current_params[i * 3 + 2] += rng.uniform(-45.0, 45.0)
            current_params[36] = rng.uniform(-45.0, 45.0)
        elif s == 3:
            current_params = init_params[:]
            for i in range(N):
                current_params[i * 3 + 2] = 30.0
            current_params[36] = 0.0
        elif s == 4:
            current_params = init_params[:]
            for i in range(N):
                current_params[i * 3 + 2] = 15.0
            current_params[36] = 15.0
        elif s == 5:
            current_params = []
            for i in range(N):
                current_params.append(rng.uniform(-4, 4))
                current_params.append(rng.uniform(-4, 4))
                current_params.append(rng.uniform(-60, 60))
            current_params.append(rng.uniform(-60, 60))
        elif s == 6:
            current_params = []
            for i in range(N):
                angle = 2 * math.pi * i / N
                r = 2.5
                current_params.append(r * math.cos(angle))
                current_params.append(r * math.sin(angle))
                current_params.append(rng.uniform(-10, 10))
            current_params.append(0.0)
        elif s == 7:
            current_params = []
            for i in range(6):
                angle = 2 * math.pi * i / 6
                r = 2.0
                current_params.append(r * math.cos(angle))
                current_params.append(r * math.sin(angle))
                current_params.append(0.0)
            for i in range(6):
                angle = 2 * math.pi * (i + 0.5) / 6
                r = 4.0
                current_params.append(r * math.cos(angle))
                current_params.append(r * math.sin(angle))
                current_params.append(0.0)
            current_params.append(0.0)
        elif s == 8:
            current_params = []
            for i, (q, r) in enumerate(init_grid_coords):
                cx, cy = grid_to_cart(q, r)
                rad = math.radians(30)
                nx = cx * math.cos(rad) - cy * math.sin(rad)
                ny = cx * math.sin(rad) + cy * math.cos(rad)
                current_params.append(nx)
                current_params.append(ny)
                current_params.append(30.0)
            current_params.append(30.0)
        elif s == 9:
            current_params = []
            for i, (q, r) in enumerate(init_grid_coords):
                cx, cy = grid_to_cart(q, r)
                rad = math.radians(15)
                nx = cx * math.cos(rad) - cy * math.sin(rad)
                ny = cx * math.sin(rad) + cy * math.cos(rad)
                current_params.append(nx)
                current_params.append(ny)
                current_params.append(15.0)
            current_params.append(15.0)
        elif s == 10:
            # Compact: 1 center + 5 ring + 6 outer
            current_params = [0.0, 0.0, 0.0]
            for i in range(5):
                angle = 2 * math.pi * i / 5
                r = 2.0
                current_params.append(r * math.cos(angle))
                current_params.append(r * math.sin(angle))
                current_params.append(0.0)
            for i in range(6):
                angle = 2 * math.pi * (i + 0.5) / 6
                r = 3.8
                current_params.append(r * math.cos(angle))
                current_params.append(r * math.sin(angle))
                current_params.append(0.0)
            current_params.append(0.0)
        elif s == 11:
            # Hexagonal arrangement with rotations
            current_params = []
            for i, (q, r) in enumerate(init_grid_coords):
                cx, cy = grid_to_cart(q, r)
                current_params.append(cx)
                current_params.append(cy)
                current_params.append(rng.uniform(-30, 30))
            current_params.append(rng.uniform(-30, 30))
        else:
            # Perturb best found so far
            if best_centers:
                current_params = [x for c in best_centers for x in c] + best_angles + [best_outer_angle]
                for i in range(N):
                    current_params[i * 3] += rng.uniform(-0.3, 0.3)
                    current_params[i * 3 + 1] += rng.uniform(-0.3, 0.3)
                    current_params[i * 3 + 2] += rng.uniform(-10.0, 10.0)
                current_params[36] += rng.uniform(-10.0, 10.0)
            else:
                current_params = init_params[:]
                for i in range(N):
                    current_params[i * 3] += rng.uniform(-2.0, 2.0)
                    current_params[i * 3 + 1] += rng.uniform(-2.0, 2.0)
                    current_params[i * 3 + 2] += rng.uniform(-30.0, 30.0)
                current_params[36] = rng.uniform(-30.0, 30.0)

        # Use L-BFGS-B as primary optimizer (exploits smooth gradients)
        try:
            res = minimize(
                objective,
                current_params,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'gtol': 1e-6, 'maxfun': 50000}
            )
        except Exception:
            res = None
        
        # Polish with Nelder-Mead if L-BFGS-B found something good
        if res is not None and res.fun < objective(current_params):
            current_params = res.x
            centers_opt = []
            angles_opt = []
            for i in range(N):
                centers_opt.append((current_params[i * 3], current_params[i * 3 + 1]))
                angles_opt.append(current_params[i * 3 + 2])
            outer_angle = current_params[36]
            
            if check_overlap(centers_opt, angles_opt):
                L = enclosing_side_length(centers_opt, angles_opt, outer_angle)
                if L < best_L:
                    best_L = L
                    best_centers = centers_opt
                    best_angles = angles_opt
                    best_outer_angle = outer_angle
                    
                    # Polish with Nelder-Mead for local refinement
                    if time.time() < deadline - 20.0:
                        try:
                            res2 = minimize(
                                objective,
                                current_params,
                                method='Nelder-Mead',
                                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                            )
                            if res2.fun < objective(current_params):
                                current_params = res2.x
                                centers_opt2 = []
                                angles_opt2 = []
                                for i in range(N):
                                    centers_opt2.append((current_params[i * 3], current_params[i * 3 + 1]))
                                    angles_opt2.append(current_params[i * 3 + 2])
                                outer_angle2 = current_params[36]
                                if check_overlap(centers_opt2, angles_opt2):
                                    L2 = enclosing_side_length(centers_opt2, angles_opt2, outer_angle2)
                                    if L2 < best_L:
                                        best_L = L2
                                        best_centers = centers_opt2
                                        best_angles = angles_opt2
                                        best_outer_angle = outer_angle2
                        except Exception:
                            pass

    # Final check and return
    if not check_overlap(best_centers, best_angles):
        best_centers = [grid_to_cart(q, r) for q, r in init_grid_coords]
        best_angles = [0.0] * N
        best_L = enclosing_side_length(best_centers, best_angles, 0.0)
        best_outer_angle = 0.0

    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], best_angles[i]])

    center = (0.0, 0.0)
    side = best_L
    angle = best_outer_angle

    return inner, center, side, angle
# EVOLVE-BLOCK-END
