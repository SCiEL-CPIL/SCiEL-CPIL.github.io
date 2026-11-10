"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import random
import numpy as np
from itertools import combinations

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

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
        """Compute minimal side length of outer hexagon containing all inner hexagons."""
        normals = []
        for k in range(6):
            angle = math.radians(outer_angle + 30 + k * 60)
            normals.append((math.cos(angle), math.sin(angle)))

        max_extent = 0.0
        for i in range(N):
            cx, cy = centers[i]
            inner_angle_rad = math.radians(angles[i])

            for nk in range(6):
                nx, ny = normals[nk]
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

        if not check_overlap(centers_opt, angles_opt):
            L = enclosing_side_length(centers_opt, angles_opt, outer_angle)
            violation = 0.0
            polys = []
            for i in range(N):
                polys.append(get_hex_vertices(centers_opt[i][0], centers_opt[i][1], angles_opt[i]))
            for i in range(N):
                for j in range(i + 1, N):
                    for p in polys[i]:
                        if point_in_convex(p[0], p[1], polys[j]):
                            violation += 1.0
                    for p in polys[j]:
                        if point_in_convex(p[0], p[1], polys[i]):
                            violation += 1.0
            return 1000.0 + L + 10.0 * violation

        return enclosing_side_length(centers_opt, angles_opt, outer_angle)

    init_params = []
    for i in range(N):
        init_params.append(centers[i][0])
        init_params.append(centers[i][1])
        init_params.append(angles[i])
    init_params.append(0.0)

    def local_search(params, max_iter=300, step_size=0.5):
        """Simple local search with adaptive step size."""
        current = params[:]
        best_obj = objective(current)
        if best_obj > 1000:
            return current, best_obj

        improved = True
        ss = step_size
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if not improved:
                break
            improved = False
            for _ in range(30):
                trial = current[:]
                idx = rng.randint(0, 35)
                trial[idx] += rng.uniform(-ss, ss)
                obj = objective(trial)
                if obj < best_obj - 1e-7:
                    current = trial
                    best_obj = obj
                    improved = True
                    ss = max(0.05, ss * 0.92)
                    break

            if not improved:
                ss *= 0.75
                if ss < 0.02:
                    break

        return current, best_obj

    def scipy_optimize(params, max_iter=200):
        """Use scipy minimize if available."""
        if not HAS_SCIPY:
            return params, objective(params)

        def obj_func(p):
            if time.time() > deadline:
                return 1e10
            return objective(p)

        try:
            result = minimize(obj_func, params, method='Nelder-Mead',
                              options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-6})
            if result.fun < 1000:
                return list(result.x), result.fun
        except Exception:
            pass
        return params, objective(params)

    # Multi-start strategies
    n_starts = 15
    for s in range(n_starts):
        if time.time() > deadline - 15.0:
            break

        if s == 0:
            current_params = init_params[:]
        elif s == 1:
            # Small perturbation
            current_params = init_params[:]
            for i in range(N):
                current_params[i * 3] += rng.uniform(-0.3, 0.3)
                current_params[i * 3 + 1] += rng.uniform(-0.3, 0.3)
                current_params[i * 3 + 2] += rng.uniform(-10.0, 10.0)
            current_params[36] = rng.uniform(-10.0, 10.0)
        elif s == 2:
            # Medium perturbation
            current_params = init_params[:]
            for i in range(N):
                current_params[i * 3] += rng.uniform(-1.0, 1.0)
                current_params[i * 3 + 1] += rng.uniform(-1.0, 1.0)
                current_params[i * 3 + 2] += rng.uniform(-30.0, 30.0)
            current_params[36] = rng.uniform(-30.0, 30.0)
        elif s == 3:
            # Rotate all by 30
            current_params = init_params[:]
            for i in range(N):
                current_params[i * 3 + 2] = 30.0
        elif s == 4:
            # Rotate all by 15
            current_params = init_params[:]
            for i in range(N):
                current_params[i * 3 + 2] = 15.0
            current_params[36] = 15.0
        elif s == 5:
            # Random
            current_params = []
            for i in range(N):
                current_params.append(rng.uniform(-3, 3))
                current_params.append(rng.uniform(-3, 3))
                current_params.append(rng.uniform(-60, 60))
            current_params.append(rng.uniform(-60, 60))
        elif s == 6:
            # Circular arrangement
            current_params = []
            for i in range(N):
                angle = 2 * math.pi * i / N
                r = 2.8
                current_params.append(r * math.cos(angle))
                current_params.append(r * math.sin(angle))
                current_params.append(rng.uniform(-10, 10))
            current_params.append(0.0)
        elif s == 7:
            # Two rows of 6
            current_params = []
            for i in range(6):
                x = -4.5 + i * 1.8
                y = -1.2
                current_params.append(x)
                current_params.append(y)
                current_params.append(0.0)
            for i in range(6):
                x = -4.5 + i * 1.8
                y = 1.2
                current_params.append(x)
                current_params.append(y)
                current_params.append(0.0)
            current_params.append(0.0)
        elif s == 8:
            # Hexagonal ring with center
            current_params = []
            current_params.append(0.0)
            current_params.append(0.0)
            current_params.append(0.0)
            for i in range(6):
                angle = 2 * math.pi * i / 6
                r = 3.0
                current_params.append(r * math.cos(angle))
                current_params.append(r * math.sin(angle))
                current_params.append(0.0)
            for i in range(5):
                angle = 2 * math.pi * i / 5 + 0.3
                r = 5.0
                current_params.append(r * math.cos(angle))
                current_params.append(r * math.sin(angle))
                current_params.append(0.0)
            current_params.append(0.0)
        elif s == 9:
            # Outer angle 30
            current_params = init_params[:]
            current_params[36] = 30.0
        elif s == 10:
            # Perturbed lattice with rotations
            current_params = init_params[:]
            for i in range(N):
                current_params[i * 3 + 2] = rng.uniform(-20, 20)
            current_params[36] = rng.uniform(-10, 10)
        elif s == 11:
            # Compact hexagonal cluster
            coords = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1),
                      (1,-1), (2,-1), (2,0), (1,1), (0,2), (-1,2)]
            current_params = []
            for q, r in coords:
                x, y = grid_to_cart(q, r)
                current_params.append(x)
                current_params.append(y)
                current_params.append(0.0)
            current_params.append(0.0)
        elif s == 12:
            # Staggered rows
            current_params = []
            for row in range(3):
                for col in range(4):
                    x = -3.0 + col * 1.8 + (0.9 if row % 2 else 0)
                    y = -1.5 + row * 1.6
                    current_params.append(x)
                    current_params.append(y)
                    current_params.append(0.0)
            current_params.append(0.0)
        elif s == 13:
            # Another random
            current_params = []
            for i in range(N):
                current_params.append(rng.uniform(-4, 4))
                current_params.append(rng.uniform(-4, 4))
                current_params.append(rng.uniform(-60, 60))
            current_params.append(rng.uniform(-60, 60))
        else:
            # Yet another random
            current_params = []
            for i in range(N):
                current_params.append(rng.uniform(-3.5, 3.5))
                current_params.append(rng.uniform(-3.5, 3.5))
                current_params.append(rng.uniform(-45, 45))
            current_params.append(rng.uniform(-45, 45))

        # Run local search
        result_params, result_obj = local_search(current_params, max_iter=250, step_size=0.5)

        # If scipy is available and we have time, try it
        if time.time() < deadline - 20.0 and result_obj < 1000:
            result_params2, result_obj2 = scipy_optimize(result_params, max_iter=100)
            if result_obj2 < result_obj:
                result_params, result_obj = result_params2, result_obj2

        # Evaluate final configuration
        centers_opt = []
        angles_opt = []
        for i in range(N):
            centers_opt.append((result_params[i * 3], result_params[i * 3 + 1]))
            angles_opt.append(result_params[i * 3 + 2])
        outer_angle = result_params[36]

        if check_overlap(centers_opt, angles_opt):
            L = enclosing_side_length(centers_opt, angles_opt, outer_angle)
            if L < best_L:
                best_L = L
                best_centers = centers_opt
                best_angles = angles_opt
                best_outer_angle = outer_angle

    # Final validation
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
