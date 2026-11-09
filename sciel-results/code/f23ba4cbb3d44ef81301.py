"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    sqrt3 = math.sqrt(3.0)
    rng = random.Random(42)

    # Precompute unit hexagon vertices (side=1, angle=0)
    unit_hex = np.array([[math.cos(i * math.pi / 3.0), math.sin(i * math.pi / 3.0)] for i in range(6)])

    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon in CCW order as numpy array."""
        angle_rad = math.radians(angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        return (R @ unit_hex.T).T * side + np.array([cx, cy])

    def min_dist_convex_polygons(v1, v2):
        """Minimum distance between two convex polygons (numpy arrays of shape (6,2)).
        Returns 0 if they overlap or touch, positive distance otherwise."""
        # Check if origin is inside Minkowski difference v1 - v2
        # For convex polygons, compute all edge normals and check
        # Use separating axis theorem: if there's a separating axis, dist > 0
        
        # Fast check: if any vertex of one is inside the other, they overlap
        def point_in_convex_np(px, py, poly):
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                if cross < -1e-9:
                    return False
            return True
        
        for p in v1:
            if point_in_convex_np(p[0], p[1], v2):
                return 0.0
        for p in v2:
            if point_in_convex_np(p[0], p[1], v1):
                return 0.0
        
        # Check edge intersections
        def seg_intersect(p1, p2, p3, p4):
            def orient(a, b, c):
                return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
            o1 = orient(p1, p2, p3)
            o2 = orient(p1, p2, p4)
            o3 = orient(p3, p4, p1)
            o4 = orient(p3, p4, p2)
            if o1 * o2 < 0 and o3 * o4 < 0:
                return True
            return False
        
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1) % 6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1) % 6]
                if seg_intersect(p1, p2, p3, p4):
                    return 0.0
        
        # Compute minimum distance via closest points on edges
        min_d = float('inf')
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1) % 6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1) % 6]
                # Point-segment distances
                for p, a, b in [(p1, p3, p4), (p2, p3, p4), (p3, p1, p2), (p4, p1, p2)]:
                    ab = b - a
                    ap = p - a
                    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-15)
                    t = max(0.0, min(1.0, t))
                    proj = a + t * ab
                    d = np.linalg.norm(p - proj)
                    if d < min_d:
                        min_d = d
        return min_d

    def check_overlap_fast(centers, angles):
        """Check if any two inner hexagons overlap using fast distance check."""
        verts = [get_hex_vertices(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if min_dist_convex_polygons(verts[i], verts[j]) < 1e-8:
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

    if not check_overlap_fast(centers, angles):
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

        if not check_overlap_fast(centers_opt, angles_opt):
            L = enclosing_side_length(centers_opt, angles_opt, outer_angle)
            # Compute penalty based on minimum overlap distance
            verts = [get_hex_vertices(centers_opt[i][0], centers_opt[i][1], angles_opt[i]) for i in range(N)]
            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    d = min_dist_convex_polygons(verts[i], verts[j])
                    if d < 1e-8:
                        penalty += 1.0
            return 1000.0 + L + 10.0 * penalty

        return enclosing_side_length(centers_opt, angles_opt, outer_angle)

    # Define bounds for differential evolution
    bounds = []
    for i in range(N):
        bounds.append((-6, 6))   # x
        bounds.append((-6, 6))   # y
        bounds.append((-60, 60)) # angle
    bounds.append((-30, 30))     # outer angle

    # Run differential evolution
    try:
        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=50,
            popsize=20,
            tol=1e-5,
            mutation=(0.3, 1.0),
            recombination=0.9,
            polish=False,
            updating='immediate',
            init='sobol'
        )
        
        if result.success or result.fun < best_L:
            params = result.x
            centers_opt = []
            angles_opt = []
            for i in range(N):
                centers_opt.append((params[i * 3], params[i * 3 + 1]))
                angles_opt.append(params[i * 3 + 2])
            outer_angle = params[36]
            
            if check_overlap_fast(centers_opt, angles_opt):
                L = enclosing_side_length(centers_opt, angles_opt, outer_angle)
                if L < best_L:
                    best_L = L
                    best_centers = centers_opt
                    best_angles = angles_opt
                    best_outer_angle = outer_angle
    except Exception as e:
        pass

    # Local refinement from best solution found
    if time.time() < deadline - 10.0:
        current_params = []
        for i in range(N):
            current_params.append(best_centers[i][0])
            current_params.append(best_centers[i][1])
            current_params.append(best_angles[i])
        current_params.append(best_outer_angle)
        
        improved = True
        max_iter = 100
        step_size = 0.3
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if not improved:
                break
            improved = False
            best_obj = objective(current_params)
            if best_obj > 1000:
                break

            for _ in range(30):
                trial = current_params[:]
                idx = rng.randint(0, 35)
                trial[idx] += rng.uniform(-step_size, step_size)
                obj = objective(trial)
                if obj < best_obj - 1e-8:
                    current_params = trial
                    best_obj = obj
                    improved = True
                    step_size = max(0.05, step_size * 0.9)
                    break

            if not improved:
                step_size *= 0.8

        # Update best
        centers_opt = []
        angles_opt = []
        for i in range(N):
            centers_opt.append((current_params[i * 3], current_params[i * 3 + 1]))
            angles_opt.append(current_params[i * 3 + 2])
        outer_angle = current_params[36]

        if check_overlap_fast(centers_opt, angles_opt):
            L = enclosing_side_length(centers_opt, angles_opt, outer_angle)
            if L < best_L:
                best_L = L
                best_centers = centers_opt
                best_angles = angles_opt
                best_outer_angle = outer_angle

    # Final validation
    if not check_overlap_fast(best_centers, best_angles):
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
