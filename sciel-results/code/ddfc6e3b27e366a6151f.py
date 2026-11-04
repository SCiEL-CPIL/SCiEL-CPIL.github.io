import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, differential_evolution
from scipy.spatial import ConvexHull

N = 11

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    safety_margin = 8.0

    d = math.sqrt(3)

    # Unit hexagon vertices (flat-top, centered at origin)
    verts_rel = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])

    def get_hex_verts(cx, cy, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        rx = verts_rel[:, 0] * c - verts_rel[:, 1] * s
        ry = verts_rel[:, 0] * s + verts_rel[:, 1] * c
        return np.column_stack([rx + cx, ry + cy])

    def polygon_distance(v1, v2):
        """Compute minimum distance between two convex polygons (positive if separated, 0 if touching, negative if overlapping)."""
        min_dist = float('inf')
        # Check all edges of both polygons
        for v in (v1, v2):
            for i in range(6):
                p1 = v[i]
                p2 = v[(i + 1) % 6]
                edge = p2 - p1
                edge_len = np.linalg.norm(edge)
                if edge_len < 1e-10:
                    continue
                edge_dir = edge / edge_len
                normal = np.array([-edge_dir[1], edge_dir[0]])
                
                # Project other polygon vertices onto this edge's normal
                other = v2 if v is v1 else v1
                for j in range(6):
                    pt = other[j]
                    rel = pt - p1
                    t = np.dot(rel, edge_dir)
                    t = max(0, min(edge_len, t))
                    closest = p1 + t * edge_dir
                    dist = np.linalg.norm(pt - closest)
                    min_dist = min(min_dist, dist)
        
        # Also check if one polygon contains a vertex of the other (overlap detection)
        # Simple point-in-polygon test using winding number
        def point_in_hex(pt, hex_verts):
            # Ray casting
            inside = False
            n = len(hex_verts)
            for i in range(n):
                p1 = hex_verts[i]
                p2 = hex_verts[(i + 1) % n]
                if ((p1[1] > pt[1]) != (p2[1] > pt[1])) and \
                   (pt[0] < (p2[0] - p1[0]) * (pt[1] - p1[1]) / (p2[1] - p1[1]) + p1[0]):
                    inside = not inside
            return inside
        
        # Check if any vertex of v1 is inside v2 or vice versa
        for pt in v1:
            if point_in_hex(pt, v2):
                return -0.01  # Overlapping
        for pt in v2:
            if point_in_hex(pt, v1):
                return -0.01  # Overlapping
        
        return min_dist

    def compute_S(params, outer_alpha=0.0):
        """Compute the enclosing regular hexagon side length for a set of hexagons."""
        n_hex = len(params) // 3
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]

        # Collect all vertices
        all_pts = []
        for i in range(n_hex):
            verts = get_hex_verts(xs[i], ys[i], thetas[i])
            all_pts.append(verts)
        all_pts = np.vstack(all_pts)

        # Center the points for better numerical behavior
        centroid = np.mean(all_pts, axis=0)

        # Side normals at angles outer_alpha + k*pi/3 for k=0..5
        k_angles = outer_alpha + np.arange(6) * math.pi / 3
        normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)

        # Projection of each point onto each normal
        dots = np.sum((all_pts - centroid)[np.newaxis, :, :] * normals[:, np.newaxis, :], axis=-1)

        # For each normal direction, find the extent (max - min)
        extents = np.max(dots, axis=1) - np.min(dots, axis=1)
        max_extent = np.max(extents)

        # Side length S: for a regular hexagon with side S, the distance between
        # opposite sides is S * sqrt(3). So S = max_extent / sqrt(3)
        return max_extent / math.sqrt(3)

    def objective(params):
        if time.time() > deadline - safety_margin:
            return 1e10

        n_hex = len(params) // 3
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]

        # Compute exact polygon distances for hard constraint
        verts_list = [get_hex_verts(xs[i], ys[i], thetas[i]) for i in range(n_hex)]
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                dist = polygon_distance(verts_list[i], verts_list[j])
                if dist < 0:
                    # Overlapping: heavy penalty
                    penalty += abs(dist) * 1000.0
                elif dist < 0.01:
                    # Too close: moderate penalty
                    penalty += (0.01 - dist) * 100.0

        S = compute_S(params)
        return S + penalty

    # Use Differential Evolution for global search
    # Bounds: x, y in [-5, 5], angle in [-pi/3, pi/3] (hexagons have 6-fold symmetry)
    bounds = [(-5, 5), (-5, 5), (-math.pi/3, math.pi/3)] * N

    best_S = float('inf')
    best_params = None
    best_outer_alpha = 0.0

    # Run Differential Evolution with time-limited callback
    def de_callback(xk, convergence):
        nonlocal best_S, best_params
        if time.time() > deadline - safety_margin:
            return True
        return False

    try:
        de_result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.8,
            seed=42,
            callback=de_callback,
            polish=True,
            init='sobol'
        )
        
        if de_result.success and de_result.fun < best_S:
            best_S = de_result.fun
            best_params = de_result.x.copy()
        elif de_result.fun < best_S:
            best_S = de_result.fun
            best_params = de_result.x.copy()
    except Exception as e:
        print(f"DE failed: {e}")

    # If DE didn't find a good solution, try a heuristic fallback
    if best_params is None or best_S > 5.0:
        # Fallback: hexagonal lattice arrangement
        base_centers = [(0.0, 0.0)]
        for i in range(6):
            angle = i * math.pi / 3
            x = 2 * d * math.cos(angle)
            y = 2 * d * math.sin(angle)
            base_centers.append((x, y))
        
        ring2_centers = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) == 2:
                    spacing = 2 * d
                    x = spacing * (q + r / 2.0)
                    y = spacing * (d / 2.0) * r
                    ring2_centers.append((x, y))
        
        final_centers = base_centers + ring2_centers[:4]
        best_params = np.zeros(N * 3)
        for i, (cx, cy) in enumerate(final_centers[:N]):
            best_params[3 * i] = cx
            best_params[3 * i + 1] = cy
            best_params[3 * i + 2] = 0.0
        best_S = compute_S(best_params)

    # Optimize the outer hexagon rotation
    if time.time() < deadline - safety_margin:
        def outer_obj(alpha):
            return compute_S(best_params, alpha)

        try:
            res_outer = minimize_scalar(outer_obj, bounds=(0, math.pi / 3), method='bounded',
                                        options={'xatol': 1e-10})
            if res_outer.success:
                best_outer_alpha = res_outer.x
                best_S = res_outer.fun
        except Exception:
            best_outer_alpha = 0.0

    # Final refinement with local optimization from DE result
    if time.time() < deadline - safety_margin:
        def obj_refine(params):
            if time.time() > deadline - safety_margin:
                return 1e10
            n_hex = len(params) // 3
            xs = params[0::3]
            ys = params[1::3]
            thetas = params[2::3]
            verts_list = [get_hex_verts(xs[i], ys[i], thetas[i]) for i in range(n_hex)]
            penalty = 0.0
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    dist = polygon_distance(verts_list[i], verts_list[j])
                    if dist < 0:
                        penalty += abs(dist) * 1000.0
                    elif dist < 0.01:
                        penalty += (0.01 - dist) * 100.0
            S = compute_S(params, best_outer_alpha)
            return S + penalty

        try:
            bounds_refine = [(-5, 5), (-5, 5), (-math.pi/3, math.pi/3)] * N
            res_refine = minimize(obj_refine, best_params, method='L-BFGS-B',
                                  bounds=bounds_refine,
                                  options={'maxiter': 200, 'ftol': 1e-9})
            if res_refine.fun < best_S:
                best_S = res_refine.fun
                best_params = res_refine.x.copy()
        except Exception:
            pass

    # Construct the inner hexagons from best_params
    inner = []
    for i in range(N):
        x = best_params[3 * i]
        y = best_params[3 * i + 1]
        angle_deg = math.degrees(best_params[3 * i + 2])
        inner.append([x, y, angle_deg])

    return inner, (0.0, 0.0), best_S, math.degrees(best_outer_alpha)
