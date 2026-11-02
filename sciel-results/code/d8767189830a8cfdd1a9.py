import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def sat_overlap(v1, v2):
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        return np.min(overlaps)

    # Outer hexagon normals for "pointy-top" or "flat-top"?
    # Let's assume the outer hexagon is fixed orientation to simplify, 
    # but the inner can rotate. 
    # If we fix outer hexagon orientation, we just minimize the max projection.
    # The returned outer_angle_deg is 0.0, implying fixed orientation.
    # Normals at 30, 90, ... correspond to a hexagon with vertices at 0, 60, ...?
    # Let's verify.
    # If normal is at 90 (Up), edge is Horizontal.
    # If normal is at 30, edge is at 120.
    # This is a standard hexagon.
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    outer_normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    def compute_min_side(hexes):
        max_proj = 0.0
        for h in hexes:
            proj = h @ outer_normals.T
            max_proj = max(max_proj, np.max(proj))
        return max_proj * 2.0 / np.sqrt(3.0)

    def compute_smooth_min_side(hexes, beta=100.0):
        all_projs = []
        for h in hexes:
            proj = h @ outer_normals.T
            all_projs.append(proj)
        all_projs = np.concatenate(all_projs, axis=1)
        scaled = beta * all_projs
        max_scaled = np.max(scaled, axis=0)
        # Avoid overflow
        max_scaled = np.where(np.isinf(max_scaled), 0, max_scaled)
        soft_max = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled), axis=0)) / beta) / beta
        return np.max(soft_max) * 2.0 / np.sqrt(3.0)

    def compute_overlap_penalty(hexes, k=100.0):
        penalty = 0.0
        n = len(hexes)
        for i in range(n):
            min1 = hexes[i].min(axis=0)
            max1 = hexes[i].max(axis=0)
            for j in range(i + 1, n):
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                # Bounding box check
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[0]:
                    continue
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    # Smooth penalty: log(1 + exp(k*overlap)) approximates max(0, k*overlap)
                    penalty += np.log1p(np.exp(k * overlap))
        return penalty

    def objective_free(params, beta=100.0, k=100.0):
        # params: 12 * 3 = 36 values: x1, y1, a1, x2, y2, a2, ...
        hexes = []
        for i in range(12):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            hexes.append(hex_vertices(x, y, 1.0, a))
        
        s_min = compute_smooth_min_side(hexes, beta=beta)
        penalty = compute_overlap_penalty(hexes, k=k)
        return s_min + penalty

    def generate_hexes_6fold(params):
        # 2 orbits of 6
        # params: x1, y1, a1, x2, y2, a2
        hexes = []
        x1, y1, a1, x2, y2, a2 = params
        for k in range(6):
            angle = np.radians(60 * k)
            c = np.cos(angle)
            s = np.sin(angle)
            x1r = x1 * c - y1 * s
            y1r = x1 * s + y1 * c
            a1r = a1 + angle
            x2r = x2 * c - y2 * s
            y2r = x2 * s + y2 * c
            a2r = a2 + angle
            hexes.append(hex_vertices(x1r, y1r, 1.0, a1r))
            hexes.append(hex_vertices(x2r, y2r, 1.0, a2r))
        return hexes

    def objective_6fold(params, beta=100.0, k=100.0):
        hexes = generate_hexes_6fold(params)
        s_min = compute_smooth_min_side(hexes, beta=beta)
        penalty = compute_overlap_penalty(hexes, k=k)
        return s_min + penalty

    best_s = np.inf
    best_params_free = None
    best_params_6 = None
    best_type = None

    # --- Free Search ---
    # Bounds
    bounds_free = []
    for _ in range(12):
        bounds_free.extend([(-5.0, 5.0), (-5.0, 5.0), (0.0, 2*np.pi)])
    
    # Initial guesses for Free
    # 1. Random
    # 2. Hexagonal Grid
    
    def get_grid_init():
        # Hexagonal grid centers
        # u = (sqrt(3), 0), v = (sqrt(3)/2, 1.5)
        u = np.array([np.sqrt(3), 0])
        v = np.array([np.sqrt(3)/2, 1.5])
        centers = []
        # Pick 12 points from a small grid
        # i, j in range(-2, 3)
        for i in range(-2, 3):
            for j in range(-2, 3):
                if len(centers) >= 12:
                    break
                # Simple selection: compact cluster
                dist = np.linalg.norm(i*u + j*v)
                centers.append(i*u + j*v)
        # Sort by distance to center to get compact cluster
        centers.sort(key=lambda c: np.linalg.norm(c))
        centers = centers[:12]
        
        # Center the cluster
        centroid = np.mean(centers, axis=0)
        centers = [c - centroid for c in centers]
        
        params = []
        for c in centers:
            params.extend([c[0], c[1], 0.0]) # Angle 0
        return np.array(params)

    inits_free = []
    # Add grid init
    inits_free.append(get_grid_init())
    # Add random inits
    for _ in range(10):
        p = np.random.uniform(-2, 2, size=36)
        # Random angles
        p[2::3] = np.random.uniform(0, 2*np.pi, size=12)
        inits_free.append(p)
        
    for init in inits_free:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                lambda p: objective_free(p),
                init,
                method='L-BFGS-B',
                bounds=bounds_free,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if res.fun < best_s:
                best_s = res.fun
                best_params_free = res.x.copy()
                best_type = 'free'
        except Exception:
            pass

    # --- 6-fold Symmetry Search ---
    bounds_6fold = [
        (-5.0, 5.0), (-5.0, 5.0), (0.0, 2*np.pi),
        (-5.0, 5.0), (-5.0, 5.0), (0.0, 2*np.pi)
    ]
    
    inits_6 = []
    # Guess: Two rings
    # Ring 1 radius ~ 1.7, Ring 2 radius ~ 3.4
    inits_6.append(np.array([1.7, 0, 0, 3.4, 0, 0]))
    inits_6.append(np.array([1.5, 0.5, 0, 3.0, 0.5, 0]))
    for _ in range(10):
        p = np.random.uniform(-3, 3, size=6)
        p[2] = np.random.uniform(0, 2*np.pi)
        p[5] = np.random.uniform(0, 2*np.pi)
        inits_6.append(p)

    for init in inits_6:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                lambda p: objective_6fold(p),
                init,
                method='L-BFGS-B',
                bounds=bounds_6fold,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if res.fun < best_s:
                best_s = res.fun
                best_params_6 = res.x.copy()
                best_type = '6fold'
        except Exception:
            pass

    # Select best
    if best_type == 'free':
        final_params = best_params_free
        hexes = []
        for i in range(12):
            x = final_params[3*i]
            y = final_params[3*i+1]
            a = final_params[3*i+2]
            hexes.append(hex_vertices(x, y, 1.0, a))
    else:
        final_params = best_params_6
        hexes = generate_hexes_6fold(final_params)

    # Fallback
    if not hexes:
        hexes = [hex_vertices(0, 0, 1.0, 0)] * 12

    inner = []
    for h in hexes:
        cx = np.mean(h[:, 0])
        cy = np.mean(h[:, 1])
        edge = h[1] - h[0]
        angle = np.arctan2(edge[1], edge[0])
        inner.append([cx, cy, np.degrees(angle) % 360])

    s_final = compute_min_side(hexes)
    outer_center = (0.0, 0.0)
    outer_side = s_final
    outer_angle_deg = 0.0

    return inner, outer_center, outer_side, outer_angle_deg
