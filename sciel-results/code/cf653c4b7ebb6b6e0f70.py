import numpy as np
from scipy.optimize import minimize, basinhopping
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # --- Geometry Helpers ---
    
    def hex_vertices(cx, cy, side, angle_rad):
        """Returns 6x2 array of vertices for a regular hexagon."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        """Returns 6x2 array of outward normals for a CCW polygon."""
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def sat_overlap(v1, v2):
        """
        Returns min overlap depth (negative if disjoint) using SAT.
        v1, v2 are 6x2 arrays of vertices.
        """
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
        min_overlap = np.min(overlaps)
        
        return min_overlap

    def compute_min_side_smooth(hexes, k=20.0):
        """
        Calculate the minimum side length of a regular hexagon (centered at origin, fixed orientation)
        that contains all given hexagons, using a smooth approximation.
        """
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        all_maxes = []
        for h in hexes:
            proj = h @ normals.T
            # proj is 6x6. We want the max projection for each of the 6 normals.
            max_per_normal = np.max(proj, axis=0)
            all_maxes.extend(max_per_normal)
            
        all_maxes = np.array(all_maxes)
        
        # Soft max to make it differentiable
        # L_raw = max(all_maxes)
        # L_smooth = log(sum(exp(k * all_maxes))) / k
        # To avoid overflow, subtract max
        max_val = np.max(all_maxes)
        L_smooth = (max_val + np.log(np.sum(np.exp(k * (all_maxes - max_val)))) / k)
        
        # L = L_raw * 2 / sqrt(3)
        s = L_smooth * 2.0 / np.sqrt(3.0)
        return s

    def compute_objective(params, k_soft=20.0, penalty_weight=10.0):
        """
        params: 36 floats [x1, y1, theta1, ..., x12, y12, theta12]
        Returns outer side length + penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        hexes = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3 + 1]
            theta = params[i*3 + 2]
            hexes.append(hex_vertices(x, y, 1.0, theta))
            
        s_min = compute_min_side_smooth(hexes, k_soft)
        
        penalty = 0.0
        # Check overlaps
        for i in range(N):
            min1 = hexes[i].min(axis=0)
            max1 = hexes[i].max(axis=0)
            for j in range(i + 1, N):
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                # Bounding box check
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    # Smooth penalty
                    penalty += (overlap * penalty_weight) ** 2
                    
        return s_min + penalty

    def get_hexes_from_params(params):
        hexes = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3 + 1]
            theta = params[i*3 + 2]
            hexes.append(hex_vertices(x, y, 1.0, theta))
        return hexes

    def compute_exact_side(hexes):
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        s = max_proj * 2.0 / np.sqrt(3.0)
        return s

    # --- Initialization Strategies ---
    
    def init_honeycomb():
        """
        1 center, 6 in ring 1, 5 in ring 2.
        Distance d = 2.0 (safe, will be optimized down).
        """
        params = np.zeros(N * 3)
        d = 2.0
        # Center
        params[0:3] = [0, 0, 0]
        
        # Ring 1
        for i in range(6):
            angle = i * np.pi / 3
            x = d * np.cos(angle)
            y = d * np.sin(angle)
            theta = 0.0
            idx = (i + 1) * 3
            params[idx:idx+3] = [x, y, theta]
            
        # Ring 2 (5 hexes)
        # Angles: 0, 60, 120, 180, 240 (skip 300)
        d2 = 2 * d
        angles_ring2 = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3]
        for i, angle in enumerate(angles_ring2):
            x = d2 * np.cos(angle)
            y = d2 * np.sin(angle)
            theta = 0.0
            idx = (i + 7) * 3
            params[idx:idx+3] = [x, y, theta]
            
        return params

    def init_symmetric_two_orbits():
        """
        2 orbits of 6.
        """
        params = np.zeros(N * 3)
        d1 = 1.5
        d2 = 3.0
        for k in range(6):
            angle = k * np.pi / 3
            # Orbit 1
            x1 = d1 * np.cos(angle)
            y1 = d1 * np.sin(angle)
            idx1 = k * 3
            params[idx1:idx1+3] = [x1, y1, 0.0]
            
            # Orbit 2
            x2 = d2 * np.cos(angle)
            y2 = d2 * np.sin(angle)
            idx2 = (k + 6) * 3
            params[idx2:idx2+3] = [x2, y2, 0.0]
            
        return params

    def init_random():
        params = np.random.uniform(-3, 3, N * 3)
        # Random rotations
        for i in range(N):
            params[i*3+2] = np.random.uniform(0, np.pi/3)
        return params

    # --- Optimization ---
    
    best_params = None
    best_score = np.inf
    
    inits = [init_honeycomb(), init_symmetric_two_orbits(), init_random()]
    
    # Phase 1: Local Optimization from multiple starts
    for p0 in inits:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(
                compute_objective,
                p0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Basin Hopping
    if best_params is not None and time.time() < deadline - 20:
        try:
            def local_minimize(x):
                res = minimize(
                    compute_objective,
                    x,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                return res.fun, res.x, {}
                
            res_bh = basinhopping(
                local_minimize,
                best_params,
                niter=50,
                T=0.5,
                stepsize=0.5,
                seed=42,
                minimizer_kwargs={'options': {'maxiter': 100}}
            )
            if res_bh.fun < best_score:
                best_score = res_bh.fun
                best_params = res_bh.x.copy()
        except Exception:
            pass

    # Phase 3: Final Polish
    if best_params is not None and time.time() < deadline - 5:
        try:
            res_local = minimize(
                compute_objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_local.fun < best_score:
                best_score = res_local.fun
                best_params = res_local.x.copy()
        except Exception:
            pass

    # Fallback
    if best_params is None:
        best_params = init_honeycomb()

    # Generate output
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3 + 1]
        theta_deg = np.degrees(best_params[i*3 + 2])
        inner.append([x, y, theta_deg])
        
    hexes = get_hexes_from_params(best_params)
    outer_side = compute_exact_side(hexes)
    
    outer_center = (0.0, 0.0)
    outer_angle_deg = 0.0
    
    return inner, outer_center, outer_side, outer_angle_deg
