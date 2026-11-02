import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping, linprog
import time

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

    def compute_min_side_fixed_orientation(hexes, center=(0.0, 0.0)):
        """
        Calculate the minimum side length of a regular hexagon (centered at given center, fixed orientation)
        that contains all given hexagons.
        """
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = (h - center) @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        s = max_proj * 2.0 / np.sqrt(3.0)
        return s

    def get_hexes_from_params(params):
        """
        params: [x0, y0, a0, r1, th1, al1, r2, th2, al2]
        Structure: 1 center, 5 inner ring, 6 outer ring.
        """
        x0, y0, a0, r1, th1, al1, r2, th2, al2 = params
        
        hexes = []
        
        # Center hexagon
        hexes.append(hex_vertices(x0, y0, 1.0, a0))
        
        # Inner Ring (5 hexagons)
        for k in range(5):
            angle = th1 + 2 * np.pi * k / 5.0
            x1 = r1 * np.cos(angle)
            y1 = r1 * np.sin(angle)
            # Rotation: base rotation + angular position (to maintain local symmetry)
            a1 = al1 + angle
            hexes.append(hex_vertices(x1, y1, 1.0, a1))
            
        # Outer Ring (6 hexagons)
        for k in range(6):
            angle = th2 + 2 * np.pi * k / 6.0
            x2 = r2 * np.cos(angle)
            y2 = r2 * np.sin(angle)
            # Rotation: base rotation + angular position
            a2 = al2 + angle
            hexes.append(hex_vertices(x2, y2, 1.0, a2))
            
        return hexes

    def compute_smooth_objective(params, beta=50.0):
        """
        Smooth objective for global search using soft-max penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        hexes = get_hexes_from_params(params)
        
        s_min = compute_min_side_fixed_orientation(hexes)
        
        penalty = 0.0
        n = len(hexes)
        for i in range(n):
            for j in range(i + 1, n):
                # Quick bounding box check
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    # Soft-max smooth penalty with adaptive beta
                    effective_beta = min(beta, 100.0)
                    penalty += np.log(1.0 + np.exp(effective_beta * overlap)) / effective_beta
                    
        return s_min + penalty

    def compute_objective(params):
        """
        Objective: min_side + penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        hexes = get_hexes_from_params(params)
        
        s_min = compute_min_side_fixed_orientation(hexes)
        
        penalty = 0.0
        n = len(hexes)
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * 10.0
                    
        return s_min + penalty

    def compute_min_side_rotated(hexes, angle_deg):
        """
        Calculate min side length of outer hexagon with given rotation angle.
        """
        angle_rad = np.radians(angle_deg)
        normal_angles = angle_rad + np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        s = max_proj * 2.0 / np.sqrt(3.0)
        return s

    def optimize_outer_rotation(hexes):
        """
        Optimize the outer hexagon rotation angle to minimize side length.
        Uses golden section search over [0, 60) degrees.
        """
        best_angle = 0.0
        best_s = np.inf
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2  # golden ratio
        a, b = 0.0, 60.0
        
        # Evaluate at initial points
        x1 = b - gr * (b - a)
        x2 = a + gr * (b - a)
        f1 = compute_min_side_rotated(hexes, x1)
        f2 = compute_min_side_rotated(hexes, x2)
        
        # Iterate
        for _ in range(50):
            if time.time() > deadline:
                break
                
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - gr * (b - a)
                f1 = compute_min_side_rotated(hexes, x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + gr * (b - a)
                f2 = compute_min_side_rotated(hexes, x2)
        
        # Check endpoints and midpoint
        candidates = [0.0, 30.0, (a + b) / 2]
        for ang in candidates:
            s = compute_min_side_rotated(hexes, ang)
            if s < best_s:
                best_s = s
                best_angle = ang
        
        return best_s, best_angle

    # --- Optimization ---
    
    # Bounds for 1-5-6 structure
    # x0, y0, a0: Center hex
    # r1, th1, al1: Inner ring (5)
    # r2, th2, al2: Outer ring (6)
    
    bounds = [
        (-1.0, 1.0),  # x0
        (-1.0, 1.0),  # y0
        (0.0, np.pi/3), # a0
        (1.5, 4.0),   # r1 (inner ring radius)
        (0.0, 2*np.pi), # th1
        (0.0, np.pi/3), # al1
        (3.0, 6.0),   # r2 (outer ring radius)
        (0.0, 2*np.pi), # th2
        (0.0, np.pi/3), # al2
    ]
    
    best_s = np.inf
    best_params = None
    best_outer_angle = 0.0
    
    # Phase 1: Differential Evolution with smooth objective
    try:
        res_de = differential_evolution(
            lambda p: compute_smooth_objective(p, beta=30.0),
            bounds,
            maxiter=300,
            popsize=25,
            tol=1e-10,
            seed=42,
            polish=True
        )
        if res_de.fun < best_s:
            best_s = res_de.fun
            best_params = res_de.x.copy()
    except Exception:
        pass
    
    # Phase 1b: Additional DE run with different seed and higher beta
    if time.time() < deadline - 60:
        try:
            res_de2 = differential_evolution(
                lambda p: compute_smooth_objective(p, beta=50.0),
                bounds,
                maxiter=200,
                popsize=20,
                tol=1e-10,
                seed=123,
                polish=True
            )
            if res_de2.fun < best_s:
                best_s = res_de2.fun
                best_params = res_de2.x.copy()
        except Exception:
            pass

    # Phase 2: Basin Hopping
    if best_params is not None:
        try:
            def local_minimize(x):
                res = minimize(
                    compute_objective,
                    x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-14}
                )
                return res.fun, res.x, {}
                
            res_bh = basinhopping(
                local_minimize,
                best_params,
                niter=30,
                T=0.5,
                stepsize=0.3,
                seed=42
            )
            if res_bh.fun < best_s:
                best_s = res_bh.fun
                best_params = res_bh.x.copy()
        except Exception:
            pass

    # Phase 3: Local Refinement with high precision
    if best_params is not None:
        try:
            res_local = minimize(
                compute_objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res_local.fun < best_s:
                best_s = res_local.fun
                best_params = res_local.x.copy()
        except Exception:
            pass

    # Phase 4: Random Restarts with better sampling
    if time.time() < deadline - 15:
        n_restarts = 20
        np.random.seed(123)
        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Sample from a broader range with some structure
            x0 = np.random.uniform(-0.5, 0.5)
            y0 = np.random.uniform(-0.5, 0.5)
            a0 = np.random.uniform(0, np.pi/3)
            r1 = np.random.uniform(1.8, 3.5)
            th1 = np.random.uniform(0, 2*np.pi)
            al1 = np.random.uniform(0, np.pi/3)
            r2 = np.random.uniform(3.5, 5.5)
            th2 = np.random.uniform(0, 2*np.pi)
            al2 = np.random.uniform(0, np.pi/3)
            
            p = np.array([x0, y0, a0, r1, th1, al1, r2, th2, al2])
            
            try:
                res_local = minimize(
                    compute_objective,
                    p,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params = res_local.x.copy()
            except Exception:
                pass

    # Phase 5: Final high-precision polish
    if best_params is not None and time.time() < deadline - 5:
        try:
            res_polish = minimize(
                compute_objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-18, 'maxls': 300}
            )
            if res_polish.fun < best_s:
                best_s = res_polish.fun
                best_params = res_polish.x.copy()
        except Exception:
            pass
        
        # Phase 5b: Nelder-Mead refinement for final touch
        if best_params is not None and time.time() < deadline - 3:
            try:
                res_nm = minimize(
                    compute_objective,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                if res_nm.fun < best_s:
                    best_s = res_nm.fun
                    best_params = res_nm.x.copy()
            except Exception:
                pass

    # Fallback
    if best_params is None:
        best_params = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0])
        
    x0, y0, a0, r1, th1, al1, r2, th2, al2 = best_params
    
    # Generate the 12 inner hexagons
    inner = []
    hexes_list = []
    
    # Center
    inner.append([float(x0), float(y0), float(np.degrees(a0))])
    hexes_list.append(hex_vertices(x0, y0, 1.0, a0))
    
    # Inner Ring
    for k in range(5):
        angle = th1 + 2 * np.pi * k / 5.0
        x1 = r1 * np.cos(angle)
        y1 = r1 * np.sin(angle)
        a1 = al1 + angle
        
        inner.append([float(x1), float(y1), float(np.degrees(a1))])
        hexes_list.append(hex_vertices(x1, y1, 1.0, a1))
        
    # Outer Ring
    for k in range(6):
        angle = th2 + 2 * np.pi * k / 6.0
        x2 = r2 * np.cos(angle)
        y2 = r2 * np.sin(angle)
        a2 = al2 + angle
        
        inner.append([float(x2), float(y2), float(np.degrees(a2))])
        hexes_list.append(hex_vertices(x2, y2, 1.0, a2))
    
    # Optimize outer hexagon rotation
    s_final, outer_angle_deg = optimize_outer_rotation(hexes_list)
    
    outer_center = (0.0, 0.0)
    outer_side = float(s_final)

    return inner, outer_center, outer_side, float(outer_angle_deg)
