import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

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

    def compute_min_side(hexes, outer_center, outer_angle_rad):
        """
        Calculate the minimum side length of a regular hexagon 
        (centered at outer_center, rotated by outer_angle_rad)
        that contains all given hexagons.
        """
        # Normals for the outer hexagon
        # Outer hexagon vertices are at angles outer_angle_rad + k*60
        # Normals are at outer_angle_rad + 30 + k*60
        normal_angles = outer_angle_rad + np.radians(30) + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            # Shift to outer center
            h_shifted = h - outer_center
            proj = h_shifted @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        # max_proj = s * sqrt(3) / 2
        # s = max_proj * 2 / sqrt(3)
        s = max_proj * 2.0 / np.sqrt(3.0)
        return s

    def get_hexes_from_params(params):
        """
        params: [r1, theta1, alpha1, r2, theta2, alpha2]
        Returns list of 12 hexagons (6x2 arrays).
        Uses 6-fold symmetry.
        """
        r1, th1, al1, r2, th2, al2 = params
        
        hexes = []
        # Orbit 1
        for k in range(6):
            angle = np.radians(60 * k)
            # Position
            x1 = r1 * np.cos(th1 + angle)
            y1 = r1 * np.sin(th1 + angle)
            # Rotation: base rotation + angle of position to maintain symmetry relative to container
            a1 = al1 + angle
            hexes.append(hex_vertices(x1, y1, 1.0, a1))
            
        # Orbit 2
        for k in range(6):
            angle = np.radians(60 * k)
            # Position
            x2 = r2 * np.cos(th2 + angle)
            y2 = r2 * np.sin(th2 + angle)
            # Rotation
            a2 = al2 + angle
            hexes.append(hex_vertices(x2, y2, 1.0, a2))
            
        return hexes

    def compute_objective_smooth(params):
        """
        Smooth objective for global search (DE, Basin Hopping).
        Uses softplus for penalty to be differentiable.
        Also optimizes outer hexagon rotation and center (center is centroid for symmetry).
        """
        if time.time() > deadline:
            return 1e10
            
        hexes = get_hexes_from_params(params)
        
        # For symmetric configurations, the optimal outer center is the origin (0,0)
        # and the optimal rotation is 0.0 relative to the symmetry axis.
        # However, to be robust, we can scan a few outer angles or just fix it.
        # Given the 6-fold symmetry of the inner configuration (if al1, al2 are consistent),
        # the optimal outer hexagon should align with the symmetry.
        # Let's assume outer center = (0,0) and outer angle = 0.
        # If the inner config is not perfectly symmetric due to optimizer noise, 
        # the centroid might shift, but for this parameterization, it's centered at 0.
        
        s_min = compute_min_side(hexes, np.array([0.0, 0.0]), 0.0)
        
        penalty = 0.0
        # Check overlaps
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                # Bounding box check
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    # Softplus: log(1 + exp(k*x)) approximates max(0, x) smoothly
                    penalty += np.log(1.0 + np.exp(10.0 * overlap))
                    
        return s_min + penalty

    def compute_objective_exact(params):
        """
        Exact objective for final polish (L-BFGS-B).
        Uses hard penalty.
        """
        if time.time() > deadline:
            return 1e10
            
        hexes = get_hexes_from_params(params)
        
        s_min = compute_min_side(hexes, np.array([0.0, 0.0]), 0.0)
        
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * 100.0 # High penalty weight
                    
        return s_min + penalty

    def refine_outer(hexes):
        """
        Refine the outer hexagon center and rotation to minimize side length.
        Returns (side, center, angle_deg).
        """
        # 1. Get all vertices
        all_verts = np.vstack(hexes)
        
        # 2. Initial guess for center: centroid
        center = np.mean(all_verts, axis=0)
        offsets = all_verts - center
        
        # 3. Optimize rotation phi
        def obj_phi(phi_rad):
            max_extent = 0.0
            for k in range(6):
                normal_angle = phi_rad + np.pi/6 + k * np.pi/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                max_proj = np.max(np.abs(proj))
                
                side_k = (2.0 / math.sqrt(3)) * max_proj
                if side_k > max_extent:
                    max_extent = side_k
            return max_extent

        # Coarse scan for phi
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 5):
            phi = np.radians(phi_deg)
            s = obj_phi(phi)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Fine scan
        for phi_deg in np.arange(best_phi - 5, best_phi + 5, 0.5):
            if phi_deg < 0 or phi_deg >= 60: continue
            phi = np.radians(phi_deg)
            s = obj_phi(phi)
            if s < best_side:
                best_side = s
                best_phi = phi_deg

        # Polish with Nelder-Mead
        try:
            res = minimize(obj_phi, np.radians(best_phi), method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 100})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = np.degrees(res.x[0]) % 60.0
        except:
            pass

        # Note: Optimizing center is harder because obj_phi depends on center.
        # For symmetric configurations, center should be (0,0).
        # Let's stick with centroid for center, or (0,0) if we enforce symmetry.
        # Given the parameterization generates symmetric configs, (0,0) is likely optimal.
        # But let's use the computed centroid to be safe.
        
        return best_side, center, best_phi

    # --- Optimization ---
    
    bounds = [
        (0.5, 5.0),  # r1
        (0.0, 2*np.pi), # theta1
        (0.0, np.pi/3), # alpha1
        (0.5, 5.0),  # r2
        (0.0, 2*np.pi), # theta2
        (0.0, np.pi/3), # alpha2
    ]
    
    best_s = np.inf
    best_params = None
    
    # Phase 1: Differential Evolution
    try:
        res_de = differential_evolution(
            compute_objective_smooth,
            bounds,
            maxiter=200,
            popsize=30,
            tol=1e-8,
            seed=42,
            polish=True,
            polish_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 100}}
        )
        if res_de.fun < best_s:
            best_s = res_de.fun
            best_params = res_de.x.copy()
    except Exception:
        pass

    # Phase 2: Basin Hopping
    if best_params is not None and time.time() < deadline - 20:
        try:
            def local_minimize(x):
                res = minimize(
                    compute_objective_smooth,
                    x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100}
                )
                return res.fun, res.x, {}
                
            res_bh = basinhopping(
                local_minimize,
                best_params,
                niter=30,
                T=0.5,
                stepsize=0.5,
                minimizer_kwargs={'options': {'maxiter': 100}},
                seed=42
            )
            if res_bh.fun < best_s:
                best_s = res_bh.fun
                best_params = res_bh.x.copy()
        except Exception:
            pass

    # Phase 3: High-Precision Local Refinement
    if best_params is not None and time.time() < deadline - 5:
        try:
            res_local = minimize(
                compute_objective_exact,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if res_local.fun < best_s:
                best_s = res_local.fun
                best_params = res_local.x.copy()
        except Exception:
            pass

    # Fallback
    if best_params is None:
        best_params = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])

    # Generate final configuration
    hexes = get_hexes_from_params(best_params)
    
    # Refine outer hexagon
    outer_side, outer_center, outer_angle_deg = refine_outer(hexes)
    
    # Ensure outer_center is returned as (x, y)
    outer_center = (float(outer_center[0]), float(outer_center[1]))
    
    # Construct inner list
    r1, th1, al1, r2, th2, al2 = best_params
    inner = []
    for k in range(6):
        angle = np.radians(60 * k)
        x1 = r1 * np.cos(th1 + angle)
        y1 = r1 * np.sin(th1 + angle)
        a1 = al1 + angle
        inner.append([x1, y1, np.degrees(a1)])
        
        x2 = r2 * np.cos(th2 + angle)
        y2 = r2 * np.sin(th2 + angle)
        a2 = al2 + angle
        inner.append([x2, y2, np.degrees(a2)])

    return inner, outer_center, outer_side, outer_angle_deg
