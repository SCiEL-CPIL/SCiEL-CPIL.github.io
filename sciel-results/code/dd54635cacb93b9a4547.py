import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

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

    def compute_min_side(hexes):
        """
        Calculate the minimum side length of a regular hexagon (centered at origin, fixed orientation)
        that contains all given hexagons.
        """
        # The outer hexagon has fixed orientation (angle 0).
        # Its supporting lines are defined by the 6 outward normals of a hexagon with side 1, angle 0.
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        # For a regular hexagon of side s, the distance from center to side is s * sqrt(3) / 2.
        # So max_proj = s * sqrt(3) / 2
        return max_proj * 2.0 / np.sqrt(3.0)

    def smooth_max(values, temperature=0.01):
        """
        Smooth approximation of max using log-sum-exp.
        """
        if len(values) == 0:
            return 0.0
        v_max = np.max(values)
        if v_max == -np.inf:
            return -np.inf
        shifted = (values - v_max) / temperature
        # Clip to avoid overflow
        shifted = np.clip(shifted, -50, 50)
        result = v_max + temperature * np.log(np.sum(np.exp(shifted)))
        return result

    def compute_objective_exact(params):
        """
        params: [x1, y1, a1, x2, y2, a2]
        Generates 12 hexagons using 6-fold symmetry.
        Returns min_side + penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        x1, y1, a1, x2, y2, a2 = params
        
        # Generate the 12 hexagons
        # Base hexagons
        base1 = hex_vertices(x1, y1, 1.0, a1)
        base2 = hex_vertices(x2, y2, 1.0, a2)
        
        hexes = []
        for k in range(6):
            angle = np.radians(60 * k)
            c = np.cos(angle)
            s_ = np.sin(angle)
            rot = np.array([[c, -s_], [s_, c]])
            
            # Rotate base1 and base2
            h1 = base1 @ rot.T
            h2 = base2 @ rot.T
            
            hexes.append(h1)
            hexes.append(h2)
        
        # Compute minimum outer side length
        min_s = compute_min_side(hexes)
        
        penalty = 0.0
        
        # Check overlaps
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick bounding box check
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * 10.0
                    
        return min_s + penalty

    def compute_objective_smooth(params, temperature=0.01):
        """
        Smooth version of the objective for gradient-based polishing.
        Uses log-sum-exp for max projection and overlap.
        """
        if time.time() > deadline:
            return 1e10
            
        x1, y1, a1, x2, y2, a2 = params
        
        # Generate the 12 hexagons
        # Base hexagons
        base1 = hex_vertices(x1, y1, 1.0, a1)
        base2 = hex_vertices(x2, y2, 1.0, a2)
        
        hexes = []
        for k in range(6):
            angle = np.radians(60 * k)
            c = np.cos(angle)
            s_ = np.sin(angle)
            rot = np.array([[c, -s_], [s_, c]])
            
            # Rotate base1 and base2
            h1 = base1 @ rot.T
            h2 = base2 @ rot.T
            
            hexes.append(h1)
            hexes.append(h2)
        
        # Compute minimum outer side length using smooth max
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Collect all projections for smooth max
        all_projs = []
        for h in hexes:
            proj = h @ normals.T
            all_projs.extend(proj.flatten())
        
        # Smooth max of all projections
        max_proj_smooth = smooth_max(np.array(all_projs), temperature)
        min_s_smooth = max_proj_smooth * 2.0 / np.sqrt(3.0)
        
        penalty = 0.0
        
        # Check overlaps with smooth penalty
        # We only care about positive overlaps
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick bounding box check to skip far pairs
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    # Use smooth square or just smooth max of overlap?
                    # Since overlap is a single scalar per pair, we can just use it directly if positive.
                    # To make it smooth, we can use softplus or similar, but overlap itself is continuous 
                    # except at the exact touch point. The SAT min is non-smooth.
                    # For now, let's just accumulate the raw overlap if positive, 
                    # relying on the smoothness of the geometry to help.
                    # Or better, use a smooth approximation of max(0, overlap).
                    # softplus(x) = log(1 + exp(x)) approx max(0, x)
                    penalty += np.log(1 + np.exp(overlap / temperature)) * temperature * 10.0
                    
        return min_s_smooth + penalty

    # --- Optimization ---
    
    # Bounds for the 2 base hexagons
    # x, y in [-4, 4]
    # angle in [0, pi/3]
    bounds = [
        (-4.0, 4.0),  # x1
        (-4.0, 4.0),  # y1
        (0.0, np.pi/3), # a1
        (-4.0, 4.0),  # x2
        (-4.0, 4.0),  # y2
        (0.0, np.pi/3), # a2
    ]
    
    best_s = np.inf
    best_params = None
    
    # Use Differential Evolution for global search on the EXACT objective
    # DE is robust to non-smoothness
    try:
        res = differential_evolution(
            compute_objective_exact,
            bounds,
            maxiter=300,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=True
        )
        
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x
            
    except Exception as e:
        pass
    
    # Polish with L-BFGS-B on the SMOOTH objective
    # This should navigate the local landscape better than the exact non-smooth one
    if best_params is not None:
        try:
            # Start with a slightly higher temperature for smoothness, then reduce?
            # Or just use a fixed small temperature.
            res_local = minimize(
                lambda p: compute_objective_smooth(p, 0.01),
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            # Evaluate the EXACT objective at this new point
            exact_val = compute_objective_exact(res_local.x)
            if exact_val < best_s:
                best_s = exact_val
                best_params = res_local.x
        except Exception:
            pass
            
    # Final polish on EXACT objective to clean up any smoothing artifacts
    if best_params is not None:
        try:
            res_exact_polish = minimize(
                compute_objective_exact,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_exact_polish.fun < best_s:
                best_s = res_exact_polish.fun
                best_params = res_exact_polish.x
        except Exception:
            pass

    # Construct the final answer from best_params
    if best_params is None:
        # Fallback: Simple grid or known approximate
        for _ in range(10):
            p = np.array([
                np.random.uniform(1, 3), np.random.uniform(-0.5, 0.5), np.random.uniform(0, np.pi/3),
                np.random.uniform(2, 4), np.random.uniform(-0.5, 0.5), np.random.uniform(0, np.pi/3)
            ])
            val = compute_objective_exact(p)
            if val < best_s:
                best_s = val
                best_params = p
            try:
                res_local = minimize(
                    compute_objective_exact,
                    p,
                    method='Nelder-Mead',
                    options={'maxiter': 100}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params = res_local.x
            except Exception:
                pass
            if time.time() > deadline - 5:
                break
                
    x1, y1, a1, x2, y2, a2 = best_params
    
    # Generate the 12 inner hexagons
    inner = []
    for k in range(6):
        angle = np.radians(60 * k)
        c = np.cos(angle)
        s_ = np.sin(angle)
        
        # Rotate (x1, y1)
        x1_rot = x1 * c - y1 * s_
        y1_rot = x1 * s_ + y1 * c
        a1_rot = a1 + angle
        
        # Rotate (x2, y2)
        x2_rot = x2 * c - y2 * s_
        y2_rot = x2 * s_ + y2 * c
        a2_rot = a2 + angle
        
        inner.append([x1_rot, y1_rot, np.degrees(a1_rot)])
        inner.append([x2_rot, y2_rot, np.degrees(a2_rot)])
        
    # Compute the actual minimum outer side length
    # Reconstruct hexes to compute min side
    base1 = hex_vertices(x1, y1, 1.0, a1)
    base2 = hex_vertices(x2, y2, 1.0, a2)
    
    hexes = []
    for k in range(6):
        angle = np.radians(60 * k)
        c = np.cos(angle)
        s_ = np.sin(angle)
        rot = np.array([[c, -s_], [s_, c]])
        
        h1 = base1 @ rot.T
        h2 = base2 @ rot.T
        
        hexes.append(h1)
        hexes.append(h2)
    
    outer_side = compute_min_side(hexes)
    
    outer_center = (0.0, 0.0)
    outer_angle_deg = 0.0
    
    return inner, outer_center, outer_side, outer_angle_deg
