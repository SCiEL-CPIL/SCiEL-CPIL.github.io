"""
Strategy: 6-fold symmetric honeycomb packing with flexible 2-base-hexagon parameterization.
Uses smooth log-sum-exp objectives for both outer hexagon containment and overlap penalties.
Multi-phase optimization: differential evolution for global search, L-BFGS-B for smooth
polishing, Nelder-Mead for final exact polishing.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

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

    def sat_overlap_depth(v1, v2):
        """
        Returns minimum overlap depth (negative if disjoint) using SAT.
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

    def compute_min_side_exact(hexes, outer_angle_rad=0.0):
        """
        Calculate the minimum side length of a regular hexagon (centered at origin,
        given orientation) that contains all given hexagons.
        """
        normal_angles = outer_angle_rad + np.radians(30) + np.radians(60) * np.arange(6)
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
        shifted = np.clip(shifted, -50, 50)
        result = v_max + temperature * np.log(np.sum(np.exp(shifted)))
        return result

    def generate_hexes(params):
        """
        params: [x1, y1, a1, x2, y2, a2]
        Generates 12 hexagons using 6-fold symmetry.
        Returns list of 12 (6x2) vertex arrays.
        """
        x1, y1, a1, x2, y2, a2 = params
        
        hexes = []
        for k in range(6):
            angle = np.radians(60 * k)
            c = np.cos(angle)
            s_ = np.sin(angle)
            rot = np.array([[c, -s_], [s_, c]])
            
            # Base hexagons rotated by symmetry
            base1 = hex_vertices(x1, y1, 1.0, a1)
            base2 = hex_vertices(x2, y2, 1.0, a2)
            
            h1 = base1 @ rot.T
            h2 = base2 @ rot.T
            
            hexes.append(h1)
            hexes.append(h2)
        
        return hexes

    def compute_objective_smooth(params, temperature=0.01):
        """
        Smooth version of the objective for gradient-based polishing.
        Uses log-sum-exp for max projection and smooth overlap penalty.
        """
        if time.time() > deadline - 2:
            return 1e10
            
        hexes = generate_hexes(params)
        
        # Compute minimum outer side length using smooth max
        outer_angle_rad = 0.0  # Fixed outer hexagon orientation for simplicity
        normal_angles = outer_angle_rad + np.radians(30) + np.radians(60) * np.arange(6)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Collect all projections for smooth max
        all_projs = []
        for h in hexes:
            proj = h @ normals.T
            all_projs.extend(proj.flatten())
        
        max_proj_smooth = smooth_max(np.array(all_projs), temperature)
        min_s_smooth = max_proj_smooth * 2.0 / np.sqrt(3.0)
        
        penalty = 0.0
        
        # Check overlaps with smooth penalty
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick bounding box check to skip far pairs
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap_depth(hexes[i], hexes[j])
                if overlap > 0:
                    # Use softplus for smooth max(0, overlap)
                    penalty += np.log(1 + np.exp(overlap / temperature)) * temperature * 10.0
        
        return min_s_smooth + penalty

    def compute_objective_exact(params):
        """
        Exact objective with disjointness penalty.
        """
        if time.time() > deadline - 2:
            return 1e10
            
        hexes = generate_hexes(params)
        min_s = compute_min_side_exact(hexes)
        
        penalty = 0.0
        
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick bounding box check
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap_depth(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * 10.0
        
        return min_s + penalty

    def is_valid(params):
        """Check if all hexagons are pairwise disjoint."""
        hexes = generate_hexes(params)
        
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                
                overlap = sat_overlap_depth(hexes[i], hexes[j])
                if overlap > 1e-9:
                    return False
        
        return True

    # --- Optimization ---
    
    # Bounds for the 2 base hexagons
    bounds = [
        (-4.0, 4.0),  # x1
        (-4.0, 4.0),  # y1
        (0.0, np.pi/3), # a1
        (-4.0, 4.0),  # x2
        (-4.0, 4.0),  # y2
        (0.0, np.pi/3), # a2
    ]

    best_side = float('inf')
    best_params = None

    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 30:
        try:
            de_result = differential_evolution(
                compute_objective_smooth,
                bounds=bounds,
                maxiter=50,
                tol=1e-8,
                popsize=15,
                seed=42,
                polish=False,
                init='sobol'
            )
            if de_result.fun < 50:
                params = de_result.x
                exact_val = compute_min_side_exact(generate_hexes(params))
                if is_valid(params) and exact_val < best_side:
                    best_side = exact_val
                    best_params = params.copy()
        except Exception:
            pass

    # Phase 2: L-BFGS-B polishing with smooth objective from multiple starts
    if time.time() < deadline - 20:
        # Start from DE result if available, plus some heuristic starts
        starts = []
        if best_params is not None:
            starts.append(best_params.copy())
        
        # Heuristic starts based on honeycomb geometry
        r1 = 2.0
        r2 = 2 * math.sqrt(3)
        
        # Ring 1 at radius r1, angle 0; Ring 2 at radius r2, angle 30 deg
        for a1_deg in [0, 15, 30]:
            for a2_deg in [0, 15, 30]:
                x1 = r1
                y1 = 0
                x2 = r2 * math.cos(math.radians(30))
                y2 = r2 * math.sin(math.radians(30))
                starts.append([x1, y1, math.radians(a1_deg), x2, y2, math.radians(a2_deg)])
        
        for start in starts:
            if time.time() > deadline - 10:
                break
            
            try:
                result = minimize(
                    compute_objective_smooth,
                    x0=np.array(start, dtype=float),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500}
                )
                if result.fun < 50:
                    params = result.x
                    exact_val = compute_min_side_exact(generate_hexes(params))
                    if is_valid(params) and exact_val < best_side:
                        best_side = exact_val
                        best_params = params.copy()
            except Exception:
                pass

    # Phase 3: Nelder-Mead final polish on exact objective
    if best_params is not None and time.time() < deadline - 5:
        try:
            result_final = minimize(
                compute_objective_exact,
                x0=best_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result_final.fun < 50:
                params = result_final.x
                exact_val = compute_min_side_exact(generate_hexes(params))
                if is_valid(params) and exact_val < best_side:
                    best_side = exact_val
                    best_params = params.copy()
        except Exception:
            pass

    # If no valid solution found, use a known valid configuration
    if best_params is None:
        # Fallback: standard honeycomb
        r1 = 2.0
        r2 = 2 * math.sqrt(3)
        x1, y1, a1 = r1, 0.0, 0.0
        x2, y2, a2 = r2 * math.cos(math.radians(30)), r2 * math.sin(math.radians(30)), 0.0
        best_params = np.array([x1, y1, a1, x2, y2, a2])
        best_side = compute_min_side_exact(generate_hexes(best_params))

    # Build the output
    x1, y1, a1, x2, y2, a2 = best_params
    a1_deg = math.degrees(a1) % 360
    a2_deg = math.degrees(a2) % 360
    
    inner = []
    for k in range(6):
        angle = math.radians(60 * k)
        c = math.cos(angle)
        s_ = math.sin(angle)
        
        # Rotate base1
        rx1 = x1 * c - y1 * s_
        ry1 = x1 * s_ + y1 * c
        ra1 = math.degrees(a1 + angle) % 360
        inner.append((rx1, ry1, ra1))
        
        # Rotate base2
        rx2 = x2 * c - y2 * s_
        ry2 = x2 * s_ + y2 * c
        ra2 = math.degrees(a2 + angle) % 360
        inner.append((rx2, ry2, ra2))
    
    # Outer hexagon: centered at origin, orientation 0
    oc = (0.0, 0.0)
    oa = 0.0
    os = compute_min_side_exact(generate_hexes(best_params))
    
    return inner, oc, os, oa
