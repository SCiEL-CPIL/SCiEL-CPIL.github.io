"""
Strategy: Discrete Topology Search with Continuous Refinement.
Instead of assuming a fixed 1-6-5 hexagonal symmetry, we explicitly search
over different topological arrangements of the 12 hexagons (e.g., 1-6-5, 0-6-6, 2-4-6, etc.).
For each topology, we use a fast gradient-free optimization (Nelder-Mead) to refine
the positions and rotations, minimizing the outer hexagon side length while enforcing
disjointness via a hard constraint penalty.

This approach decouples the combinatorial structure search from the continuous
geometric optimization, allowing us to find non-symmetric or less obvious packings.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_normals(angle_deg):
        base_angle = np.radians(angle_deg + 30)
        angles = base_angle + np.radians(np.arange(0, 360, 60))
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def compute_minimal_outer_side(inner_positions, outer_center, outer_angle):
        normals = get_normals(outer_angle)
        max_proj = 0.0
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            verts_rel = verts - np.array(outer_center)
            projs = verts_rel @ normals.T
            max_proj = max(max_proj, np.max(projs))
        apothem = max_proj
        side = apothem / (math.sqrt(3) / 2)
        return side

    def polygons_disjoint_sat(v1, v2):
        """
        Check if two convex polygons are disjoint using SAT.
        Returns True if disjoint, False if overlapping.
        Also returns the minimum gap (positive if disjoint, negative if overlapping).
        """
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        max_gap = -float('inf')
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap1 = min2 - max1
            gap2 = min1 - max2
            gap = max(gap1, gap2)
            
            if gap > max_gap:
                max_gap = gap
                
        return max_gap

    def objective(params):
        if time.time() > deadline - 2:
            return 1000.0

        # Unpack parameters
        # params: [x1, y1, a1, ..., x12, y12, a12, outer_x, outer_y, outer_angle]
        # Total 37 params
        
        inner_positions = []
        for i in range(12):
            x = params[i*3]
            y = params[i*3 + 1]
            a = params[i*3 + 2]
            inner_positions.append((x, y, a))
        
        outer_x = params[36]
        outer_y = params[37]
        outer_angle = params[38]
        
        # Compute outer side
        os = compute_minimal_outer_side(inner_positions, (outer_x, outer_y), outer_angle)
        
        # Compute collision penalty
        penalty = 0.0
        verts_list = []
        for i in range(12):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            verts_list.append(verts)
            
        for i in range(12):
            for j in range(i+1, 12):
                ci = inner_positions[i]
                cj = inner_positions[j]
                dist_sq = (ci[0]-cj[0])**2 + (ci[1]-cj[1])**2
                if dist_sq > 12.0: # 3.46^2
                    continue
                    
                gap = polygons_disjoint_sat(verts_list[i], verts_list[j])
                # gap > 0 means disjoint.
                # gap < 0 means overlapping.
                # We want to penalize overlap.
                # Use softplus for smoothness.
                k_smooth = 20.0
                arg = -gap * k_smooth
                if arg > 50:
                    penalty += arg
                else:
                    penalty += math.log(1 + math.exp(arg))
                    
        return os + penalty * 10.0

    # Define topologies
    # Each topology is a list of (radius, angle_offset, count) for rings, plus a center flag
    topologies = [
        # 1 center, 6 ring1, 5 ring2
        {"center": True, "rings": [(2.0, 0.0, 6), (3.5, 30.0, 5)]},
        # 0 center, 6 ring1, 6 ring2
        {"center": False, "rings": [(2.0, 0.0, 6), (3.5, 0.0, 6)]},
        # 2 center, 4 ring1, 6 ring2
        {"center": True, "rings": [(2.0, 0.0, 4), (3.5, 30.0, 6)]}, # 2 centers? No, 1 center + 1 extra?
        # Let's stick to simple rings for now.
        # 1 center, 5 ring1, 6 ring2
        {"center": True, "rings": [(2.0, 0.0, 5), (3.5, 30.0, 6)]},
    ]

    best_result = None
    best_score = float('inf')

    for topo in topologies:
        if time.time() > deadline - 10:
            break

        # Generate initial guess for this topology
        init_params = np.zeros(39)
        
        idx = 0
        if topo["center"]:
            init_params[0] = 0.0
            init_params[1] = 0.0
            init_params[2] = 0.0
            idx = 1
        
        ring_idx = 0
        for ring in topo["rings"]:
            r, offset, count = ring
            for k in range(count):
                if idx >= 12: break
                angle = np.radians(60 * k + offset)
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                init_params[idx*3] = x
                init_params[idx*3 + 1] = y
                init_params[idx*3 + 2] = offset # Initial rotation same as offset
                idx += 1
            ring_idx += 1
        
        # Fill remaining with random positions
        while idx < 12:
            init_params[idx*3] = np.random.uniform(-2, 2)
            init_params[idx*3 + 1] = np.random.uniform(-2, 2)
            init_params[idx*3 + 2] = np.random.uniform(0, 60)
            idx += 1
        
        init_params[36] = 0.0
        init_params[37] = 0.0
        init_params[38] = 30.0
        
        bounds = [(-10, 10), (-10, 10), (0, 60)] * 12 + [(-10, 10), (-10, 10), (0, 360)]
        
        try:
            res = minimize(
                objective, 
                init_params, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to a simple valid configuration
        inner = []
        inner.append((0.0, 0.0, 0.0))
        for k in range(6):
            angle = np.radians(60 * k)
            x = 2.05 * np.cos(angle)
            y = 2.05 * np.sin(angle)
            inner.append((x, y, 0.0))
        for k in range(5):
            angle = np.radians(60 * k + 30)
            x = 3.6 * np.cos(angle)
            y = 3.6 * np.sin(angle)
            inner.append((x, y, 30.0))
        oc = (0.0, 0.0)
        oa = 30.0
        os = compute_minimal_outer_side(inner, oc, oa)
        return inner, oc, os, oa

    # Extract best result
    inner = []
    for i in range(12):
        x = best_result[i*3]
        y = best_result[i*3 + 1]
        a = best_result[i*3 + 2]
        inner.append((x, y, a))
    
    oc = (best_result[36], best_result[37])
    oa = best_result[38]
    os = compute_minimal_outer_side(inner, oc, oa)
    
    return inner, oc, os, oa
