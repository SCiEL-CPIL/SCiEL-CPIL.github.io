"""
Strategy: Symmetry-Constrained Nelder-Mead.
Exploits 6-fold rotational symmetry to reduce 36 DOF to 6 DOF.
We model the 12 inner hexagons as:
- 1 hexagon at the center (0,0)
- 6 hexagons in a ring of radius r1, rotated by a1
- 5 hexagons in a ring of radius r2, rotated by a2 (offset by 30 deg)
The outer hexagon is centered at (0,0) with rotation o_a.
We minimize the outer side length using Nelder-Mead.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

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

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        apothem = side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def polygons_disjoint_sat(v1, v2):
        """
        Check if two convex polygons are disjoint using Separating Axis Theorem.
        v1, v2: Nx2 arrays of vertices.
        Returns True if disjoint, False if overlapping.
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
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        return False

    def all_inside_outer(inner_positions, outer_center, outer_side, outer_angle):
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    return False
        return True

    def pairwise_disjoint(inner_positions):
        for i in range(len(inner_positions)):
            for j in range(i+1, len(inner_positions)):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                if not polygons_disjoint_sat(v1, v2):
                    return False
        return True

    def is_valid(inner, oc, os, oa):
        return all_inside_outer(inner, oc, os, oa) and pairwise_disjoint(inner)

    def compute_minimal_outer_side(inner_positions, outer_center, outer_angle):
        """
        Given fixed inner positions and outer center/angle, find the minimal
        outer side length such that all inner hexagons fit inside.
        """
        apothem_factor = np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        angle_rad = np.radians(outer_angle)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        
        max_required = 0.0
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                dx = vx - outer_center[0]
                dy = vy - outer_center[1]
                local_x = dx * cos_a - dy * sin_a
                local_y = dx * sin_a + dy * cos_a
                dists = local_x * normals[:, 0] + local_y * normals[:, 1]
                max_required = max(max_required, np.max(dists))
        
        return max_required / apothem_factor

    def objective(params):
        """
        params: [r1, a1, r2, a2, o_a]
        r1: radius of inner ring (6 hexagons)
        a1: angle of inner ring hexagons
        r2: radius of outer ring (5 hexagons)
        a2: angle of outer ring hexagons
        o_a: rotation of outer hexagon
        """
        if time.time() > deadline - 2:
            return 1000.0

        r1, a1, r2, a2, o_a = params
        
        # Generate inner positions
        inner_positions = []
        
        # Center hexagon
        inner_positions.append((0.0, 0.0, 0.0))
        
        # Inner ring: 6 hexagons
        for k in range(6):
            angle = np.radians(60 * k)
            x = r1 * np.cos(angle)
            y = r1 * np.sin(angle)
            inner_positions.append((x, y, a1))
            
        # Outer ring: 5 hexagons (offset by 30 degrees to fit in gaps)
        for k in range(5):
            angle = np.radians(60 * k + 30)
            x = r2 * np.cos(angle)
            y = r2 * np.sin(angle)
            inner_positions.append((x, y, a2))
        
        # Check disjointness
        if not pairwise_disjoint(inner_positions):
            # Penalty for overlapping
            penalty = 0.0
            for i in range(len(inner_positions)):
                for j in range(i+1, len(inner_positions)):
                    ci = (inner_positions[i][0], inner_positions[i][1])
                    cj = (inner_positions[j][0], inner_positions[j][1])
                    ai = inner_positions[i][2]
                    aj = inner_positions[j][2]
                    v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                    v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                    if not polygons_disjoint_sat(v1, v2):
                        penalty += 1.0
            return 100.0 + penalty * 10.0
        
        # Compute minimal outer side
        os = compute_minimal_outer_side(inner_positions, (0.0, 0.0), o_a)
        return os

    # Initial guess
    # Standard hexagonal packing:
    # Inner ring at distance 2.0 (touching center)
    # Outer ring at distance ~3.46 (touching inner ring)
    # Angles: 0 for inner, 30 for outer (to fit in gaps)
    # Outer rotation: 30 (flat top) or 0 (pointy top)
    
    initial_guess = np.array([2.0, 0.0, 3.5, 30.0, 30.0])
    
    # Bounds for Nelder-Mead (not strictly enforced by NM, but we can clip)
    # r1, r2 > 0
    # Angles 0-360
    
    result = None
    try:
        result = minimize(
            objective, 
            initial_guess, 
            method='Nelder-Mead', 
            options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6}
        )
    except Exception as e:
        pass

    if result is None or not result.success:
        # Fallback to a known valid configuration if optimization fails
        # Use the initial guess if it's valid
        best_params = initial_guess
    else:
        best_params = result.x

    # Ensure angles are in [0, 360)
    best_params[1] = np.mod(best_params[1], 360)
    best_params[3] = np.mod(best_params[3], 360)
    best_params[4] = np.mod(best_params[4], 360)
    
    # Ensure radii are positive
    if best_params[0] < 0: best_params[0] = 0.1
    if best_params[2] < 0: best_params[2] = 0.1

    r1, a1, r2, a2, o_a = best_params
    
    # Construct final inner positions
    inner = []
    inner.append((0.0, 0.0, 0.0))
    for k in range(6):
        angle = np.radians(60 * k)
        x = r1 * np.cos(angle)
        y = r1 * np.sin(angle)
        inner.append((x, y, a1))
        
    for k in range(5):
        angle = np.radians(60 * k + 30)
        x = r2 * np.cos(angle)
        y = r2 * np.sin(angle)
        inner.append((x, y, a2))
        
    oc = (0.0, 0.0)
    oa = o_a
    os = compute_minimal_outer_side(inner, oc, oa)
    
    # Final validation
    if not is_valid(inner, oc, os, oa):
        # Fallback: use a known valid configuration
        # Grid-like packing
        config_fallback = []
        # Center
        config_fallback.append([0.0, 0.0, 0.0])
        # Ring 1
        for k in range(6):
            angle = np.radians(60 * k)
            config_fallback.append([2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0])
        # Ring 2 (5 hexagons)
        for k in range(5):
            angle = np.radians(60 * k + 30)
            config_fallback.append([3.5 * np.cos(angle), 3.5 * np.sin(angle), 30.0])
            
        inner = [(p[0], p[1], p[2]) for p in config_fallback]
        oc = (0.0, 0.0)
        oa = 30.0
        os = compute_minimal_outer_side(inner, oc, oa)
        # Expand if needed
        while not all_inside_outer(inner, oc, os, oa):
            os += 0.1
            if os > 10:
                break

    return inner, oc, os, oa
