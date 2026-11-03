"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Helper: get vertices of a regular hexagon with side length 1
    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    # Helper: check if two hexagons overlap using SAT (Separating Axis Theorem)
    def hexagons_overlap(v1, v2):
        """Check if two hexagons overlap using Separating Axis Theorem"""
        axes = []
        for i in range(3):
            # Edge normals for hexagon 1
            edge = v1[i+1] - v1[i]
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            axes.append(normal)
            # Edge normals for hexagon 2
            edge = v2[i+1] - v2[i]
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            axes.append(normal)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 or max2 < min1:
                return False
        return True

    # Helper: compute minimum outer side length needed to contain all inner hexagons
    def compute_min_side(inner_hexes, outer_center, outer_angle_deg):
        """Compute minimum outer hexagon side length to contain all inner hexagons"""
        outer_center = np.array(outer_center)
        outer_angle_rad = math.radians(outer_angle_deg)
        
        # Support function: max projection onto each of the 6 outer hexagon normals
        max_proj = 0.0
        for i in range(6):
            # Normal direction for outer hexagon face i
            normal_angle = outer_angle_rad + i * math.pi / 3
            normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
            
            # For each inner hexagon, find max projection of its vertices
            for cx, cy, angle_deg in inner_hexes:
                verts = hex_vertices(cx, cy, angle_deg)
                # Project vertices relative to outer center
                rel_verts = verts - outer_center
                proj = np.dot(rel_verts, normal)
                max_proj = max(max_proj, np.max(proj))
        
        # The outer hexagon with side length S has apothem (inradius) = S * sqrt(3)/2
        # So S = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / math.sqrt(3)

    # Helper: check if all hexagons are inside the outer hexagon
    def is_inside_outer(inner_hexes, outer_center, outer_side, outer_angle_deg):
        outer_center = np.array(outer_center)
        outer_angle_rad = math.radians(outer_angle_deg)
        apothem = outer_side * math.sqrt(3) / 2.0
        
        for i in range(6):
            normal_angle = outer_angle_rad + i * math.pi / 3
            normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
            
            for cx, cy, angle_deg in inner_hexes:
                verts = hex_vertices(cx, cy, angle_deg)
                rel_verts = verts - outer_center
                proj = np.dot(rel_verts, normal)
                if np.max(proj) > apothem + 1e-6:
                    return False
        return True

    # Objective function for optimization
    def objective(params):
        """Optimize positions and rotations to minimize outer side length"""
        # params: 36 values = 12 * (x, y, angle_deg)
        inner_hexes = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            angle = params[i*3+2]
            inner_hexes.append((x, y, angle))
        
        # Check for overlaps - add penalty if any overlap
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                v1 = hex_vertices(*inner_hexes[i])
                v2 = hex_vertices(*inner_hexes[j])
                if hexagons_overlap(v1, v2):
                    overlap_penalty += 100.0
        
        # Compute required outer side length
        side = compute_min_side(inner_hexes, (0.0, 0.0), 0.0)
        
        return side + overlap_penalty

    # Initial guess: Valid 3x4 grid of hexagons
    # Hexagon side 1. 
    # For angle 0, width is 2, height is sqrt(3).
    # To be disjoint:
    # x-spacing >= 2.0 (vertex to vertex)
    # y-spacing >= sqrt(3) (flat to flat)
    # Using 2.0 and sqrt(3) ensures they touch but do not overlap.
    
    grid_xs = [-2.0, 0.0, 2.0]
    grid_ys = [0.0, math.sqrt(3), 2*math.sqrt(3), 3*math.sqrt(3)]
    
    positions = []
    for y in grid_ys:
        for x in grid_xs:
            positions.append((x, y, 0.0))
            
    # Center the arrangement at (0,0)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    # Parameters: 12*(x,y,angle) + 2*(outer_cx, outer_cy) + 1*(outer_angle)
    # Total 38 parameters
    initial_params = []
    for px, py, angle in positions:
        initial_params.extend([px - cx, py - cy, angle])
    initial_params.extend([0.0, 0.0, 0.0]) # Outer center and angle
    
    # Objective function with outer center/angle optimization
    def objective_full(params):
        inner_hexes = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            angle = params[i*3+2]
            inner_hexes.append((x, y, angle))
        
        outer_cx = params[N*3]
        outer_cy = params[N*3+1]
        outer_angle = params[N*3+2]
        
        # Check overlaps
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                v1 = hex_vertices(*inner_hexes[i])
                v2 = hex_vertices(*inner_hexes[j])
                if hexagons_overlap(v1, v2):
                    overlap_penalty += 1000.0 # High penalty
        
        # Compute side
        side = compute_min_side(inner_hexes, (outer_cx, outer_cy), outer_angle)
        
        return side + overlap_penalty

    best_result = None
    best_side = float('inf')
    
    # Use differential evolution for global search
    try:
        bounds = []
        for i in range(N):
            bounds.append((-10, 10))  # x
            bounds.append((-10, 10))  # y
            bounds.append((0, 360))   # angle
        bounds.append((-5, 5))  # outer_cx
        bounds.append((-5, 5))  # outer_cy
        bounds.append((0, 60))  # outer_angle
        
        result = minimize(
            objective_full,
            initial_params,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'maxfev': 10000,
                'xatol': 1e-6,
                'fatol': 1e-6,
                'adaptive': True
            }
        )
        
        if result.fun < best_side:
            best_result = result.x
            best_side = result.fun
    except Exception as e:
        pass
    
    # Fallback to initial valid grid if optimization fails or returns invalid (overlap)
    if best_result is None:
        best_result = np.array(initial_params)
        
    # Verify final result for overlaps. If overlaps exist, we must return a valid config.
    # We check the best_result. If it has overlaps, we revert to the initial grid (which is valid).
    final_inner_hexes = []
    for i in range(N):
        x = best_result[i*3]
        y = best_result[i*3+1]
        angle = best_result[i*3+2]
        final_inner_hexes.append((x, y, angle))
        
    has_overlap = False
    for i in range(N):
        for j in range(i+1, N):
            v1 = hex_vertices(*final_inner_hexes[i])
            v2 = hex_vertices(*final_hexes[j]) if False else hex_vertices(*final_inner_hexes[j])
            if hexagons_overlap(v1, v2):
                has_overlap = True
                break
        if has_overlap:
            break
            
    if has_overlap:
        # Revert to initial valid grid
        final_inner_hexes = []
        for px, py, angle in positions:
            final_inner_hexes.append((px - cx, py - cy, angle))
        outer_cx, outer_cy, outer_angle = 0.0, 0.0, 0.0
    else:
        outer_cx = best_result[N*3]
        outer_cy = best_result[N*3+1]
        outer_angle = best_result[N*3+2]
        
    inner = []
    for x, y, angle in final_inner_hexes:
        inner.append([float(x), float(y), float(angle)])
        
    final_side = compute_min_side(final_inner_hexes, (outer_cx, outer_cy), outer_angle)
    
    return inner, (float(outer_cx), float(outer_cy)), float(final_side), float(outer_angle)
# EVOLVE-BLOCK-END
