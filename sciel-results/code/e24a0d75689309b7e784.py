import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Normals for a hexagon with angle 0
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        # Optimization: If all inner hexagons have the same angle,
        # we can pre-compute vertices relative to center.
        # But here we handle general case.
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    return 2.0 * max_projection / SQRT3

def compute_overlap_penalty(inner, min_dist=SQRT3):
    """
    Penalty for overlapping hexagons.
    Assumes all hexagons are aligned (angle 0 relative to each other).
    """
    penalty = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i+1, n):
            xj, yj, _ = inner[j]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < min_dist:
                penalty += (min_dist - dist)**2
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # 1. Define initial inner hexagons
    # Hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial inner: (x, y, angle)
    # Assume all aligned at 0
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Helper to update best
    def update_best(inner, oc, oa, side):
        nonlocal best_side, best_inner, best_outer_center, best_outer_angle
        if side < best_side:
            best_side = side
            best_inner = [list(i) for i in inner]
            best_outer_center = oc
            best_outer_angle = oa

    # 2. Optimize Outer Frame for fixed inner
    def objective_outer(params):
        cx, cy, angle = params
        side = compute_required_side(initial_inner, (cx, cy), angle)
        return side

    # Initial guess for outer center: centroid
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # Try a few initial angles
    for init_angle in [0, 30, 60, 90]:
        if time.time() > deadline - 5:
            break
        x0 = [cx_avg, cy_avg, init_angle]
        res = minimize(objective_outer, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.success or res.fun < best_side:
            cx, cy, angle = res.x
            side = res.fun
            update_best(initial_inner, (cx, cy), angle, side)

    # 3. Optimize Inner Positions (with overlap penalty)
    # We will optimize the x, y positions of the inner hexagons.
    # Angle is fixed at 0 for all.
    
    def objective_inner(params):
        # params: [x1, y1, x2, y2, ..., xN, yN]
        inner_opt = []
        for i in range(N):
            x = params[2*i]
            y = params[2*i+1]
            inner_opt.append([x, y, 0.0])
        
        # Overlap penalty
        pen = compute_overlap_penalty(inner_opt)
        
        # If penalty is high, return large value
        if pen > 1e-3:
            return 1e6 + pen
        
        # Compute required side using best outer frame from previous step?
        # Or re-optimize outer frame?
        # Re-optimizing outer frame is expensive.
        # Let's use the current best outer frame.
        side = compute_required_side(inner_opt, best_outer_center, best_outer_angle)
        
        return side + 1000.0 * pen

    # Initial params
    x0_inner = []
    for x, y, _ in initial_inner:
        x0_inner.append(x)
        x0_inner.append(y)
        
    # Optimize inner positions
    if time.time() < deadline - 10:
        res_inner = minimize(objective_inner, x0_inner, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res_inner.fun < 1e5: # Valid solution
            params = res_inner.x
            inner_opt = []
            for i in range(N):
                x = params[2*i]
                y = params[2*i+1]
                inner_opt.append([x, y, 0.0])
            
            # Re-optimize outer frame for this new inner layout
            def objective_outer2(params):
                cx, cy, angle = params
                side = compute_required_side(inner_opt, (cx, cy), angle)
                return side
            
            # Use the previous best outer frame as initial guess
            x0_outer = [best_outer_center[0], best_outer_center[1], best_outer_angle]
            res_outer2 = minimize(objective_outer2, x0_outer, method='Nelder-Mead',
                                  options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res_outer2.fun < best_side:
                cx, cy, angle = res_outer2.x
                side = res_outer2.fun
                update_best(inner_opt, (cx, cy), angle, side)

    # Final polish
    if time.time() < deadline - 2:
        # One more round of inner optimization if time permits
        x0_inner = []
        for x, y, _ in best_inner:
            x0_inner.append(x)
            x0_inner.append(y)
            
        def objective_inner_final(params):
            inner_opt = []
            for i in range(N):
                x = params[2*i]
                y = params[2*i+1]
                inner_opt.append([x, y, 0.0])
            pen = compute_overlap_penalty(inner_opt)
            if pen > 1e-3:
                return 1e6 + pen
            side = compute_required_side(inner_opt, best_outer_center, best_outer_angle)
            return side + 1000.0 * pen

        res_inner_final = minimize(objective_inner_final, x0_inner, method='Nelder-Mead',
                                   options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if res_inner_final.fun < 1e5:
            params = res_inner_final.x
            inner_opt = []
            for i in range(N):
                x = params[2*i]
                y = params[2*i+1]
                inner_opt.append([x, y, 0.0])
            
            def objective_outer_final(params):
                cx, cy, angle = params
                side = compute_required_side(inner_opt, (cx, cy), angle)
                return side

            x0_outer = [best_outer_center[0], best_outer_center[1], best_outer_angle]
            res_outer_final = minimize(objective_outer_final, x0_outer, method='Nelder-Mead',
                                       options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res_outer_final.fun < best_side:
                cx, cy, angle = res_outer_final.x
                side = res_outer_final.fun
                update_best(inner_opt, (cx, cy), angle, side)

    best_side += 1e-9
    return best_inner, best_outer_center, best_side, best_outer_angle
