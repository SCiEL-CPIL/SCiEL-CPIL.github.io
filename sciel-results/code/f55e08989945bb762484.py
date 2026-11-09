"""Packing 12 unit hexagons inside a regular hexagon.

Strategy:
  1. Use a hexagonal lattice arrangement as the initial placement.
  2. Compute the minimal enclosing regular hexagon side length.
  3. Use a combined approach: 
     - First, try known good arrangements (hexagonal lattice, 4x3 grid).
     - Then, use SLSQP with SAT-based separation constraints to tighten.
     - Finally, use Nelder-Mead for local refinement.
  4. The key insight is that rotating inner hexagons can reduce the required
     outer hexagon size, and SAT-based constraints allow SLSQP to work.
"""

import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0


def _elapsed():
    return time.time() - START_TIME


def get_hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with given center, side, rotation."""
    angle_rad = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts[k, 0] = cx + side * np.cos(a)
        verts[k, 1] = cy + side * np.sin(a)
    return verts


def compute_minimal_enclosing_side(inner_hexes, outer_rotation_deg=0.0):
    """Compute minimal side length of outer hexagon (with given rotation)
    that contains all inner hexagons. Returns (side, center_x, center_y)."""
    all_verts = np.vstack([
        get_hex_vertices(h[0], h[1], 1.0, h[2]) for h in inner_hexes
    ])

    rot_rad = np.radians(outer_rotation_deg)
    normals = np.empty((6, 2))
    for k in range(6):
        a = rot_rad + k * np.pi / 3 + np.pi / 6
        normals[k, 0] = np.cos(a)
        normals[k, 1] = np.sin(a)

    proj_max = np.max(np.dot(all_verts, normals.T), axis=0)
    proj_min = np.min(np.dot(all_verts, normals.T), axis=0)

    def objective(c):
        cx, cy = c
        center_dot = cx * normals[:, 0] + cy * normals[:, 1]
        hw = np.maximum(proj_max - center_dot, center_dot - proj_min)
        return hw.max()

    c0 = np.array([all_verts[:, 0].mean(), all_verts[:, 1].mean()])
    result = minimize(objective, c0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})

    cx_opt, cy_opt = result.x
    center_dot = cx_opt * normals[:, 0] + cy_opt * normals[:, 1]
    hw = np.maximum(proj_max - center_dot, center_dot - proj_min)
    L = hw.max() / (np.sqrt(3) / 2)

    return L, cx_opt, cy_opt


def get_penetration_depth(v1, v2):
    """Calculate the minimum translation distance to separate two convex polygons.
    Returns 0 if separated, positive value if overlapping (penetration depth)."""
    min_sep = np.inf
    for verts in (v1, v2):
        for i in range(6):
            a = verts[i]
            b = verts[(i + 1) % 6]
            edge = b - a
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            gap = min(proj1.min() - proj2.max(), proj2.min() - proj1.max())
            min_sep = min(min_sep, gap)
            
    if min_sep > 0:
        return 0.0
    else:
        return -min_sep


def separation_constraints(params, rot):
    """Create SLSQP constraints for non-overlap using SAT-based separation."""
    hexes = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(N)]
    verts_list = [get_hex_vertices(h[0], h[1], 1.0, h[2]) for h in hexes]
    
    cons = []
    for i in range(N):
        for j in range(i + 1, N):
            # Constraint: penetration depth <= 0 (i.e., separated)
            # We want: -penetration_depth >= 0
            def sep_constraint(p, i=i, j=j):
                hexes_p = [[p[3*k], p[3*k+1], p[3*k+2]] for k in range(N)]
                verts_i = get_hex_vertices(hexes_p[i][0], hexes_p[i][1], 1.0, hexes_p[i][2])
                verts_j = get_hex_vertices(hexes_p[j][0], hexes_p[j][1], 1.0, hexes_p[j][2])
                return -get_penetration_depth(verts_i, verts_j)
            
            cons.append({'type': 'ineq', 'fun': sep_constraint})
    
    return cons


def make_initial_hex_grid():
    """Create a hexagonal lattice arrangement of 12 unit hexagons."""
    dx = 3.0
    dy = np.sqrt(3)
    inner = []
    # 4 columns, 3 rows with offset
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx * 0.5
            y = row * dy
            inner.append([x, y, 0.0])
    return inner


def make_initial_4x3_grid():
    """Create a 4x3 rectangular grid arrangement."""
    dx = 3.0
    dy = np.sqrt(3)
    inner = []
    for row in range(3):
        for col in range(4):
            x = col * dx
            y = row * dy
            inner.append([x, y, 0.0])
    return inner


def construct():
    global START_TIME
    START_TIME = time.time()

    np.random.seed(42)

    # Try multiple initial configurations
    initial_configs = [
        make_initial_hex_grid(),
        make_initial_4x3_grid(),
    ]
    
    # Add some random configurations
    for _ in range(3):
        random_inner = []
        for _ in range(N):
            x = np.random.uniform(-6, 6)
            y = np.random.uniform(-6, 6)
            angle = np.random.uniform(0, 60)
            random_inner.append([x, y, angle])
        initial_configs.append(random_inner)

    best_side = np.inf
    best_inner = [list(h) for h in initial_configs[0]]
    best_center = [0.0, 0.0]
    best_rot = 0.0

    rotations = [0.0, 30.0, 15.0, 45.0]
    
    for config_idx, initial in enumerate(initial_configs):
        if _elapsed() > TIME_LIMIT - 30:
            break
        
        for rot in rotations:
            if _elapsed() > TIME_LIMIT - 20:
                break
            
            params0 = np.array([p for h in initial for p in h])
            
            # Try SLSQP with separation constraints
            try:
                cons = separation_constraints(params0, rot)
                
                def objective_slsqp(p):
                    if _elapsed() > TIME_LIMIT - 5:
                        return 1e6
                    hexes = [[p[3*i], p[3*i+1], p[3*i+2]] for i in range(N)]
                    s, _, _ = compute_minimal_enclosing_side(hexes, rot)
                    return s
                
                result = minimize(objective_slsqp, params0, method='SLSQP',
                                  constraints=cons,
                                  options={'maxiter': 200, 'ftol': 1e-8, 'disp': False})
                
                if result.fun < best_side and result.fun < 1e5:
                    final_hexes = [[result.x[3*i], result.x[3*i+1], result.x[3*i+2]] for i in range(N)]
                    verts_list = [get_hex_vertices(h[0], h[1], 1.0, h[2]) for h in final_hexes]
                    valid = True
                    for i in range(N):
                        for j in range(i + 1, N):
                            if get_penetration_depth(verts_list[i], verts_list[j]) > 1e-4:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        s, cx, cy = compute_minimal_enclosing_side(final_hexes, rot)
                        if s < best_side:
                            best_side = s
                            best_inner = final_hexes
                            best_center = [cx, cy]
                            best_rot = rot
            except Exception as e:
                pass
            
            # Try Nelder-Mead for local refinement
            if _elapsed() > TIME_LIMIT - 15:
                continue
            
            try:
                def objective_nm(p):
                    if _elapsed() > TIME_LIMIT - 5:
                        return 1e6
                    hexes = [[p[3*i], p[3*i+1], p[3*i+2]] for i in range(N)]
                    verts_list = [get_hex_vertices(h[0], h[1], 1.0, h[2]) for h in hexes]
                    penalty = 0.0
                    for i in range(N):
                        for j in range(i + 1, N):
                            depth = get_penetration_depth(verts_list[i], verts_list[j])
                            if depth > 0:
                                penalty += depth * depth
                    
                    s, _, _ = compute_minimal_enclosing_side(hexes, rot)
                    return s + 10.0 * penalty
                
                result = minimize(objective_nm, params0, method='Nelder-Mead',
                                  options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
                
                if result.fun < best_side and result.fun < 1e5:
                    final_hexes = [[result.x[3*i], result.x[3*i+1], result.x[3*i+2]] for i in range(N)]
                    verts_list = [get_hex_vertices(h[0], h[1], 1.0, h[2]) for h in final_hexes]
                    valid = True
                    for i in range(N):
                        for j in range(i + 1, N):
                            if get_penetration_depth(verts_list[i], verts_list[j]) > 1e-4:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        s, cx, cy = compute_minimal_enclosing_side(final_hexes, rot)
                        if s < best_side:
                            best_side = s
                            best_inner = final_hexes
                            best_center = [cx, cy]
                            best_rot = rot
            except Exception as e:
                pass

    # Final precise computation
    final_side, final_cx, final_cy = compute_minimal_enclosing_side(best_inner, best_rot)

    return best_inner, (final_cx, final_cy), final_side, best_rot
