# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import differential_evolution, minimize

def _hex_support(cx, cy, alpha_deg, nx, ny):
    """
    Calculate the support function of a unit hexagon centered at (cx, cy)
    with orientation alpha_deg in the direction (nx, ny).
    """
    # Direction of the normal in the hexagon's local frame
    # Global angle of the normal
    theta_global = np.arctan2(ny, nx)
    # Local angle relative to the hexagon's orientation
    # Hexagon vertices are at alpha + 0, 60, ...
    # Normals to sides are at alpha + 30, 90, ...
    # We need the angle relative to the nearest normal.
    # The support function h(phi) = (sqrt(3)/2) * sec(phi) for phi in [-30, 30]
    
    alpha_rad = np.deg2rad(alpha_deg)
    # Angle of the normal relative to the hexagon's local x-axis
    # The hexagon's local x-axis is at alpha_rad.
    # The normals are at alpha_rad + 30 + 60k.
    # We want the angle between (nx, ny) and the nearest normal.
    
    # Compute angle difference
    diff = theta_global - alpha_rad
    # Normalize to [-30, 30] relative to the nearest normal (which is at 30, 90, etc)
    # Actually, the normals are at 30, 90, 150...
    # So we subtract 30 degrees first?
    # Let's define local angle phi = theta_global - alpha_rad.
    # The normals are at 30, 90, ... in local coords.
    # So we map phi to [0, 60) and find distance to 30.
    
    phi = np.mod(diff, np.deg2rad(60))
    # Distance from the nearest normal (which is at 30 deg in local coords)
    # If phi is in [0, 60), the nearest normal is at 30.
    # The angle from the normal is phi - 30.
    # But we need the absolute angle from the normal.
    # If phi < 30, angle is 30 - phi.
    # If phi > 30, angle is phi - 30.
    # Wait, the normal is at 30.
    # So the deviation is abs(phi - 30).
    # But if phi is close to 0 or 60, it's close to the *next* normal?
    # No, the period is 60.
    # The normals are at 30, 90, 150.
    # In the interval [0, 60), the normal is at 30.
    # So deviation is abs(phi - 30).
    # However, if phi is 59, it's close to 60 (which is 0 of next cycle, normal at 30+60=90).
    # So we should map phi to [-30, 30] relative to the normal.
    
    # Simpler:
    # Rotate global normal to local frame.
    # local_angle = theta_global - alpha_rad
    # The normals are at 30, 90, ...
    # Subtract 30 to align first normal to 0.
    # local_angle_adj = local_angle - 30 deg
    # Map to [-30, 30]
    
    local_angle_adj = diff - np.deg2rad(30)
    # Map to [-30, 30]
    local_angle_adj = (local_angle_adj + np.deg2rad(30)) % np.deg2rad(60) - np.deg2rad(30)
    
    # Support distance from center
    h = (np.sqrt(3) / 2) / np.cos(local_angle_adj)
    
    # Project center
    center_proj = cx * nx + cy * ny
    
    return center_proj + h

def _objective(params):
    """
    params: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10, beta]
    """
    # Extract parameters
    centers = params[:33].reshape(11, 3)
    beta = params[33]
    
    # Check collisions
    # Pre-compute vertices? No, use gap function.
    # To be fast, we can check distance between centers first.
    # If dist > 4, they are definitely separated (max width 2).
    # If dist < 2, they overlap.
    # If 2 <= dist <= 4, need precise check.
    
    min_gap = np.inf
    for i in range(11):
        for j in range(i + 1, 11):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1.9: # Very close, likely overlap
                return 1e10
            if dist > 4.1: # Far apart
                continue
            
            # Precise check
            # Compute vertices
            a_i = np.deg2rad(centers[i, 2])
            a_j = np.deg2rad(centers[j, 2])
            
            verts_i = np.column_stack([
                centers[i, 0] + np.cos(a_i + np.deg2rad(60 * np.arange(6))),
                centers[i, 1] + np.sin(a_i + np.deg2rad(60 * np.arange(6)))
            ])
            verts_j = np.column_stack([
                centers[j, 0] + np.cos(a_j + np.deg2rad(60 * np.arange(6))),
                centers[j, 1] + np.sin(a_j + np.deg2rad(60 * np.arange(6)))
            ])
            
            # Normals
            n_i = np.column_stack([
                np.cos(a_i + np.deg2rad(30 + 60 * np.arange(6))),
                np.sin(a_i + np.deg2rad(30 + 60 * np.arange(6)))
            ])
            n_j = np.column_stack([
                np.cos(a_j + np.deg2rad(30 + 60 * np.arange(6))),
                np.sin(a_j + np.deg2rad(30 + 60 * np.arange(6)))
            ])
            
            # Project
            proj_i_i = verts_i @ n_i.T
            proj_j_i = verts_j @ n_i.T
            gap_i = np.maximum(
                proj_j_i.min(axis=0) - proj_i_i.max(axis=0),
                proj_i_i.min(axis=0) - proj_j_i.max(axis=0)
            )
            
            proj_i_j = verts_i @ n_j.T
            proj_j_j = verts_j @ n_j.T
            gap_j = np.maximum(
                proj_j_j.min(axis=0) - proj_i_j.max(axis=0),
                proj_i_j.min(axis=0) - proj_j_j.max(axis=0)
            )
            
            g = max(gap_i.max(), gap_j.max())
            if g < min_gap:
                min_gap = g
            if min_gap < 1e-6:
                return 1e6 * (1e-6 - min_gap) + 100.0
    
    if min_gap < 1e-6:
        return 1e6 * (1e-6 - min_gap) + 100.0

    # Calculate outer hexagon side
    # Outer hexagon is centered at (0,0) with angle beta
    # Normals of outer hexagon
    beta_rad = np.deg2rad(beta)
    n_outer = np.column_stack([
        np.cos(beta_rad + np.deg2rad(30 + 60 * np.arange(6))),
        np.sin(beta_rad + np.deg2rad(30 + 60 * np.arange(6)))
    ])
    
    max_r = 0
    for k in range(6):
        nx, ny = n_outer[k]
        for i in range(11):
            cx, cy, ai = centers[i]
            r = _hex_support(cx, cy, ai, nx, ny)
            if r > max_r:
                max_r = r
    
    side = 2.0 * max_r / np.sqrt(3.0)
    return side

def hexagon_packing_11():
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0
    
    # Initial guess: Flower
    init = []
    init.extend([0.0, 0.0, 0.0])
    for k in range(6):
        ang = np.deg2rad(60 * k)
        init.extend([2.0 * np.cos(ang), 2.0 * np.sin(ang), 0.0])
    # 4 outer
    for k in [0, 2, 4, 5]: # Arbitrary selection for symmetry breaking?
        ang = np.deg2rad(60 * k + 30)
        init.extend([2 * np.sqrt(3) * np.cos(ang), 2 * np.sqrt(3) * np.sin(ang), 0.0])
    init.append(0.0) # Outer angle
    
    init_params = np.array(init)
    
    bounds = []
    for _ in range(11):
        bounds.append((-6.0, 6.0))
        bounds.append((-6.0, 6.0))
        bounds.append((0.0, 60.0))
    bounds.append((0.0, 60.0))
    
    best_params = init_params.copy()
    best_side = _objective(best_params)
    
    # Differential Evolution
    if time.time() - start_time < time_limit - safety_margin:
        def de_obj(x):
            if time.time() - start_time > time_limit - safety_margin:
                return 1e10
            return _objective(x)
        
        try:
            result = differential_evolution(
                de_obj,
                bounds,
                x0=init_params,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=False,
                init='sobol',
                workers=-1,
            )
            if result.fun < best_side:
                best_params = result.x
                best_side = result.fun
        except Exception:
            pass
    
    # Local Refinement with Powell
    if time.time() - start_time < time_limit - safety_margin:
        remaining = time_limit - safety_margin - (time.time() - start_time)
        if remaining > 5:
            def obj(x):
                if time.time() - start_time > time_limit - safety_margin:
                    return 1e10
                return _objective(x)
            
            try:
                res = minimize(
                    obj, best_params,
                    method='Powell',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.fun < best_side:
                    best_params = res.x
                    best_side = res.fun
            except Exception:
                pass
    
    # Build Result
    inner_hex_data = np.empty((11, 3))
    for i in range(11):
        inner_hex_data[i] = best_params[3 * i:3 * i + 3]
    
    # Shift to center of mass? No, contract says "shifted by outer center".
    # The outer center is (0,0) in our optimization.
    # So we just return the coordinates.
    
    outer_hex_data = np.array([0.0, 0.0, best_params[33]])
    outer_hex_side_length = float(best_side)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
