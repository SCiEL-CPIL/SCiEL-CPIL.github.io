import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angles = np.radians(angle_deg) + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def violation_penalty(inner, outer_center, outer_side, outer_angle):
    """Compute squared violation of inner hexagons against outer hexagon boundary."""
    ox, oy = outer_center
    theta = np.radians(outer_angle)
    inradius = outer_side * np.sqrt(3) / 2
    total_viol = 0.0
    for ix, iy, iangle in inner:
        verts = hexagon_vertices(ix, iy, 1.0, iangle)
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            lx = dx * np.cos(theta) + dy * np.sin(theta)
            ly = -dx * np.sin(theta) + dy * np.cos(theta)
            for k in range(6):
                alpha = np.radians(30 + k * 60)
                proj = lx * np.cos(alpha) + ly * np.sin(alpha)
                if proj > inradius:
                    total_viol += (proj - inradius) ** 2
    return total_viol

def objective(params):
    """Objective: outer side length + penalty for violations."""
    ox, oy = params[0], params[1]
    outer_side = params[2]
    outer_angle = params[3]
    
    if outer_side < 1.0:
        return 1000.0
    
    inner = []
    for i in range(N):
        ix = params[4 + i*3]
        iy = params[5 + i*3]
        ia = params[6 + i*3]
        inner.append((ix, iy, ia))
    
    viol = violation_penalty(inner, (ox, oy), outer_side, outer_angle)
    return outer_side + 5.0 * np.sqrt(viol)

def construct():
    global START_TIME
    
    # Initial placement: hexagonal lattice cluster with all same rotation
    s = np.sqrt(3)  # center-to-center distance for touching hexagons
    
    # Axial coordinates for 12 hexagons
    # Use a compact hexagonal cluster
    positions_axial = [
        (0, 0),      # center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # ring 1
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2)  # part of ring 2
    ]
    
    inner = []
    for q, r in positions_axial:
        x = s * (q + r/2.0)
        y = s * np.sqrt(3)/2 * r
        inner.append([x, y, 0.0])
    
    # Compute initial bounding hexagon
    cx = np.mean([p[0] for p in inner])
    cy = np.mean([p[1] for p in inner])
    
    # Estimate side length
    max_dist = 0
    for ix, iy, ia in inner:
        verts = hexagon_vertices(ix, iy, 1.0, ia)
        for vx, vy in verts:
            d = np.sqrt((vx-cx)**2 + (vy-cy)**2)
            max_dist = max(max_dist, d)
    
    outer_side_init = max_dist * 2 / np.sqrt(3) + 0.3
    outer_angle_init = 0.0
    
    # Build initial params
    params = [cx, cy, outer_side_init, outer_angle_init]
    for ix, iy, ia in inner:
        params.extend([ix, iy, ia])
    
    best_params = params.copy()
    best_obj = objective(params)
    
    elapsed = time.time() - START_TIME
    
    # Differential evolution
    if elapsed < TIME_LIMIT - 40:
        bounds = [(-15, 15), (-15, 15), (1.0, 25), (0, 360)]
        for i in range(N):
            bounds.extend([(-15, 15), (-15, 15), (0, 360)])
        
        try:
            result = differential_evolution(
                objective, bounds,
                maxiter=30, popsize=12,
                tol=1e-3, seed=42,
                polish=False,
                init='latinhypercube'
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Local refinement
    elapsed = time.time() - START_TIME
    if elapsed < TIME_LIMIT - 15:
        try:
            result2 = minimize(
                objective, best_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            if result2.fun < best_obj:
                best_obj = result2.fun
                best_params = result2.x
        except Exception:
            pass
    
    # Extract best result
    ox, oy = best_params[0], best_params[1]
    outer_side = best_params[2]
    outer_angle = best_params[3]
    
    inner_final = []
    for i in range(N):
        ix = best_params[4 + i*3]
        iy = best_params[5 + i*3]
        ia = best_params[6 + i*3]
        inner_final.append([float(ix), float(iy), float(ia)])
    
    # Final check and adjustment
    viol = violation_penalty(inner_final, (ox, oy), outer_side, outer_angle)
    if viol > 1e-8:
        outer_side += np.sqrt(viol) * 0.3 + 0.01
    
    return inner_final, (float(ox), float(oy)), float(outer_side), float(outer_angle)
