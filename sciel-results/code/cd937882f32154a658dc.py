import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, s, angle_deg):
    """Get vertices of a regular hexagon with center (cx,cy), side length s, rotation angle_deg."""
    theta = np.radians(angle_deg)
    angles = theta + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def max_violation_point(px, py, cx, cy, s, outer_angle_deg):
    """Return max violation (positive if outside) of point (px,py) w.r.t. hexagon."""
    dx = px - cx
    dy = py - cy
    theta = np.radians(outer_angle_deg)
    # Outward normals at angles theta + pi/6 + k*pi/3
    normal_angles = theta + np.pi/6 + np.arange(6) * np.pi / 3
    cos_n = np.cos(normal_angles)
    sin_n = np.sin(normal_angles)
    projections = dx * cos_n + dy * sin_n
    inradius = s * np.sqrt(3) / 2
    violations = projections - inradius
    return np.max(violations)

def objective(params):
    """Minimize s + penalty for constraint violations."""
    s = params[0]
    cx = params[1]
    cy = params[2]
    outer_angle = params[3]
    
    max_viol = 0.0
    for i in range(N):
        ix = params[4 + i*3]
        iy = params[5 + i*3]
        ia = params[6 + i*3]
        
        # Get vertices of inner hexagon
        theta = np.radians(ia)
        angles = theta + np.arange(6) * np.pi / 3
        vx = ix + np.cos(angles)
        vy = iy + np.sin(angles)
        
        for j in range(6):
            v = max_violation_point(vx[j], vy[j], cx, cy, s, outer_angle)
            if v > max_viol:
                max_viol = v
    
    penalty = 1000.0 * max_viol ** 2 if max_viol > 0 else 0.0
    return s + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Generate initial honeycomb positions
    def initial_positions():
        positions = []
        r = np.sqrt(3)  # center-to-center for edge-sharing hexagons
        
        # Center
        positions.append((0.0, 0.0, 0.0))
        
        # Ring 1: 6 hexes
        for k in range(6):
            angle = k * np.pi / 3
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            positions.append((x, y, 0.0))
        
        # Ring 2: 5 hexes
        ring2_angles = []
        for k in range(6):
            base_angle = k * np.pi / 3
            a1 = base_angle - np.pi/6
            a2 = base_angle + np.pi/6
            ring2_angles.extend([a1, a2])
        
        ring2_angles.sort()
        selected = [ring2_angles[i] for i in range(0, 12, 2)][:5]
        
        for angle in selected:
            x = 2 * r * np.cos(angle)
            y = 2 * r * np.sin(angle)
            positions.append((x, y, 0.0))
        
        return positions
    
    positions = initial_positions()
    
    # Try multiple outer angles
    best_params = None
    best_score = float('inf')
    
    for outer_angle_try in [0, 30, 60, 15, 45, 75, 90]:
        p = [6.0, 0.0, 0.0, float(outer_angle_try)]
        for (x, y, a) in positions:
            p.extend([x, y, a])
        
        try:
            result = minimize(objective, p, method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            score = result.fun
            if score < best_score:
                best_score = score
                best_params = result.x.copy()
        except:
            pass
        
        if time.time() > deadline - 30:
            break
    
    if best_params is None:
        inner = [[p[0], p[1], p[2]] for p in positions]
        return inner, (0.0, 0.0), 8.0, 0.0
    
    s = best_params[0]
    cx = best_params[1]
    cy = best_params[2]
    outer_angle = best_params[3]
    
    inner = []
    for i in range(N):
        x = best_params[4 + i*3]
        y = best_params[5 + i*3]
        a = best_params[6 + i*3]
        inner.append([x, y, a])
    
    # Polish
    if time.time() < deadline - 20:
        try:
            result2 = minimize(objective, best_params, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if result2.fun < best_score:
                best_params = result2.x.copy()
                s = best_params[0]
                cx = best_params[1]
                cy = best_params[2]
                outer_angle = best_params[3]
                inner = []
                for i in range(N):
                    x = best_params[4 + i*3]
                    y = best_params[5 + i*3]
                    a = best_params[6 + i*3]
                    inner.append([x, y, a])
        except:
            pass
    
    # Validate and fix
    max_viol = 0.0
    for i in range(N):
        theta = np.radians(inner[i][2])
        angles = theta + np.arange(6) * np.pi / 3
        vx = inner[i][0] + np.cos(angles)
        vy = inner[i][1] + np.sin(angles)
        for j in range(6):
            v = max_violation_point(vx[j], vy[j], cx, cy, s, outer_angle)
            if v > max_viol:
                max_viol = v
    
    if max_viol > 1e-6:
        s += max_viol / (np.sqrt(3) / 2) + 0.001
    
    return inner, (cx, cy), s, outer_angle
