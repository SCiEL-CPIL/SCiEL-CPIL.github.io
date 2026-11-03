import numpy as np
from scipy.optimize import minimize
import time

N = 12
SQRT3 = np.sqrt(3)

def compute_min_side(pos, outer_center, outer_angle_deg):
    """Compute minimum outer hexagon side to contain all unit hexagons at pos (all at angle 0)."""
    cx, cy = outer_center
    oangle_rad = np.radians(outer_angle_deg)
    cos_o = np.cos(-oangle_rad)
    sin_o = np.sin(-oangle_rad)
    
    max_val = 0.0
    for i in range(N):
        ix, iy = pos[i]
        for j in range(6):
            a = j * np.pi / 3
            vx = ix + np.cos(a)
            vy = iy + np.sin(a)
            dx = vx - cx
            dy = vy - cy
            lx = dx * cos_o - dy * sin_o
            ly = dx * sin_o + dy * cos_o
            for k in range(6):
                a_k = k * np.pi / 3
                val = lx * np.cos(a_k) + ly * np.sin(a_k)
                if val > max_val:
                    max_val = val
    
    return max(0.0, max_val * 2 / SQRT3)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial arrangement: 3x4 hexagonal grid
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * SQRT3 + (row % 2) * SQRT3 / 2
            y = row * 1.5
            positions.append([x, y])
    
    pos_arr = np.array(positions)
    centroid = pos_arr.mean(axis=0)
    pos_arr -= centroid
    
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    
    def objective(vars):
        pos = vars[:24].reshape(12, 2)
        oc = vars[24:26]
        oangle = vars[26]
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < SQRT3:
                    penalty += (SQRT3 - d) ** 2
        
        side = compute_min_side(pos, oc, oangle)
        return side + 50.0 * penalty
    
    x0 = np.concatenate([pos_arr.flatten(), outer_center, [outer_angle]])
    
    best_side = compute_min_side(pos_arr, outer_center, outer_angle)
    best_vars = x0.copy()
    
    starts = [x0]
    
    # Scaled up
    starts.append(np.concatenate([(pos_arr * 1.3).flatten(), outer_center, [outer_angle]]))
    
    # Rotated 30 degrees
    theta = np.radians(30)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pos3 = pos_arr @ rot.T
    starts.append(np.concatenate([pos3.flatten(), outer_center, [outer_angle]]))
    
    # Rotated 15 degrees
    theta = np.radians(15)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pos4 = pos_arr @ rot.T
    starts.append(np.concatenate([pos4.flatten(), outer_center, [outer_angle]]))
    
    for x_start in starts:
        if time.time() > deadline - 30:
            break
        try:
            result = minimize(objective, x_start, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_side + 0.1:
                pos = result.x[:24].reshape(12, 2)
                oc = result.x[24:26]
                oangle = result.x[26]
                
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if np.linalg.norm(pos[i] - pos[j]) < SQRT3 - 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    side = compute_min_side(pos, oc, oangle)
                    if side < best_side:
                        best_side = side
                        best_vars = result.x.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 20:
        try:
            result = minimize(objective, best_vars, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                                     'initial_simplex': 1e-7})
            pos = result.x[:24].reshape(12, 2)
            oc = result.x[24:26]
            oangle = result.x[26]
            
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if np.linalg.norm(pos[i] - pos[j]) < SQRT3 - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                side = compute_min_side(pos, oc, oangle)
                if side < best_side:
                    best_side = side
                    best_vars = result.x.copy()
        except Exception:
            pass
    
    pos = best_vars[:24].reshape(12, 2)
    oc = best_vars[24:26]
    oangle = best_vars[26]
    
    inner = [[pos[i,0], pos[i,1], 0.0] for i in range(N)]
    return inner, (oc[0], oc[1]), best_side, oangle
