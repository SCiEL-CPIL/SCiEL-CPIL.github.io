import numpy as np
from scipy.optimize import minimize
import time

N = 12

def get_hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    angles = angle_rad + np.deg2rad(np.arange(0, 360, 60))
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def sat_overlap(v1, v2):
    # v1, v2 are 6x2 arrays
    n1 = []
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        ex = p2[0] - p1[0]
        ey = p2[1] - p1[1]
        n1.append([-ey, ex])
    n1 = np.array(n1)
    
    n2 = []
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i+1)%6]
        ex = p2[0] - p1[0]
        ey = p2[1] - p1[1]
        n2.append([-ey, ex])
    n2 = np.array(n2)
    
    all_normals = np.vstack((n1, n2))
    
    for n in all_normals:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def is_inside_container(vertex, L):
    apothem = L * np.sqrt(3) / 2
    x, y = vertex
    for k in range(6):
        alpha = np.deg2rad(30 + k * 60)
        d = x * np.cos(alpha) + y * np.sin(alpha)
        if d > apothem:
            return False
    return True

def objective(params):
    L = params[0]
    if L < 2.5:
        return 1e6
        
    penalty = 0.0
    hexes = []
    centers = []
    
    for i in range(N):
        idx = 1 + i * 3
        xi, yi, thi = params[idx], params[idx+1], params[idx+2]
        centers.append([xi, yi])
        verts = get_hex_vertices(xi, yi, 1.0, thi)
        
        # Containment penalty
        apothem = L * np.sqrt(3) / 2
        for v in verts:
            for k in range(6):
                alpha = np.deg2rad(30 + k * 60)
                d = v[0] * np.cos(alpha) + v[1] * np.sin(alpha)
                if d > apothem:
                    penalty += (d - apothem) ** 2 * 50.0
        hexes.append(verts)
        
    # Overlap penalty
    for i in range(N):
        for j in range(i+1, N):
            c1 = centers[i]
            c2 = centers[j]
            dist = np.linalg.norm(np.array(c1) - np.array(c2))
            
            # Only check SAT if they are close enough to potentially overlap
            # Max distance for overlap is 2 (point to point)
            if dist < 2.0:
                if sat_overlap(hexes[i], hexes[j]):
                    # Penalty scales with how much they violate the min distance
                    # Min distance is approx 1.732
                    violation = max(0, 1.732 - dist)
                    penalty += violation ** 2 * 1000.0
            else:
                # If dist >= 2.0, they cannot overlap for unit hexagons?
                # Wait, if two hexagons are point-to-point, distance 2 means they touch.
                # So if dist > 2, they are definitely separated.
                pass

    return L + penalty

def construct():
    best_L = 40.0
    best_params = None
    
    # Initial guess: 7 cluster + 5 outer
    x0 = [0.0, 0.0, 0.0]
    ring1 = []
    for k in range(6):
        angle = k * 60
        r = np.sqrt(3)
        x = r * np.cos(np.deg2rad(angle))
        y = r * np.sin(np.deg2rad(angle))
        ring1.append([x, y, 0.0])
    ring2 = []
    for k in range(5):
        angle = 30 + k * 60
        r = 2 * np.sqrt(3)
        x = r * np.cos(np.deg2rad(angle))
        y = r * np.sin(np.deg2rad(angle))
        ring2.append([x, y, 0.0])
        
    init_hexes = [x0] + ring1 + ring2
    init_L = 3.5
    
    init_params = [init_L]
    for h in init_hexes:
        init_params.extend(h)
        
    start_time = time.time()
    
    # Run optimization
    while time.time() - start_time < 110:
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-5})
        
        L_opt = res.x[0]
        valid = True
        hexes = []
        centers = []
        
        for i in range(N):
            idx = 1 + i * 3
            xi, yi, thi = res.x[idx], res.x[idx+1], res.x[idx+2]
            centers.append([xi, yi])
            verts = get_hex_vertices(xi, yi, 1.0, thi)
            for v in verts:
                if not is_inside_container(v, L_opt):
                    valid = False
                    break
            if not valid: break
            hexes.append(verts)
            
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    if sat_overlap(hexes[i], hexes[j]):
                        valid = False
                        break
                if not valid: break
                
        if valid and L_opt < best_L:
            best_L = L_opt
            best_params = res.x.copy()
            
        # Prepare next iteration
        # Add noise
        init_params = res.x + np.random.normal(0, 0.05, size=len(res.x))
        init_params[0] = max(3.0, init_params[0] + np.random.normal(0, 0.1))
        
        if best_L < 3.2:
            break

    if best_params is None:
        # Fallback to initial guess if no valid solution found
        best_params = init_params
        best_L = init_params[0]

    inner = []
    for i in range(N):
        idx = 1 + i * 3
        xi, yi, thi = best_params[idx], best_params[idx+1], best_params[idx+2]
        inner.append([xi, yi, thi])
        
    return inner, (0.0, 0.0), best_L, 0.0
