import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM_FACTOR = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: Simple grid
        return get_fallback()

    # Initial configuration: Hex flower
    # Axial coords: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (0,-2)
    axial = [
        (0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    
    initial_centers = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        initial_centers.append([x, y])
    
    initial_angles = [0.0] * N
    initial_C = [0.0, 0.0]
    initial_Phi = 0.0
    initial_S = 6.0
    
    x0 = []
    for i in range(N):
        x0.append(initial_centers[i][0])
        x0.append(initial_centers[i][1])
        x0.append(initial_angles[i])
    x0.extend(initial_C)
    x0.append(initial_Phi)
    x0.append(initial_S)
    
    bounds = []
    for i in range(N):
        bounds.append((-6.0, 6.0)) # x
        bounds.append((-6.0, 6.0)) # y
        bounds.append((0.0, math.pi/3)) # angle
    bounds.append((-6.0, 6.0)) # Cx
    bounds.append((-6.0, 6.0)) # Cy
    bounds.append((0.0, math.pi/3)) # Phi
    bounds.append((3.5, 10.0)) # S
    
    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10
            
        # Unpack
        centers = np.array(params[:3*N]).reshape(N, 3)
        Cx, Cy = params[3*N], params[3*N+1]
        Phi = params[3*N+2]
        S = params[3*N+3]
        
        # 1. Containment Penalty
        A = S * APOTHEM_FACTOR
        # Outer hex normals
        angles = Phi + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        penalty_contain = 0.0
        for i in range(N):
            cx, cy, theta = centers[i]
            # Vertices of inner hex i
            v_angles = theta + np.arange(6) * math.pi / 3.0
            vx = cx + np.cos(v_angles)
            vy = cy + np.sin(v_angles)
            
            # Project vertices onto outer normals
            # Vector from C to vertex
            dx = vx - Cx
            dy = vy - Cy
            projs = np.abs(dx * normals[:, 0] + dy * normals[:, 1])
            
            # Violation: max(0, proj - A)
            violations = np.maximum(0, projs - A)
            penalty_contain += np.sum(violations ** 2)
            
        # 2. Overlap Penalty
        penalty_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 2.0:
                    violation = 2.0 - dist
                    penalty_overlap += violation ** 2
                    
        # Objective: Minimize S, penalize violations
        return S + 10.0 * penalty_contain + 10.0 * penalty_overlap

    # Optimize
    best_x = x0
    best_f = objective(x0)
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds,
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        if res.fun < best_f:
            best_x = res.x
            best_f = res.fun
    except Exception:
        pass
        
    # Try L-BFGS-B as well
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res2 = minimize(objective, best_x, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 500, 'ftol': 1e-10})
            if res2.fun < best_f:
                best_x = res2.x
                best_f = res2.fun
    except Exception:
        pass

    # Extract best result
    centers = np.array(best_x[:3*N]).reshape(N, 3)
    Cx, Cy = best_x[3*N], best_x[3*N+1]
    Phi = best_x[3*N+2]
    S = best_x[3*N+3]
    
    # Verify and adjust S to be exactly the minimum required
    # Recalculate required S for the found positions
    required_S = 3.5
    A = S * APOTHEM_FACTOR
    angles = Phi + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    max_proj = 0
    for i in range(N):
        cx, cy, theta = centers[i]
        v_angles = theta + np.arange(6) * math.pi / 3.0
        vx = cx + np.cos(v_angles)
        vy = cy + np.sin(v_angles)
        dx = vx - Cx
        dy = vy - Cy
        projs = np.abs(dx * normals[:, 0] + dy * normals[:, 1])
        max_proj = max(max_proj, np.max(projs))
        
    # The required apothem is max_proj
    # S = A / APOTHEM_FACTOR
    S_final = max_proj / APOTHEM_FACTOR
    
    # Add small safety margin
    S_final *= 1.001
    
    # Construct return values
    inner = []
    for i in range(N):
        cx, cy, theta = centers[i]
        inner.append([float(cx), float(cy), float(math.degrees(theta))])
        
    center = [float(Cx), float(Cy)]
    side = float(S_final)
    rotation = float(math.degrees(Phi))
    
    return inner, center, side, rotation

def get_fallback():
    # Simple valid configuration
    axial = [
        (0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    inner = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([float(x), float(y), 0.0])
        
    # Calculate bounding box
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    max_x = max(xs) + 1.0
    min_x = min(xs) - 1.0
    max_y = max(ys) + 1.0
    min_y = min(ys) - 1.0
    
    # Approximate side length for a hex covering this
    # Width = max_x - min_x, Height = max_y - min_y
    # For a flat-top hex, width = 2*S, height = sqrt(3)*S
    # For pointy-top, width = sqrt(3)*S, height = 2*S
    # We can rotate the outer hex to fit better, but for fallback, just take max
    S = max(max_x - min_x, (max_y - min_y) / APOTHEM_FACTOR) / 2.0
    S = max(S, 5.0) # Ensure it's big enough
    
    return inner, [0.0, 0.0], float(S), 0.0
