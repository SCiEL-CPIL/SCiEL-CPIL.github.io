import numpy as np
from scipy.optimize import minimize
import time

N = 12
SQRT3 = np.sqrt(3.0)

def hex_norm(p):
    """
    Calculate the hexagonal norm of a point p (x, y).
    Assumes hexagon aligned with x-axis (flat top/bottom).
    """
    x, y = p
    v1 = abs(x)
    v2 = abs(0.5 * x + SQRT3 / 2.0 * y)
    v3 = abs(-0.5 * x + SQRT3 / 2.0 * y)
    return max(v1, v2, v3)

def min_dist_hex(delta_deg):
    """
    Calculate the minimum center distance for two unit hexagons
    with relative rotation delta_deg to be non-overlapping.
    This is the max of the support function of the Minkowski sum.
    """
    delta_rad = np.deg2rad(delta_deg)
    # Sample angles 0 to 360
    # h(phi) = (sqrt(3)/2) / cos(mod(phi, 60) - 30)
    # We need max of h(phi) + h(phi + delta)
    phis = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    
    # Calculate h(phi)
    mod_phis = np.mod(phis, np.pi / 3.0)
    h1 = (SQRT3 / 2.0) / np.cos(mod_phis - np.pi / 6.0)
    
    # Calculate h(phi + delta)
    phis2 = phis + delta_rad
    mod_phis2 = np.mod(phis2, np.pi / 3.0)
    h2 = (SQRT3 / 2.0) / np.cos(mod_phis2 - np.pi / 6.0)
    
    max_val = np.max(h1 + h2)
    return max_val

def calc_S_and_penalty(centers, angles_deg):
    """
    Calculate S and collision penalty.
    """
    max_norm = 0.0
    base_angles = np.deg2rad(np.arange(0, 360, 60))
    
    # 1. Calculate S (max extent)
    for i in range(len(centers)):
        cx, cy = centers[i]
        angle_rad = np.deg2rad(angles_deg[i])
        
        for ba in base_angles:
            vx = np.cos(ba)
            vy = np.sin(ba)
            
            rx = vx * np.cos(angle_rad) - vy * np.sin(angle_rad)
            ry = vx * np.sin(angle_rad) + vy * np.cos(angle_rad)
            
            px = cx + rx
            py = cy + ry
            
            n = hex_norm((px, py))
            if n > max_norm:
                max_norm = n
                
    S = (2.0 / SQRT3) * max_norm
    
    # 2. Calculate Penalty
    penalty = 0.0
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            
            delta = abs(angles_deg[i] - angles_deg[j]) % 360.0
            # Symmetry: min_dist(360-d) == min_dist(d)
            if delta > 180:
                delta = 360 - delta
                
            req_dist = min_dist_hex(delta)
            
            if dist < req_dist:
                penalty += (req_dist - dist) ** 2
                
    return S, penalty

def objective(x):
    """
    x is a flat array of 3*N coordinates: [x1, y1, a1, x2, y2, a2, ...]
    """
    params = np.array(x).reshape(N, 3)
    centers = params[:, :2]
    angles = params[:, 2]
    
    S, pen = calc_S_and_penalty(centers, angles)
    
    # Weight for penalty
    # S is around 3-4. Penalty should be significant.
    # Use a large weight to ensure non-overlap
    return S + 100.0 * pen

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = float('inf')
    best_params = None
    
    # Initial guesses
    guesses = []
    
    # 1. Lattice cluster
    centers_list = []
    centers_list.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers_list.append((SQRT3 * np.cos(angle), SQRT3 * np.sin(angle)))
    for k in range(5):
        angle = np.deg2rad(k * 60)
        centers_list.append((2 * SQRT3 * np.cos(angle), 2 * SQRT3 * np.sin(angle)))
        
    angles_list = [0.0] * N
    params_list = []
    for i in range(N):
        params_list.append([centers_list[i][0], centers_list[i][1], angles_list[i]])
    guesses.append(np.array(params_list).flatten())
    
    # 2. Random guesses
    np.random.seed(42)
    for _ in range(10):
        # Random positions in a circle of radius 4
        r = np.random.uniform(0, 4, size=(N, 2))
        # Convert polar to cartesian? No, just random x,y
        # Actually, generate random points in a hexagon of side 4
        # Simplified: random in square [-4, 4]
        pts = np.random.uniform(-4, 4, size=(N, 2))
        angs = np.random.uniform(0, 60, size=N)
        p_list = []
        for i in range(N):
            p_list.append([pts[i][0], pts[i][1], angs[i]])
        guesses.append(np.array(p_list).flatten())

    # Run optimization on each guess
    for x0 in guesses:
        if time.time() > deadline - 10.0:
            break
            
        try:
            result = minimize(objective, x0, method='Powell', options={'maxiter': 300, 'disp': False})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass

    if best_params is None:
        # Fallback
        inner = [[0.0, 0.0, 0.0]] * N
        return inner, (0.0, 0.0), 10.0, 0.0

    # Extract best result
    params = best_params.reshape(N, 3)
    centers = params[:, :2]
    angles = params[:, 2]
    
    S, _ = calc_S_and_penalty(centers, angles)
    
    inner = []
    for i in range(N):
        inner.append([float(centers[i][0]), float(centers[i][1]), float(angles[i])])
        
    return inner, (0.0, 0.0), float(S), 0.0
