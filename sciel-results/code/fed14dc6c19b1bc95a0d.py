import numpy as np
import time
import scipy.optimize

N = 11

def hex_vertices(x, y, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    verts = verts @ R.T
    verts += np.array([x, y])
    return verts

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Initial guess: 7-hex cluster + 4 extensions
    d = np.sqrt(3)
    # 7 hex cluster
    centers = [
        (0, 0),
        (d, 0), (d/2, 1.5), (-d/2, 1.5),
        (-d, 0), (-d/2, -1.5), (d/2, -1.5)
    ]
    # 4 extensions
    # Extend arms at 0, 60, 120, 180 degrees
    # Arm 0 (right): (2d, 0)
    # Arm 60 (top-right): (1.5d, 3.0)  -- Wait, (d/2, 1.5) + (d/2, 1.5) = (d, 3.0)? No.
    # Vector to (d/2, 1.5) is (d/2, 1.5). Next one is (d, 3.0).
    # Let's check distance: sqrt((d/2)^2 + 1.5^2) = sqrt(3/4 + 2.25) = sqrt(3) = d. Correct.
    # So (d, 3.0) is the next one in that direction?
    # (d/2, 1.5) is at angle 60.
    # (d, 3.0) is at angle 60? tan(60) = 1.732. 3/d = 3/1.732 = 1.732. Yes.
    # So (d, 3.0) is correct for the 2nd step.
    # Similarly for others.
    
    extensions = [
        (2*d, 0),          # 0 deg
        (d, 3.0),          # 60 deg
        (-d, 3.0),         # 120 deg
        (-2*d, 0)          # 180 deg
    ]
    
    all_centers = centers + extensions
    
    # Scale up slightly to avoid initial overlap issues in optimizer
    scale = 1.05
    init_params = []
    for x, y in all_centers:
        init_params.extend([x * scale, y * scale, 0.0])
    
    # Outer angle
    init_params.append(0.0)
    init_params = np.array(init_params)
    
    def calc_side_and_penalty(params):
        pts = params[:-1].reshape(N, 3)
        phi = params[-1]
        
        xs = pts[:, 0]
        ys = pts[:, 1]
        thetas = pts[:, 2]
        
        # Normals of outer hexagon
        phi_rad = np.deg2rad(phi)
        normals = np.array([
            [np.cos(phi_rad + k * np.pi / 3), np.sin(phi_rad + k * np.pi / 3)]
            for k in range(6)
        ])
        
        max_proj = 0.0
        for k in range(N):
            verts = hex_vertices(xs[k], ys[k], thetas[k])
            projs = verts @ normals.T
            max_proj = max(max_proj, np.max(projs))
            
        side = 2.0 * max_proj / np.sqrt(3)
        
        # Penalty
        penalty = 0.0
        min_dist = np.sqrt(3)
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
        
        return side, penalty

    def objective(params):
        side, penalty = calc_side_and_penalty(params)
        return side + 100.0 * penalty

    best_side = float('inf')
    best_params = init_params.copy()
    
    np.random.seed(42)
    
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        if r == 0:
            current_params = init_params.copy()
        else:
            # Perturb
            current_params = init_params + np.random.normal(0, 0.5, len(init_params))
            # Randomize angles a bit
            current_params[2::3] = np.random.uniform(-10, 10, N)
            # Randomize outer angle
            current_params[-1] = np.random.uniform(-30, 30)
            
        try:
            res = scipy.optimize.minimize(
                objective, 
                current_params, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            
            side, pen = calc_side_and_penalty(res.x)
            if pen < 1e-4:
                if side < best_side:
                    best_side = side
                    best_params = res.x.copy()
        except Exception:
            pass

    if best_side == float('inf'):
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for i in range(N):
        x, y, ang = best_params[i*3], best_params[i*3+1], best_params[i*3+2]
        inner.append([float(x), float(y), float(ang)])
        
    outer_angle = float(best_params[-1])
    return inner, (0.0, 0.0), float(best_side), outer_angle
