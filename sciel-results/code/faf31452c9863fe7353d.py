import numpy as np
from scipy.optimize import minimize
import time

N = 12

def construct():
    start_time = time.time()
    
    # Initial configuration: 3 rows, 4 columns
    # Grid spacing: dx = sqrt(3), dy = 1.5
    # Stagger: odd rows shifted by sqrt(3)/2
    
    rows = 3
    cols = 4
    
    init_params = []
    for i in range(rows):
        for j in range(cols):
            x = j * np.sqrt(3)
            y = i * 1.5
            # Center the grid
            # x range: 0 to 3*sqrt(3). Center x = 1.5*sqrt(3)
            # y range: 0 to 2*1.5 = 3. Center y = 1.5
            x -= 1.5 * np.sqrt(3)
            y -= 1.5
            if i % 2 == 1:
                x += np.sqrt(3) / 2
            angle = 0.0
            init_params.extend([x, y, angle])
            
    init_params = np.array(init_params)
    
    def objective(params):
        # Reshape params
        hexes = params.reshape((-1, 3))
        # Calculate vertices for each hexagon
        # Hexagon vertices relative to center (side 1)
        # Angles: 0, 60, 120, 180, 240, 300
        base_angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
        
        max_s_y = 0.0
        max_s_x = 0.0
        
        for k in range(N):
            cx, cy, ang_deg = hexes[k]
            ang_rad = np.deg2rad(ang_deg)
            
            # Vertices
            dx = np.cos(base_angles + ang_rad)
            dy = np.sin(base_angles + ang_rad)
            
            vx = cx + dx
            vy = cy + dy
            
            # Constraints for point (x,y) to be inside hex of side S centered at 0
            # 1. |y| <= S * sqrt(3)/2  => S >= 2/sqrt(3) * |y|
            # 2. |x| <= S - |y|/sqrt(3) => S >= |x| + |y|/sqrt(3)
            
            s1 = (2.0 / np.sqrt(3)) * np.abs(vy)
            s2 = np.abs(vx) + (1.0 / np.sqrt(3)) * np.abs(vy)
            
            local_max = np.max(np.maximum(s1, s2))
            
            if local_max > max_s_y:
                max_s_y = local_max
                
        return max_s_y

    # Optimization
    # Use Nelder-Mead for derivative-free optimization
    # Time limit check
    best_result = None
    best_score = objective(init_params)
    
    # Run optimization with a time limit
    # scipy minimize doesn't have a direct time limit, so we wrap it or just run it and hope it's fast.
    # Nelder-Mead on 36 dimensions might be slow. 
    # Let's try to reduce dimensions or use a simpler approach if it's too slow.
    # However, 36 dimensions is manageable for a few hundred iterations.
    
    try:
        # Set a reasonable maxiter to prevent hanging
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x
        else:
            best_result = init_params
            
    except Exception as e:
        # Fallback to initial if optimization fails
        best_result = init_params
        best_score = objective(init_params)
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 110:
        # If we are running out of time, just return what we have
        pass
        
    # Extract final parameters
    hexes = best_result.reshape((-1, 3))
    inner = [[float(x), float(y), float(a)] for x, y, a in hexes]
    
    # The outer hexagon is centered at (0,0), side length best_score, rotation 0
    center = (0.0, 0.0)
    side = float(best_score)
    angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
