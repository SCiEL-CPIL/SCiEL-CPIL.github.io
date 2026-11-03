import numpy as np
from scipy.optimize import minimize

def construct():
    N = 11
    
    # Vertices of a unit regular hexagon (flat-top, side 1)
    # Centered at origin
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Inequalities for outer hexagon (flat-top, side L, centered at 0,0)
    # We want: A @ point <= b
    # The 6 supporting lines:
    # 1. x <= L  =>  x <= L
    # 2. x >= -L => -x <= L
    # 3. (sqrt(3)/2)x + (1/2)y <= (sqrt(3)/2)L
    # 4. (sqrt(3)/2)x + (1/2)y >= -(sqrt(3)/2)L  => -(sqrt(3)/2)x - (1/2)y <= (sqrt(3)/2)L
    # 5. -(sqrt(3)/2)x + (1/2)y <= (sqrt(3)/2)L
    # 6. -(sqrt(3)/2)x + (1/2)y >= -(sqrt(3)/2)L => (sqrt(3)/2)x - (1/2)y <= (sqrt(3)/2)L
    
    # Coefficients for the 6 inequalities: [A_x, A_y, b_coeff]
    # A_x * x + A_y * y <= b_coeff * L
    outer_ineq = [
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [np.sqrt(3)/2, 0.5, np.sqrt(3)/2],
        [-np.sqrt(3)/2, -0.5, np.sqrt(3)/2],
        [-np.sqrt(3)/2, 0.5, np.sqrt(3)/2],
        [np.sqrt(3)/2, -0.5, np.sqrt(3)/2]
    ]
    
    def constraints_func(x):
        # x layout: [x1, y1, ..., x11, y11, L]
        L = x[-1]
        if L < 1.0:
            return [1.0] # Infeasible
        
        pos = x[:-1].reshape(N, 2)
        
        cons = []
        
        # 1. Non-overlap: dist(i,j) >= 2
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                # dist^2 >= 4
                cons.append(dx*dx + dy*dy - 4.0)
                
        # 2. Inside outer hex
        # For each inner hex, check all 6 vertices
        for i in range(N):
            cx, cy = pos[i]
            for k in range(6):
                vx = cx + hex_verts[k,0]
                vy = cy + hex_verts[k,1]
                for m in range(6):
                    # A_x * vx + A_y * vy <= b_coeff * L
                    val = outer_ineq[m][0]*vx + outer_ineq[m][1]*vy - outer_ineq[m][2]*L
                    cons.append(-val) # We want val <= 0, so -val >= 0
            
        return cons

    def objective(x):
        return x[-1]
        
    best_L = float('inf')
    best_x = None
    
    # Run multiple optimizations with different initial guesses
    for seed in range(10):
        np.random.seed(seed)
        
        # Initial guess
        # Try a compact cluster
        # 1 center, 6 ring, 4 extra
        init_pos = []
        # Center
        init_pos.append([0.0, 0.0])
        # Ring 1 (distance 2)
        for k in range(6):
            theta = k * np.pi / 3
            init_pos.append([2.0 * np.cos(theta), 2.0 * np.sin(theta)])
        # Ring 2 (distance 3? or 2?)
        # Let's place 4 more at distance 3 in diagonal directions
        for k in range(4):
            theta = (k * 0.5 + 0.25) * np.pi / 3 # Offset
            init_pos.append([3.0 * np.cos(theta), 3.0 * np.sin(theta)])
            
        init_pos = np.array(init_pos)
        
        # Add some noise
        init_pos += np.random.uniform(-0.5, 0.5, init_pos.shape)
        
        L_init = 4.5
        x0 = np.concatenate([init_pos.flatten(), [L_init]])
        
        res = minimize(objective, x0, method='SLSQP', 
                       constraints=[{'type': 'ineq', 'fun': constraints_func}],
                       options={'maxiter': 1000, 'ftol': 1e-6})
        
        if res.success:
            if res.x[-1] < best_L:
                best_L = res.x[-1]
                best_x = res.x.copy()
        else:
            # Even if not fully successful, check if it's a valid improvement
            # and better than current best
            if res.x[-1] < best_L:
                # Verify constraints
                cons_val = constraints_func(res.x)
                if np.all(cons_val > -1e-4):
                    best_L = res.x[-1]
                    best_x = res.x.copy()

    if best_x is None:
        # Fallback to a known valid configuration (e.g. line)
        # Though unlikely to be reached
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Extract results
    pos = best_x[:-1].reshape(N, 2)
    L_out = best_x[-1]
    
    # Construct inner list
    inner = []
    for i in range(N):
        inner.append([pos[i,0], pos[i,1], 0.0])
        
    return inner, (0.0, 0.0), L_out, 0.0

