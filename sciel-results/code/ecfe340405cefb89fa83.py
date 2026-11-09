import numpy as np
from scipy.optimize import differential_evolution
import time

def squares_in_square10() -> tuple:
    n = 10
    # Variables: u_i, v_i in [0,1], s_i in [eps, 1]
    # x_i = s_i/2 + u_i * (1 - s_i)
    # y_i = s_i/2 + v_i * (1 - s_i)
    
    eps = 1e-6
    
    def unpack(params):
        # params: array of size 3*n
        # [u_0..u_n-1, v_0..v_n-1, s_0..s_n-1]
        u = params[:n]
        v = params[n:2*n]
        s = params[2*n:]
        
        x = s / 2.0 + u * (1.0 - s)
        y = s / 2.0 + v * (1.0 - s)
        
        centers = np.stack([x, y], axis=1)
        return centers, s

    def objective(params):
        centers, s = unpack(params)
        x = centers[:, 0]
        y = centers[:, 1]
        
        # Objective: maximize sum(s)
        val = np.sum(s)
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                # Check overlap
                dx = abs(x[i] - x[j])
                dy = abs(y[i] - y[j])
                t = (s[i] + s[j]) / 2.0
                
                # If dx < t and dy < t, they overlap
                # Penalty is the amount of overlap squared or product
                # Using a smooth penalty:
                # If dx < t, viol_x = t - dx, else 0
                # If dy < t, viol_y = t - dy, else 0
                # Penalty = viol_x * viol_y * M
                # This is 0 if no overlap, positive if overlap.
                
                viol_x = max(0.0, t - dx)
                viol_y = max(0.0, t - dy)
                
                # If both are positive, they overlap.
                # The product measures the area of intersection of the "forbidden" zones?
                # Actually, just penalize if they overlap.
                # A strong penalty is needed.
                if viol_x > 0 and viol_y > 0:
                    penalty += viol_x * viol_y
        
        # We want to maximize val - penalty * M
        # DE minimizes, so we return penalty * M - val
        M = 1000.0
        return penalty * M - val

    # Bounds for DE
    # u, v in [0, 1]
    # s in [eps, 1]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # u
    for _ in range(n):
        bounds.append((0.0, 1.0)) # v
    for _ in range(n):
        bounds.append((eps, 1.0)) # s

    # Run DE
    # maxiter and popsize need to be tuned for speed (600s limit)
    # 30 dimensions is moderate.
    # Let's try a reasonable number of iterations.
    
    start_time = time.time()
    time_limit = 550.0 # Safety margin
    
    # We can use a callback to check time
    def callback(xk, convergence):
        if time.time() - start_time > time_limit:
            return True # Stop optimization
        return False

    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=500, 
            popsize=20, # 20 * 30 = 600 individuals
            tol=1e-8, 
            mutation=(0.5, 1.5), 
            recombination=0.7,
            seed=42,
            callback=callback,
            polish=True # Local polish at the end
        )
        
        best_params = result.x
        centers, sides = unpack(best_params)
        
        # Verify validity
        # Check bounds
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -1e-7 or cx + s/2 > 1 + 1e-7 or cy - s/2 < -1e-7 or cy + s/2 > 1 + 1e-7:
                # Invalid, return fallback
                return _get_fallback()
                
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i,0] - centers[j,0])
                dy = abs(centers[i,1] - centers[j,1])
                t = (sides[i] + sides[j]) / 2.0 - 1e-7
                if dx < t and dy < t:
                    # Overlap detected
                    return _get_fallback()
                    
        return centers, sides
        
    except Exception:
        return _get_fallback()

def _get_fallback():
    n = 10
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides
