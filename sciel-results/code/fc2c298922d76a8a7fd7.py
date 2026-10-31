import numpy as np
from scipy.optimize import minimize
import time

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2

def construct():
    start_time = time.time()
    best_t = 0.0
    best_points = None
    
    # Precompute triplet indices
    triplets = [(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)]
    
    def objective(vars):
        # vars: [x1, y1, ..., xN, yN, t]
        t = vars[-1]
        return -t

    def constraints(vars):
        cons = []
        t = vars[-1]
        pts = vars[:-1].reshape(N, 2)
        
        # 1. Triangle constraints for each point
        # SLSQP 'ineq' means fun(x) >= 0
        for i in range(N):
            x, y = pts[i]
            # y >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, y: y})
            # sqrt(3)*x - y >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, y: SQRT3 * x - y})
            # sqrt(3)*(1-x) - y >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, y: SQRT3 * (1 - x) - y})

        # 2. Area constraints
        # Area >= t  =>  Area - t >= 0
        for i, j, k in triplets:
            def area_con(vars, i=i, j=j, k=k):
                pts = vars[:-1].reshape(N, 2)
                xi, yi = pts[i]
                xj, yj = pts[j]
                xk, yk = pts[k]
                D = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
                # Area = 0.5 * |D|
                # Use sqrt(D^2) for differentiability (derivative is sign(D))
                return 0.5 * np.sqrt(D * D + 1e-12) - t
            cons.append({'type': 'ineq', 'fun': area_con})
            
        return cons

    # Multi-start
    n_restarts = 100
    rng = np.random.default_rng(42)
    
    for r in range(n_restarts):
        if time.time() - start_time > 110: # Safety margin
            break
            
        # Random init inside triangle using barycentric coordinates
        u = rng.random(N)
        v = rng.random(N)
        # Ensure u+v <= 1
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        w = 1 - u - v
        
        # Points: u*A + v*B + w*C
        # A=(0,0), B=(1,0), C=(0.5, H)
        x = 1 * v + 0.5 * w
        y = H * w
        pts = np.column_stack((x, y))
        
        # Initial t
        # Estimate a small t
        t0 = 0.005
        
        x0 = np.concatenate([pts.flatten(), [t0]])
        
        try:
            res = minimize(
                objective, x0, method='SLSQP', 
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            if res.success:
                t_val = res.x[-1]
                if t_val > best_t:
                    best_t = t_val
                    best_points = res.x[:-1].reshape(N, 2)
        except Exception:
            pass

    if best_points is None:
        # Fallback: return random points
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        w = 1 - u - v
        x = 1 * v + 0.5 * w
        y = H * w
        return np.column_stack((x, y))
        
    return best_points
