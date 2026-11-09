import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    timeout = 110 # Safety margin for 120s limit
    
    best_val = -np.inf
    best_res = None
    
    # Sweep aspect ratios
    # w + h = 2
    # w in [0.5, 1.5]
    for w in np.arange(0.6, 1.5, 0.1):
        h = 2.0 - w
        if time.time() - start_time > timeout:
            break
            
        # Try a few initializations
        inits = []
        
        # 1. Grid
        n_c = int(np.round(np.sqrt(N * w / h)))
        if n_c < 1: n_c = 1
        n_r = int(np.ceil(N / n_c))
        
        # Calculate grid positions
        xs = []
        ys = []
        rs = []
        for i in range(n_r):
            for j in range(n_c):
                if len(xs) >= N: break
                x = (j + 0.5) * w / n_c
                y = (i + 0.5) * h / n_r
                r = min(w/n_c, h/n_r) / 2 * 0.9 # Slightly smaller to allow movement
                xs.append(x)
                ys.append(y)
                rs.append(r)
        inits.append(np.array(xs + ys + rs)) # Wait, vector structure [x1, y1, r1, x2, y2, r2...]
        
        # Correct vector structure for init
        def make_vec(xs, ys, rs):
            v = np.zeros(3*N)
            v[0::3] = xs
            v[1::3] = ys
            v[2::3] = rs
            return v
            
        v_grid = make_vec(xs, ys, rs)
        inits = [v_grid]
        
        # 2. Random jitter
        for _ in range(3):
            v_rand = v_grid.copy()
            # Jitter positions
            v_rand[0::3] += np.random.uniform(-0.05, 0.05, N)
            v_rand[1::3] += np.random.uniform(-0.05, 0.05, N)
            # Clip to bounds
            v_rand[0::3] = np.clip(v_rand[0::3], 0, w)
            v_rand[1::3] = np.clip(v_rand[1::3], 0, h)
            inits.append(v_rand)
            
        for v0 in inits:
            if time.time() - start_time > timeout:
                break
                
            # Define objective
            def obj(v):
                x = v[0::3]
                y = v[1::3]
                r = v[2::3]
                
                # Bounds check (soft)
                pen = 0
                for i in range(N):
                    pen += max(0, r[i] - x[i])**2
                    pen += max(0, r[i] - (w - x[i]))**2
                    pen += max(0, r[i] - y[i])**2
                    pen += max(0, r[i] - (h - y[i]))**2
                    
                # Overlap
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d = np.sqrt(dx*dx + dy*dy)
                        dist_min = r[i] + r[j]
                        if d < dist_min:
                            pen += (dist_min - d)**2
                            
                return -np.sum(r) + 1000 * pen

            bounds = [(0, w), (0, h), (0, 1.0)] * N
            
            res = minimize(obj, v0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500})
            
            if res.success or res.fun < best_val:
                # Check validity
                x = res.x[0::3]
                y = res.x[1::3]
                r = res.x[2::3]
                
                # Verify constraints strictly
                valid = True
                for i in range(N):
                    if x[i] < r[i] - 1e-6 or x[i] > w - r[i] + 1e-6: valid = False
                    if y[i] < r[i] - 1e-6 or y[i] > h - r[i] + 1e-6: valid = False
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if d < r[i] + r[j] - 1e-6: valid = False
                
                if valid:
                    val = np.sum(r)
                    if val > best_val:
                        best_val = val
                        best_res = res.x.copy()

    if best_res is None:
        # Fallback to seed-like result
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Format output
    x = best_res[0::3]
    y = best_res[1::3]
    r = best_res[2::3]
    
    # Final check and adjust for floating point errors
    # Ensure perimeter constraint is met (it is by construction w+h=2)
    
    out = np.zeros((N, 3))
    out[:, 0] = x
    out[:, 1] = y
    out[:, 2] = r
    
    return out
