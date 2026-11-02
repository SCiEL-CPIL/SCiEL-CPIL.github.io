import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

N = 21

def construct():
    start_time = time.time()
    time_limit = 130.0  # Safety margin
    
    # Variables: x1, y1, r1, ..., x21, y21, r21
    # Aspect ratio a = w/h, w+h=2 => h = 2/(1+a), w = 2a/(1+a)
    
    def get_wh(a):
        h = 2.0 / (1.0 + a)
        w = 2.0 * a / (1.0 + a)
        return w, h

    def objective(params, a):
        # params: [x1, y1, r1, ..., xN, yN, rN]
        w, h = get_wh(a)
        pos = params.reshape(N, 3)
        xs = pos[:, 0]
        ys = pos[:, 1]
        rs = pos[:, 2]
        
        # Penalty for negative radii
        pen = np.sum(np.maximum(0, -rs)**2)
        
        # Penalty for boundary violations
        # x - r < 0, w - x - r < 0, etc.
        pen += np.sum(np.maximum(0, rs - xs)**2)
        pen += np.sum(np.maximum(0, rs - (w - xs))**2)
        pen += np.sum(np.maximum(0, rs - ys)**2)
        pen += np.sum(np.maximum(0, rs - (h - ys))**2)
        
        # Penalty for overlaps
        # We can vectorize this for speed
        # dist^2 = (xi-xj)^2 + (yi-yj)^2
        # overlap = max(0, ri + rj - sqrt(dist^2))^2
        # To avoid sqrt, we can use (ri+rj)^2 - dist^2 if positive? 
        # No, penalty should be smooth. 
        # Let's use a standard smooth approximation or just loop if N is small (21 is small).
        # N=21, pairs = 210. Loop is fine.
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                rsum = rs[i] + rs[j]
                if rsum > dist:
                    overlap_pen += (rsum - dist)**2
        pen += overlap_pen
        
        # Objective: minimize -sum(r) + lambda * penalty
        # Lambda should be large enough to enforce constraints
        lam = 100.0
        return -np.sum(rs) + lam * pen

    def is_valid(params, a, tol=1e-5):
        w, h = get_wh(a)
        pos = params.reshape(N, 3)
        xs = pos[:, 0]
        ys = pos[:, 1]
        rs = pos[:, 2]
        
        if np.any(rs < -tol): return False
        if np.any(xs < -tol) or np.any(w - xs < -tol): return False
        if np.any(ys < -tol) or np.any(h - ys < -tol): return False
        if np.any(rs - xs < -tol) or np.any(rs - (w - xs) < -tol): return False
        if np.any(rs - ys < -tol) or np.any(rs - (h - ys) < -tol): return False
        
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    best_val = 0.0
    best_params = None
    best_a = 1.0
    
    # Aspect ratio search
    # Try a range of aspect ratios
    a_values = np.linspace(0.8, 1.25, 20)
    
    rng = np.random.default_rng(42)
    
    for a in a_values:
        if time.time() - start_time > time_limit:
            break
            
        w, h = get_wh(a)
        
        # Multiple restarts for this aspect ratio
        n_restarts = 15
        for k in range(n_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initial guess: Grid
            # Arrange N circles in a grid
            rows = int(np.ceil(np.sqrt(N * (h/w))))
            cols = int(np.ceil(N / rows))
            
            # Create grid positions
            xs_init = np.zeros(N)
            ys_init = np.zeros(N)
            rs_init = np.zeros(N)
            
            for i in range(N):
                row = i // cols
                col = i % cols
                # Center in cell
                xs_init[i] = (col + 0.5) * (w / cols)
                ys_init[i] = (row + 0.5) * (h / rows)
                # Initial radius: small fraction of cell size
                cell_w = w / cols
                cell_h = h / rows
                rs_init[i] = 0.4 * min(cell_w, cell_h)
                
            # Add some noise to break symmetry
            xs_init += rng.normal(0, 0.01, N)
            ys_init += rng.normal(0, 0.01, N)
            rs_init += rng.normal(0, 0.01, N)
            
            # Clip to valid range initially to help optimizer
            xs_init = np.clip(xs_init, 0.1, w-0.1)
            ys_init = np.clip(ys_init, 0.1, h-0.1)
            rs_init = np.clip(rs_init, 0.01, min(w, h)/2)
            
            params0 = np.concatenate([xs_init, ys_init, rs_init])
            
            # Optimize
            try:
                res = minimize(objective, params0, args=(a,), method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
                if res.success:
                    val = -res.fun + 100.0 * 0 # Wait, objective returns -sum(r) + lam*pen
                    # We need to check validity and calculate actual sum of radii
                    if is_valid(res.x, a):
                        sum_r = np.sum(res.x[2::3])
                        if sum_r > best_val:
                            best_val = sum_r
                            best_params = res.x.copy()
                            best_a = a
            except Exception:
                pass

    # Final polish?
    # If we have a best_params, we can try to refine it further with a tighter tolerance
    if best_params is not None:
        if time.time() - start_time < time_limit - 5:
            try:
                res = minimize(objective, best_params, args=(best_a,), method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10})
                if is_valid(res.x, best_a):
                    sum_r = np.sum(res.x[2::3])
                    if sum_r > best_val:
                        best_val = sum_r
                        best_params = res.x.copy()
            except Exception:
                pass

    # Format output: array of shape (n, 3)
    # Columns: x, y, r
    if best_params is not None:
        out = best_params.reshape(N, 3)
        # Ensure radii are non-negative
        out[:, 2] = np.maximum(0, out[:, 2])
        return out
    else:
        # Fallback to seed-like behavior
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
