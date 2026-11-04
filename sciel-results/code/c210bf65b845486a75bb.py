import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    best_score = -np.inf
    best_result = None
    best_dims = (1.0, 1.0)
    
    test_dims = [
        (1.0, 1.0),
        (0.9, 1.1),
        (1.1, 0.9)
    ]
    
    n_restarts = 3
    
    def smooth_max_neg(v):
        u = -v
        return np.where(u > 0, u + np.log1p(np.exp(-u)), np.log1p(np.exp(u)))

    for W, H in test_dims:
        for restart in range(n_restarts):
            # Initialize
            cols = int(np.sqrt(N * W / H))
            rows = int(N / cols)
            if rows * cols < N:
                rows += 1
            
            x0 = np.zeros(N * 3)
            for i in range(N):
                col = i % cols
                row = i // cols
                cx = (col + 0.5) * (W / cols)
                cy = (row + 0.5) * (H / rows)
                r = min(W / cols, H / rows) * 0.4
                x0[i*3] = cx + np.random.uniform(-0.05, 0.05)
                x0[i*3+1] = cy + np.random.uniform(-0.05, 0.05)
                x0[i*3+2] = r
                
            bnds = []
            for i in range(N):
                bnds.append((0, W))
                bnds.append((0, H))
                bnds.append((0, min(W, H)/2))
                
            def make_objective(k):
                def objective(p):
                    x = p[0::3]
                    y = p[1::3]
                    r = p[2::3]
                    
                    sum_r = np.sum(r)
                    pen = 0.0
                    
                    pen += np.sum(smooth_max_neg(x - r))
                    pen += np.sum(smooth_max_neg(W - x - r))
                    pen += np.sum(smooth_max_neg(y - r))
                    pen += np.sum(smooth_max_neg(H - y - r))
                    
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            dist_sq = dx*dx + dy*dy
                            rad_sum = r[i] + r[j]
                            constraint_val = dist_sq - rad_sum*rad_sum
                            pen += smooth_max_neg(constraint_val)
                            
                    return -sum_r + k * pen
                return objective

            # Stage 1: Low penalty
            try:
                res1 = minimize(make_objective(10.0), x0, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 100})
                
                # Stage 2: High penalty
                res2 = minimize(make_objective(100.0), res1.x, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 200})
                
                x = res2.x[0::3]
                y = res2.x[1::3]
                r = res2.x[2::3]
                
                # Feasibility check
                feasible = True
                if np.any(r <= 0): feasible = False
                if np.any(x < r - 1e-6) or np.any(W - x < r - 1e-6) or np.any(y < r - 1e-6) or np.any(H - y < r - 1e-6):
                    feasible = False
                for i in range(N):
                    for j in range(i+1, N):
                        if (x[i]-x[j])**2 + (y[i]-y[j])**2 < (r[i]+r[j])**2 - 1e-6:
                            feasible = False
                            break
                    if not feasible: break
                
                if feasible:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_result = res2.x.copy()
                        best_dims = (W, H)
                        
            except Exception:
                continue
                
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    x = best_result[0::3]
    y = best_result[1::3]
    r = best_result[2::3]
    
    # Final safety shrink
    # If any constraint is violated by a tiny margin, shrink all radii by 1%
    # This ensures validity at the cost of a small score drop.
    W, H = best_dims
    def check_valid(x, y, r, W, H):
        if np.any(r <= 0): return False
        if np.any(x < r) or np.any(W - x < r) or np.any(y < r) or np.any(H - y < r): return False
        for i in range(N):
            for j in range(i+1, N):
                if (x[i]-x[j])**2 + (y[i]-y[j])**2 < (r[i]+r[j])**2:
                    return False
        return True
        
    if not check_valid(x, y, r, W, H):
        # Shrink radii
        r = r * 0.99
        # Re-check
        if not check_valid(x, y, r, W, H):
             # Return seed if still invalid
             side = int(np.ceil(np.sqrt(N)))
             cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
             return np.array(cells[:N])

    return np.column_stack((x, y, r))
