import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # Safety margin before 120s kill

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    
    np.random.seed(42)
    
    aspect_ratios = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.25]
    
    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        if h <= 0.1:
            continue
            
        max_restarts = 4
        
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE - 8:
                break
            
            # Improved initialization: hexagonal-like pattern
            cols = int(np.sqrt(N * w / h))
            if cols < 2: cols = 2
            rows = int(np.ceil(N / cols))
            while cols * rows < N:
                if cols <= rows:
                    cols += 1
                else:
                    rows += 1
            
            positions = np.zeros((N, 2))
            idx = 0
            for r_idx in range(rows):
                for c_idx in range(cols):
                    if idx >= N:
                        break
                    # Hexagonal offset for odd rows
                    x_offset = 0.0
                    if r_idx % 2 == 1 and cols > 1:
                        x_offset = 0.5 * w / cols
                    
                    x = (c_idx + 0.5) * w / cols + x_offset
                    y = (r_idx + 0.5) * h / rows
                    
                    # Jitter for different restarts
                    if restart > 0:
                        jitter = 0.15 * min(w/cols, h/rows)
                        x += np.random.uniform(-jitter, jitter)
                        y += np.random.uniform(-jitter, jitter)
                    
                    x = np.clip(x, 0.02, w - 0.02)
                    y = np.clip(y, 0.02, h - 0.02)
                    positions[idx] = [x, y]
                    idx += 1
            
            # Calculate initial radii
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
                    min_dist = min(min_dist, d)
            for i in range(N):
                min_dist = min(min_dist, positions[i,0], w-positions[i,0], positions[i,1], h-positions[i,1])
            
            r0 = 0.45 * min_dist
            radii = np.full(N, r0)
            
            x0 = np.zeros(3*N)
            for i in range(N):
                x0[3*i] = positions[i, 0]
                x0[3*i+1] = positions[i, 1]
                x0[3*i+2] = radii[i]
            
            # Phase 1: Smooth penalty-based optimization (SLSQP with soft constraints)
            def objective_smooth(vars_):
                x = vars_[0::3]
                y = vars_[1::3]
                r = vars_[2::3]
                
                # Penalty for violations
                penalty = 0.0
                # Boundary violations
                for i in range(N):
                    if x[i] - r[i] < 0: penalty += 100.0 * (x[i] - r[i])**2
                    if w - x[i] - r[i] < 0: penalty += 100.0 * (w - x[i] - r[i])**2
                    if y[i] - r[i] < 0: penalty += 100.0 * (y[i] - r[i])**2
                    if h - y[i] - r[i] < 0: penalty += 100.0 * (h - y[i] - r[i])**2
                # Overlap violations
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        overlap = r[i] + r[j] - dist
                        if overlap > 0:
                            penalty += 50.0 * overlap**2
                
                return -np.sum(r) + penalty
            
            # Run smooth phase
            try:
                result1 = minimize(
                    objective_smooth, x0, method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10}
                )
                x_smooth = result1.x
            except Exception:
                x_smooth = x0
            
            # Phase 2: SLSQP with hard constraints
            def objective(vars_):
                r = vars_[2::3]
                return -np.sum(r)
            
            def cons_inside(vars_):
                x = vars_[0::3]
                y = vars_[1::3]
                r = vars_[2::3]
                return np.concatenate([
                    x - r,
                    w - x - r,
                    y - r,
                    h - y - r
                ])
            
            def cons_non_overlap(vars_):
                x = vars_[0::3]
                y = vars_[1::3]
                r = vars_[2::3]
                cons = []
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        cons.append(dist - r[i] - r[j])
                return np.array(cons)
            
            def cons_positive_r(vars_):
                r = vars_[2::3]
                return r
            
            constraints = [
                {'type': 'ineq', 'fun': cons_inside},
                {'type': 'ineq', 'fun': cons_non_overlap},
                {'type': 'ineq', 'fun': cons_positive_r}
            ]
            
            try:
                result2 = minimize(
                    objective, x_smooth, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 300, 'ftol': 1e-10}
                )
                
                # Phase 3: High-precision polish
                x_polish = result2.x
                result3 = minimize(
                    objective, x_polish, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                x_res = result3.x
                
                x_res = x_res[0::3]
                y_res = x_res[1::3]
                r_res = x_res[2::3]
                
                # Validation
                valid = True
                if np.any(r_res < -1e-9):
                    valid = False
                else:
                    if np.any(x_res < r_res - 1e-5) or np.any(x_res > w - r_res + 1e-5):
                        valid = False
                    if np.any(y_res < r_res - 1e-5) or np.any(y_res > h - r_res + 1e-5):
                        valid = False
                    else:
                        for i in range(N):
                            for j in range(i+1, N):
                                dist = np.sqrt((x_res[i]-x_res[j])**2 + (y_res[i]-y_res[j])**2)
                                if dist < r_res[i] + r_res[j] - 1e-5:
                                    valid = False
                                    break
                            if not valid:
                                break
                
                if valid:
                    # Clip tiny negative radii to 0
                    r_res = np.maximum(r_res, 0.0)
                    score = np.sum(r_res)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((x_res, y_res, r_res)).copy()
                        
            except Exception:
                continue

    # Fallback
    if best_result is None or best_score < 0:
        w = 1.0
        h = 1.0
        r = 0.095
        positions = []
        radii = []
        count = 0
        for i in range(5):
            for j in range(5):
                if count >= N:
                    break
                x = (i + 0.5) * w / 5
                y = (j + 0.5) * h / 5
                positions.append([x, y])
                radii.append(r)
                count += 1
        
        best_result = np.column_stack((np.array(positions), np.array(radii)))
        best_score = np.sum(radii)
    
    return best_result
