import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precompute pair indices for vectorized overlap checks
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Penalty function for optimization
    def penalty(vars, weight=1000.0):
        x = vars[:N]
        y = vars[N:2*N]
        r = vars[2*N:3*N]
        
        # Boundary violations
        pen = 0.0
        pen += weight * np.sum(np.maximum(0, r - x)**2)
        pen += weight * np.sum(np.maximum(0, r - (1-x))**2)
        pen += weight * np.sum(np.maximum(0, r - y)**2)
        pen += weight * np.sum(np.maximum(0, r - (1-y))**2)
        
        # Pairwise overlap violations
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        overlap = np.maximum(0, r[i_idx] + r[j_idx] - dist)
        pen += weight * np.sum(overlap**2)
        
        return pen
    
    def objective(vars, weight=1000.0):
        # Minimize negative sum of radii + penalty
        r = vars[2*N:3*N]
        return -np.sum(r) + penalty(vars, weight)
    
    # Initialize with a jittered grid
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    side = 5
    indices = 0
    for i in range(side):
        for j in range(side):
            if indices < N:
                x[indices] = (i + 0.5) / side
                y[indices] = (j + 0.5) / side
                r[indices] = 0.5 / side * 0.95
                indices += 1
    
    # Add jitter
    x += np.random.uniform(-0.01, 0.01, N)
    y += np.random.uniform(-0.01, 0.01, N)
    
    # Ensure initial validity by shrinking if needed
    for _ in range(100):
        valid = True
        if np.any(x < r) or np.any(1-x < r) or np.any(y < r) or np.any(1-y < r):
            valid = False
            r *= 0.9
            continue
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(r[i_idx] + r[j_idx] > dist):
            valid = False
            r *= 0.9
            continue
        if valid:
            break
    
    best_score = np.sum(r)
    best_state = (x.copy(), y.copy(), r.copy())
    
    # Main optimization loop with multiple restarts
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # For each restart, do a gradient-based optimization
        current_x, current_y, current_r = best_state
        
        # Initial variables
        vars0 = np.concatenate([current_x, current_y, current_r])
        
        # Bounds
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
        
        # Use L-BFGS-B for the first stage
        try:
            result = minimize(
                objective,
                vars0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        if result.success or result.fun < objective(vars0):
            new_vars = result.x
            new_x = new_vars[:N]
            new_y = new_vars[N:2*N]
            new_r = new_vars[2*N:3*N]
            
            # Check if the solution is valid
            # Boundary check
            if np.any(new_x < new_r) or np.any(1-new_x < new_r) or np.any(new_y < new_r) or np.any(1-new_y < new_r):
                # Shrink radii to fix boundary issues
                new_r = np.minimum(new_r, np.minimum(new_x, 1-new_x))
                new_r = np.minimum(new_r, np.minimum(new_y, 1-new_y))
            
            # Pairwise check
            dx = new_x[i_idx] - new_x[j_idx]
            dy = new_y[i_idx] - new_y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            violations = new_r[i_idx] + new_r[j_idx] - dist
            if np.max(violations) > 1e-9:
                # Shrink radii to fix pairwise issues
                # Simple iterative shrink
                for _ in range(50):
                    violations = new_r[i_idx] + new_r[j_idx] - dist
                    if np.max(violations) < 1e-9:
                        break
                    new_r *= 0.98
                    new_r = np.minimum(new_r, np.minimum(new_x, 1-new_x))
                    new_r = np.minimum(new_r, np.minimum(new_y, 1-new_y))
            
            score = np.sum(new_r)
            if score > best_score:
                best_score = score
                best_state = (new_x.copy(), new_y.copy(), new_r.copy())
        
        # Second stage: SLSQP polish with tighter bounds
        if time.time() > deadline - 10.0:
            break
        
        # Use the best state found so far
        polish_x, polish_y, polish_r = best_state
        vars0 = np.concatenate([polish_x, polish_y, polish_r])
        
        # Tighter bounds based on current positions
        tight_bounds = []
        for i in range(N):
            # x bounds
            x_lo = max(0, polish_x[i] - 0.05)
            x_hi = min(1, polish_x[i] + 0.05)
            tight_bounds.append((x_lo, x_hi))
        for i in range(N):
            # y bounds
            y_lo = max(0, polish_y[i] - 0.05)
            y_hi = min(1, polish_y[i] + 0.05)
            tight_bounds.append((y_lo, y_hi))
        for i in range(N):
            # r bounds
            r_lo = max(0, polish_r[i] * 0.8)
            r_hi = min(0.5, polish_r[i] * 1.2)
            tight_bounds.append((r_lo, r_hi))
        
        try:
            result2 = minimize(
                objective,
                vars0,
                method='SLSQP',
                bounds=tight_bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'eps': 1e-8}
            )
        except Exception:
            continue
        
        if result2.success or result2.fun < objective(vars0):
            new_vars = result2.x
            new_x = new_vars[:N]
            new_y = new_vars[N:2*N]
            new_r = new_vars[2*N:3*N]
            
            # Check validity and fix if needed
            if np.any(new_x < new_r) or np.any(1-new_x < new_r) or np.any(new_y < new_r) or np.any(1-new_y < new_r):
                new_r = np.minimum(new_r, np.minimum(new_x, 1-new_x))
                new_r = np.minimum(new_r, np.minimum(new_y, 1-new_y))
            
            dx = new_x[i_idx] - new_x[j_idx]
            dy = new_y[i_idx] - new_y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            violations = new_r[i_idx] + new_r[j_idx] - dist
            if np.max(violations) > 1e-9:
                for _ in range(50):
                    violations = new_r[i_idx] + new_r[j_idx] - dist
                    if np.max(violations) < 1e-9:
                        break
                    new_r *= 0.98
                    new_r = np.minimum(new_r, np.minimum(new_x, 1-new_x))
                    new_r = np.minimum(new_r, np.minimum(new_y, 1-new_y))
            
            score = np.sum(new_r)
            if score > best_score:
                best_score = score
                best_state = (new_x.copy(), new_y.copy(), new_r.copy())
        
        # Perturb for next restart
        if restart < max_restarts - 1:
            # Add small random perturbation to escape local minima
            perturb_x = best_state[0] + np.random.uniform(-0.02, 0.02, N)
            perturb_y = best_state[1] + np.random.uniform(-0.02, 0.02, N)
            perturb_r = best_state[2] + np.random.uniform(-0.01, 0.01, N)
            
            # Ensure perturbation is valid
            perturb_x = np.clip(perturb_x, 0, 1)
            perturb_y = np.clip(perturb_y, 0, 1)
            perturb_r = np.clip(perturb_r, 0, 0.5)
            
            # Check and fix validity
            for _ in range(100):
                valid = True
                if np.any(perturb_x < perturb_r) or np.any(1-perturb_x < perturb_r) or np.any(perturb_y < perturb_r) or np.any(1-perturb_y < perturb_r):
                    valid = False
                    perturb_r *= 0.9
                    continue
                dx = perturb_x[i_idx] - perturb_x[j_idx]
                dy = perturb_y[i_idx] - perturb_y[j_idx]
                dist = np.sqrt(dx*dx + dy*dy)
                if np.any(perturb_r[i_idx] + perturb_r[j_idx] > dist):
                    valid = False
                    perturb_r *= 0.9
                    continue
                if valid:
                    break
            
            # Use perturbed state as starting point for next restart
            # But keep the best state as the fallback
            # Actually, let's just use the best state for the next restart
            # The perturbation is to explore different local minima
    
    # Final validity check and projection
    bx, by, br = best_state
    
    # Ensure strict validity
    br = np.minimum(br, np.minimum(bx, 1-bx))
    br = np.minimum(br, np.minimum(by, 1-by))
    
    for _ in range(100):
        dx = bx[i_idx] - bx[j_idx]
        dy = by[i_idx] - by[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        violations = br[i_idx] + br[j_idx] - dist
        if np.max(violations) < 1e-9:
            break
        br *= 0.98
        br = np.minimum(br, np.minimum(bx, 1-bx))
        br = np.minimum(br, np.minimum(by, 1-by))
    
    final_state = np.column_stack([bx, by, br])
    return final_state
