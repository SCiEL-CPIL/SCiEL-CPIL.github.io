"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_value = -np.inf
    best_result = None
    best_w = 1.0
    best_h = 1.0

    def is_valid(x, w, h, tol=1e-6):
        """Check if configuration is strictly valid with tolerance."""
        n = len(x) // 3
        # Check bounds
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            if ri < 1e-9:
                return False
            if xi - ri < -tol or xi + ri > w + tol or yi - ri < -tol or yi + ri > h + tol:
                return False
        
        # Check disjointness
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            for j in range(i+1, n):
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                min_dist = ri + rj
                # Use squared distance to avoid sqrt and be precise
                if dist_sq < (min_dist - tol)**2:
                    return False
        return True

    def check_and_update(x, w, h):
        nonlocal best_value, best_result, best_w, best_h
        if time.time() > deadline:
            return
        
        # We need a slightly looser check for intermediate steps to allow optimizer to work,
        # but we store the best *valid* solution.
        # Let's use a small tolerance for the "valid" check here to avoid rejecting near-valid solutions
        # that can be fixed by a final polish.
        if is_valid(x, w, h, tol=1e-5):
            val = np.sum(x[2::3])
            if val > best_value:
                best_value = val
                best_result = x.copy()
                best_w = w
                best_h = h

    def polish(x, w, h):
        """Attempt to fix small violations and maximize radii further."""
        if time.time() > deadline:
            return x
        
        # Define objective: maximize sum of radii, penalize violations
        def objective(params):
            xs = params[0::3]
            ys = params[1::3]
            rs = params[2::3]
            
            penalty = 0.0
            
            # Boundary penalty
            for i in range(N):
                if rs[i] <= 0:
                    penalty += (rs[i] + 1e-3)**2
                else:
                    # Push inside boundaries
                    if xs[i] - rs[i] < 0: penalty += (xs[i] - rs[i])**2
                    if xs[i] + rs[i] > w: penalty += (xs[i] + rs[i] - w)**2
                    if ys[i] - rs[i] < 0: penalty += (ys[i] - rs[i])**2
                    if ys[i] + rs[i] > h: penalty += (ys[i] + rs[i] - h)**2
            
            # Overlap penalty
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist)**2
            
            return penalty - np.sum(rs)

        # Bounds for polish
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-6, r_max))
            
        try:
            res = minimize(objective, x, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success and is_valid(res.x, w, h, tol=1e-6):
                # Verify it's actually better or at least valid
                if np.sum(res.x[2::3]) >= np.sum(x[2::3]) - 1e-6:
                    return res.x
        except Exception:
            pass
        return x

    # Loop over aspect ratios
    # w + h = 2
    widths = np.linspace(0.7, 1.5, 9)
    
    for w in widths:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        # Initialize with a grid
        # Approximate grid size
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
            
        # Create grid points
        centers = np.zeros((N, 2))
        # Distribute N points into a x b grid as evenly as possible
        for i in range(N):
            col = i % a
            row = i // a
            # Add slight jitter to break symmetry
            centers[i][0] = (col + 0.5) * w / a + np.random.uniform(-0.01, 0.01)
            centers[i][1] = (row + 0.5) * h / b + np.random.uniform(-0.01, 0.01)
            
        # Initial radii estimate
        # Approximate max radius is limited by grid spacing and boundaries
        r_est = 0.5 * min(w/a, h/b) * 0.9
        r_est = min(r_est, w/2, h/2)
        
        x0 = np.zeros(3*N)
        for i in range(N):
            x0[3*i] = centers[i][0]
            x0[3*i+1] = centers[i][1]
            x0[3*i+2] = r_est
            
        # Optimization
        def objective(params):
            xs = params[0::3]
            ys = params[1::3]
            rs = params[2::3]
            
            penalty = 0.0
            if np.any(rs <= 0):
                return 1e10 + np.sum(np.minimum(rs, 0)**2)
                
            # Boundary constraints
            for i in range(N):
                if xs[i] - rs[i] < 0: penalty += (xs[i] - rs[i])**2
                if xs[i] + rs[i] > w: penalty += (xs[i] + rs[i] - w)**2
                if ys[i] - rs[i] < 0: penalty += (ys[i] - rs[i])**2
                if ys[i] + rs[i] > h: penalty += (ys[i] + rs[i] - h)**2

            # Overlap constraints
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist)**2
            
            return penalty - np.sum(rs)
        
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-4, r_max))
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if time.time() > deadline:
                break
                
            # Polish the result to ensure validity
            if res.success:
                polished = polish(res.x, w, h)
                check_and_update(polished, w, h)
            else:
                # If optimization failed, try to use the last iterate if it's close
                check_and_update(res.x, w, h)
                
        except Exception:
            pass

    # Fallback if no valid result found
    if best_result is None:
        # Create a simple valid grid
        # 7 cols x 3 rows = 21
        a, b = 7, 3
        w, h = 1.4, 0.6
        # Spacing: 1.4/7 = 0.2, 0.6/3 = 0.2
        # Max radius without overlap is 0.1. 
        # To be safe against floating point, use 0.09.
        r = 0.09
        best_result = np.zeros(3*N)
        for i in range(a):
            for j in range(b):
                idx = i*b + j
                best_result[3*idx] = (i + 0.5) * w / a
                best_result[3*idx+1] = (j + 0.5) * h / b
                best_result[3*idx+2] = r
        best_w = w
        best_h = h

    # Final strict validation and correction
    # If the best result is invalid due to tight constraints, shrink radii slightly
    if not is_valid(best_result, best_w, best_h, tol=1e-7):
        # Simple fix: scale down all radii by 0.99
        for i in range(N):
            best_result[3*i+2] *= 0.99
        
        # If still invalid, try a much smaller radius
        if not is_valid(best_result, best_w, best_h, tol=1e-7):
            a, b = 7, 3
            w, h = 1.4, 0.6
            r = 0.05
            best_result = np.zeros(3*N)
            for i in range(a):
                for j in range(b):
                    idx = i*b + j
                    best_result[3*idx] = (i + 0.5) * w / a
                    best_result[3*idx+1] = (j + 0.5) * h / b
                    best_result[3*idx+2] = r
            best_w = w
            best_h = h

    # Format output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3*i:3*i+3]
    
    return result
# EVOLVE-BLOCK-END
