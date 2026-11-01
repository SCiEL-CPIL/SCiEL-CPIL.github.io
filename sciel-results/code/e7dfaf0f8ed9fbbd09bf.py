"""Improved program for circle packing in perimeter-4 rectangle."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, safety margin
    
    # Perimeter constraint: 2*(w+h) <= 4 => w+h <= 2
    # We want to maximize sum of radii subject to:
    # 1. Circles are inside the rectangle
    # 2. Circles don't overlap
    # 3. w + h <= 2
    
    # Strategy: Use optimization to place circles and determine rectangle dimensions
    # Variables: for each circle, (x_i, y_i, r_i), plus w, h
    # Total variables: 21*3 + 2 = 65
    
    def objective(params):
        """Negative sum of radii (we minimize negative to maximize)"""
        r = params[:N*3].reshape(N, 3)
        w, h = params[N*3], params[N*3+1]
        # Penalty for constraint violations
        penalty = 0.0
        # Rectangle perimeter constraint: w + h <= 2
        if w + h > 2:
            penalty += (w + h - 2) ** 2 * 1000
        # Circles must be inside rectangle
        for i in range(N):
            x, y, ri = r[i]
            if x - ri < 0: penalty += (x - ri) ** 2 * 1000
            if x + ri > w: penalty += (x + ri - w) ** 2 * 1000
            if y - ri < 0: penalty += (y - ri) ** 2 * 1000
            if y + ri > h: penalty += (y + ri - h) ** 2 * 1000
        # Non-overlap constraint
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = r[i]
                xj, yj, rj = r[j]
                dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                min_dist = ri + rj
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 1000
        return -np.sum(r[:, 2]) + penalty
    
    def gradient(params):
        """Gradient of objective (approximate via finite differences for simplicity)"""
        # Use analytical gradient where possible
        r = params[:N*3].reshape(N, 3).copy()
        w, h = params[N*3], params[N*3+1]
        grad = np.zeros_like(params)
        
        # Base gradient: -1 for each radius
        for i in range(N):
            grad[i*3+2] = -1.0
        
        # Gradient from penalties
        # Perimeter penalty
        if w + h > 2:
            pw = 2000 * (w + h - 2)
            grad[N*3] += pw
            grad[N*3+1] += pw
        
        for i in range(N):
            x, y, ri = r[i]
            # Left wall
            if x - ri < 0:
                px = 2000 * (x - ri)
                grad[i*3] += px
                grad[i*3+2] -= px
            # Right wall
            if x + ri > w:
                px = 2000 * (x + ri - w)
                grad[i*3] += px
                grad[i*3+2] += px
                grad[N*3] -= px
            # Bottom wall
            if y - ri < 0:
                py = 2000 * (y - ri)
                grad[i*3+1] += py
                grad[i*3+2] -= py
            # Top wall
            if y + ri > h:
                py = 2000 * (y + ri - h)
                grad[i*3+1] += py
                grad[i*3+2] += py
                grad[N*3+1] -= py
        
        # Non-overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = r[i]
                xj, yj, rj = r[j]
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                min_dist = ri + rj
                if dist_sq < min_dist*min_dist and dist_sq > 1e-12:
                    dist = np.sqrt(dist_sq)
                    overlap = min_dist - dist
                    if overlap > 0:
                        px = 2000 * overlap * dx / dist
                        py = 2000 * overlap * dy / dist
                        grad[i*3] += px
                        grad[i*3+1] += py
                        grad[j*3] -= px
                        grad[j*3+1] -= py
                        grad[i*3+2] += 2000 * overlap
                        grad[j*3+2] += 2000 * overlap
        
        return grad
    
    # Initial guess: grid-like placement
    def initial_guess():
        # Start with a 5x5 grid, take 21 circles
        side = 5
        w0, h0 = 1.0, 1.0
        params = np.zeros(N*3 + 2)
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) * w0 / side
                y = (j + 0.5) * h0 / side
                r = 0.5 * w0 / side * 0.9
                params[count*3] = x
                params[count*3+1] = y
                params[count*3+2] = r
                count += 1
            if count >= N:
                break
        params[N*3] = w0
        params[N*3+1] = h0
        return params
    
    best_params = initial_guess()
    best_obj = objective(best_params)
    
    # Try multiple restarts with different initial configurations
    np.random.seed(42)
    n_restarts = 20
    restart_count = 0
    
    while restart_count < n_restarts:
        if time.time() - start_time > deadline - 30:
            break
        
        # Create varied initial guesses
        params0 = initial_guess()
        
        # Add some randomness
        if restart_count > 0:
            # Perturb positions
            for i in range(N):
                params0[i*3] += np.random.uniform(-0.05, 0.05)
                params0[i*3+1] += np.random.uniform(-0.05, 0.05)
                params0[i*3+2] *= np.random.uniform(0.8, 1.2)
            # Vary rectangle aspect ratio
            w0 = np.random.uniform(0.8, 1.2)
            h0 = 2.0 - w0
            params0[N*3] = w0
            params0[N*3+1] = h0
            # Scale positions to fit new rectangle
            for i in range(N):
                params0[i*3] *= w0
                params0[i*3+1] *= h0
        
        # Run optimization
        try:
            result = minimize(
                objective,
                params0,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception as e:
            pass
        
        restart_count += 1
    
    # Extract best solution
    r = best_params[:N*3].reshape(N, 3)
    w, h = best_params[N*3], best_params[N*3+1]
    
    # Ensure validity: clip to valid range and fix any small violations
    # Adjust w, h to satisfy w+h <= 2
    if w + h > 2:
        scale = 2.0 / (w + h)
        w *= scale
        h *= scale
    
    # Final cleanup: ensure circles are inside and non-overlapping
    # If there are small violations, reduce radii slightly
    for i in range(N):
        x, y, ri = r[i]
        # Clip to rectangle
        x = max(ri, min(w - ri, x))
        y = max(ri, min(h - ri, y))
        r[i] = [x, y, ri]
    
    # Check for overlaps and reduce radii if needed
    for _ in range(10):  # Iterative fix
        modified = False
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = r[i]
                xj, yj, rj = r[j]
                dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if dist < ri + rj:
                    # Reduce the larger radius
                    if ri > rj:
                        r[i][2] = max(0.001, dist/2 - 1e-6)
                        modified = True
                    else:
                        r[j][2] = max(0.001, dist/2 - 1e-6)
                        modified = True
        if not modified:
            break
    
    # Final validation: ensure all constraints met
    # If still invalid, use a safe fallback
    valid = True
    for i in range(N):
        x, y, ri = r[i]
        if x - ri < -1e-6 or x + ri > w + 1e-6 or y - ri < -1e-6 or y + ri > h + 1e-6:
            valid = False
            break
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ri = r[i]
            xj, yj, rj = r[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - 1e-6:
                valid = False
                break
        if not valid:
            break
    
    if not valid or w + h > 2 + 1e-6:
        # Fallback: simple grid packing
        side = 5
        w_f, h_f = 1.0, 1.0
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r_val = 0.5 / side - 1e-3
                cells.append([x, y, r_val])
                count += 1
            if count >= N:
                break
        return np.array(cells[:N])
    
    return r

# EVOLVE-BLOCK-END
