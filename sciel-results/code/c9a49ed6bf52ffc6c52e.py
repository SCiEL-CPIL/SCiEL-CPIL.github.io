"""
Rewritten program: SLSQP-based optimization with multi-start and variable aspect ratio.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_result = None
    best_score = -np.inf

    def validate(w, h, x, y, r, tol=1e-7):
        if w < 1e-6 or h < 1e-6:
            return False
        if np.any(r < -tol):
            return False
        # Wall constraints
        if np.any(x - r < -tol) or np.any(w - x - r < -tol):
            return False
        if np.any(y - r < -tol) or np.any(h - y - r < -tol):
            return False
        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - tol) ** 2:
                    return False
        return True

    def update_best(w, h, x, y, r):
        nonlocal best_result, best_score
        if validate(w, h, x, y, r):
            # Clip small numerical errors to ensure strict validity for evaluator
            x_c = np.clip(x, 0, w)
            y_c = np.clip(y, 0, h)
            r_c = np.clip(r, 0, None)
            
            # Re-validate after clipping to be absolutely safe
            if validate(w, h, x_c, y_c, r_c, tol=1e-9):
                score = np.sum(r_c)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([x_c, y_c, r_c])
                    return True
        return False

    def objective(params):
        """Minimize negative sum of radii."""
        r = params[2*N:2*N+N]
        return -np.sum(r)

    def constraints(params, w_var):
        """
        Constraints are functions g(params) >= 0.
        params: [x_0..x_N-1, y_0..y_N-1, r_0..r_N-1, w]
        """
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:2*N+N]
        w = params[3*N]
        h = 2.0 - w
        
        cons = []
        
        # Wall constraints
        # x >= r
        cons.extend(x - r)
        # w - x >= r  => w - x - r >= 0
        cons.extend(w - x - r)
        # y >= r
        cons.extend(y - r)
        # h - y >= r => 2 - w - y - r >= 0
        cons.extend(2.0 - w - y - r)
        
        # Circle-circle constraints
        # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                min_dist = r[i] + r[j]
                cons.append(dx*dx + dy*dy - min_dist*min_dist)
                
        # Rectangle validity constraints
        # w > 0.1, h > 0.1
        cons.append(w - 0.1)
        cons.append(1.9 - w) # 2-w > 0.1 => w < 1.9
        
        return cons

    def run_optimization(x0, y0, r0, w0):
        """Run SLSQP from a specific initial guess."""
        if time.time() > deadline - 10.0:
            return None
            
        # Flatten parameters: x, y, r, w
        x0 = np.array(x0, dtype=float)
        y0 = np.array(y0, dtype=float)
        r0 = np.array(r0, dtype=float)
        w0 = float(w0)
        h0 = 2.0 - w0
        
        # Ensure initial guess is roughly valid to help SLSQP
        # If initial radii cause overlaps, shrink them slightly
        for i in range(N):
            # Check walls
            r_max_wall = min(x0[i], w0 - x0[i], y0[i], h0 - y0[i])
            if r0[i] > r_max_wall * 0.95:
                r0[i] = r_max_wall * 0.95
            
        # Check overlaps in initial guess and shrink if necessary
        for _ in range(10): # Simple relaxation for init
            modified = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = x0[i] - x0[j]
                    dy = y0[i] - y0[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r0[i] + r0[j]
                    if dist < min_dist:
                        # Shrink both slightly
                        scale = dist / min_dist * 0.95
                        r0[i] *= scale
                        r0[j] *= scale
                        modified = True
            if not modified:
                break

        x0 = np.clip(x0, 0, w0)
        y0 = np.clip(y0, 0, h0)
        r0 = np.clip(r0, 0, None)
        
        params0 = np.concatenate([x0, y0, r0, [w0]])
        
        # Bounds for SLSQP
        # x, y, r can be loose, SLSQP handles constraints. 
        # But providing bounds helps.
        # w is in [0.1, 1.9]
        bounds = []
        for i in range(N):
            bounds.append((0, 2.0)) # x
        for i in range(N):
            bounds.append((0, 2.0)) # y
        for i in range(N):
            bounds.append((0, 1.0)) # r (max radius can't exceed 1 if perimeter is 4)
        bounds.append((0.1, 1.9)) # w
        
        # Define constraint function for SLSQP
        def slsqp_constraints(params):
            return constraints(params, params[3*N])
            
        # SLSQP requires constraint functions to return an array or list of arrays
        # We have a list of scalar constraints.
        # We need to wrap them properly.
        
        # Alternative: Use a single function that returns the vector of constraints
        def constraint_vector(params):
            x = params[0:N]
            y = params[N:2*N]
            r = params[2*N:3*N]
            w = params[3*N]
            h = 2.0 - w
            
            c = []
            c.extend(x - r)
            c.extend(w - x - r)
            c.extend(y - r)
            c.extend(2.0 - w - y - r)
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    min_dist = r[i] + r[j]
                    c.append(dx*dx + dy*dy - min_dist*min_dist)
                    
            c.append(w - 0.1)
            c.append(1.9 - w)
            
            return np.array(c)

        # SLSQP constraint format: 
        # type='ineq' means g(x) >= 0
        # We can pass a single function that returns the vector.
        
        try:
            res = minimize(
                objective, 
                params0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=[{'type': 'ineq', 'fun': constraint_vector}],
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.success or res.status in [1, 2, 4, 5, 6, 7, 8, 9]: # Status codes that might still yield good results
                params = res.x
                x = params[0:N]
                y = params[N:2*N]
                r = params[2*N:3*N]
                w = params[3*N]
                h = 2.0 - w
                
                # Clean up
                x = np.clip(x, 0, w)
                y = np.clip(y, 0, h)
                r = np.clip(r, 0, None)
                
                update_best(w, h, x, y, r)
                
        except Exception as e:
            pass
            # print(f"Optimization failed: {e}")

    def generate_initializations():
        """Generate diverse initial configurations."""
        inits = []
        
        # 1. Grid inits with varying aspect ratios
        aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6]
        for aspect in aspect_ratios:
            w = 2.0 * aspect / (1.0 + aspect)
            h = 2.0 - w
            
            # Determine grid size
            # Aim for roughly square cells
            n_cols = int(np.round(np.sqrt(N * aspect)))
            n_rows = int(np.ceil(N / n_cols))
            
            # Adjust if N doesn't fit
            while n_cols * n_rows < N:
                if aspect >= 1.0:
                    n_cols += 1
                else:
                    n_rows += 1
                    
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            
            cell_w = w / n_cols
            cell_h = h / n_rows
            max_r = min(cell_w, cell_h) * 0.45
            
            idx = 0
            for i in range(n_rows):
                for j in range(n_cols):
                    if idx >= N:
                        break
                    x[idx] = (j + 0.5) * cell_w
                    y[idx] = (i + 0.5) * cell_h
                    r[idx] = max_r
                    idx += 1
            
            inits.append((x, y, r, w))
            
        # 2. Hexagonal-like inits
        # Try to pack in a hex pattern
        for aspect in [1.0, 1.2, 0.8]:
            w = 2.0 * aspect / (1.0 + aspect)
            h = 2.0 - w
            
            # Estimate radius for hex packing
            # Area of rectangle / Area of hexagon approx
            # Or just use a smaller radius than square grid
            r_est = 0.1
            
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            
            # Simple hex layout generation
            row_h = r_est * np.sqrt(3)
            col_w = r_est * 2.0
            
            idx = 0
            row = 0
            while idx < N:
                offset = (row % 2) * r_est
                col = 0
                while idx < N:
                    x_pos = offset + col * col_w + r_est
                    y_pos = row * row_h + r_est
                    
                    if x_pos + r_est <= w and y_pos + r_est <= h:
                        x[idx] = x_pos
                        y[idx] = y_pos
                        r[idx] = r_est
                        idx += 1
                    col += 1
                row += 1
            
            # If we didn't fill N, fill remaining with random small circles
            if idx < N:
                 # This shouldn't happen with small r_est, but just in case
                 for k in range(idx, N):
                     x[k] = random.uniform(r_est, w-r_est)
                     y[k] = random.uniform(r_est, h-r_est)
                     r[k] = r_est * 0.5

            inits.append((x, y, r, w))

        # 3. Random inits
        for _ in range(5):
            w = random.uniform(0.8, 1.2)
            h = 2.0 - w
            x = np.random.uniform(0.1, w-0.1, N)
            y = np.random.uniform(0.1, h-0.1, N)
            r = np.full(N, 0.08) # Small radius to avoid initial massive overlaps
            inits.append((x, y, r, w))

        return inits

    # Main Loop
    random.seed(42)
    np.random.seed(42)
    
    inits = generate_initializations()
    
    # Shuffle to ensure variety
    random.shuffle(inits)
    
    for x, y, r, w in inits:
        if time.time() > deadline - 15.0:
            break
            
        run_optimization(x, y, r, w)
        
        # If we have time, try a second pass with slightly perturbed initial conditions
        if time.time() < deadline - 10.0:
             x_p = x + np.random.normal(0, 0.01, N)
             y_p = y + np.random.normal(0, 0.01, N)
             run_optimization(x_p, y_p, r, w)

    # Fallback
    if best_result is None:
        # Simple grid fallback
        w = 1.0
        h = 1.0
        x = np.linspace(0.1, 0.9, N)
        y = np.linspace(0.1, 0.9, N)
        r = np.full(N, 0.05)
        # Ensure validity
        for i in range(N):
            r[i] = min(x[i], w-x[i], y[i], h-y[i]) * 0.9
        best_result = np.column_stack([x, y, r])

    return best_result
