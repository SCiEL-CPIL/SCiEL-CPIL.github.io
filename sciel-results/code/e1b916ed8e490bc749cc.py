"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    # Multi-start optimization with interior-point style approach
    # We maximize sum of radii subject to:
    # 1. Circle i inside [0,1]^2: r_i <= x_i, r_i <= y_i, r_i <= 1-x_i, r_i <= 1-y_i
    # 2. Non-overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i != j
    
    # We'll use scipy.optimize.minimize with SLSQP, minimizing negative sum of radii
    # with constraints.
    
    def objective_and_constraints(x, y, r):
        """Compute objective and check constraints."""
        n = len(r)
        # Objective: maximize sum(r) => minimize -sum(r)
        obj = -np.sum(r)
        
        # Check boundary constraints: r_i <= min(x_i, y_i, 1-x_i, 1-y_i)
        for i in range(n):
            if r[i] > x[i] + 1e-9 or r[i] > y[i] + 1e-9 or r[i] > 1 - x[i] + 1e-9 or r[i] > 1 - y[i] + 1e-9:
                return obj, False
        
        # Check non-overlap
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist*min_dist - 1e-9:
                    return obj, False
        
        return obj, True
    
    def is_valid(x, y, r, tol=1e-7):
        """Check if configuration is valid."""
        n = len(r)
        for i in range(n):
            if r[i] <= 0:
                return False
            if x[i] < -tol or x[i] > 1+tol or y[i] < -tol or y[i] > 1+tol:
                return False
            if r[i] > x[i] + tol or r[i] > y[i] + tol or r[i] > 1-x[i] + tol or r[i] > 1-y[i] + tol:
                return False
        
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist*min_dist - tol*min_dist:
                    return False
        return True
    
    def make_constraints(n):
        """Create constraint functions for scipy SLSQP."""
        cons = []
        
        # Boundary constraints: x_i >= r_i, y_i >= r_i, x_i <= 1-r_i, y_i <= 1-r_i
        for i in range(n):
            # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*n + i] - z[i]})
            # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*n + i] - z[n + i]})
            # 1 - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[i] - z[3*n + i]})
            # 1 - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[n + i] - z[3*n + i]})
        
        # Non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i+1, n):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda z, i=i, j=j: (z[i]-z[j])**2 + (z[n+i]-z[n+j])**2 - (z[3*n+i]+z[3*n+j])**2
                })
        
        return cons
    
    def run_optimization(x0, y0, r0, maxiter=200):
        """Run SLSQP optimization from initial point."""
        n = N
        # Variables: [x_1...x_n, y_1...y_n, r_1...r_n]
        z0 = np.concatenate([x0, y0, r0])
        
        # Objective: minimize -sum(r)
        def obj(z):
            return -np.sum(z[3*n:])
        
        def obj_grad(z):
            g = np.zeros(3*n)
            g[3*n:] = -1.0
            return g
        
        cons = make_constraints(n)
        
        # Bounds: x, y in [0,1], r in [0, 0.5]
        bounds = [(0, 1)] * n + [(0, 1)] * n + [(1e-6, 0.5)] * n
        
        try:
            result = minimize(obj, z0, jac=obj_grad, method='SLSQP',
                            bounds=bounds, constraints=cons,
                            options={'maxiter': maxiter, 'ftol': 1e-12})
            
            if result.success:
                z = result.x
                x = z[:n]
                y = z[n:2*n]
                r = z[3*n:]
                
                # Project to feasible region
                for i in range(n):
                    r[i] = max(1e-6, min(r[i], x[i], y[i], 1-x[i], 1-y[i]))
                
                if is_valid(x, y, r, tol=1e-6):
                    return x, y, r, np.sum(r)
                else:
                    # Try to repair
                    for i in range(n):
                        r[i] = max(1e-6, min(r[i], x[i]*0.99, y[i]*0.99, (1-x[i])*0.99, (1-y[i])*0.99))
                    if is_valid(x, y, r, tol=1e-5):
                        return x, y, r, np.sum(r)
            else:
                # Even if not successful, check if result is valid
                z = result.x
                x = z[:n]
                y = z[n:2*n]
                r = z[3*n:]
                for i in range(n):
                    r[i] = max(1e-6, min(r[i], x[i], y[i], 1-x[i], 1-y[i]))
                if is_valid(x, y, r, tol=1e-6):
                    return x, y, r, np.sum(r)
        except Exception:
            pass
        
        return None
    
    def grid_initialization(spacing, offset=0.5):
        """Create a grid-based initial configuration."""
        xs, ys, rs = [], [], []
        i = 0
        j = 0
        while len(xs) < N:
            x = (i + offset) * spacing
            y = (j + offset) * spacing
            if 0 <= x <= 1 and 0 <= y <= 1:
                # Radius limited by distance to boundary
                r = min(x, 1-x, y, 1-y, spacing/2) * 0.95
                if r > 1e-4:
                    xs.append(x)
                    ys.append(y)
                    rs.append(r)
            j += 1
            if j >= int(1/spacing):
                j = 0
                i += 1
        return np.array(xs[:N]), np.array(ys[:N]), np.array(rs[:N])
    
    def random_initialization(seed=None):
        """Create a random initial configuration."""
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = rng.uniform(0.02, 0.1, N)
        return x, y, r
    
    best_sum = 0
    best_x, best_y, best_r = None, None, None
    
    # Try multiple initializations
    initializations = []
    
    # Grid-based initializations with different spacings
    for spacing in [0.2, 0.25, 0.3, 0.35, 0.4]:
        try:
            x, y, r = grid_initialization(spacing)
            if len(x) == N:
                initializations.append((x, y, r))
        except:
            pass
    
    # Random initializations
    for seed in range(10):
        try:
            x, y, r = random_initialization(seed)
            initializations.append((x, y, r))
        except:
            pass
    
    # Also try a simple equal-size circle arrangement
    # 5x5 grid with 26 circles
    side = 5
    for offset in [0.1, 0.2, 0.3]:
        try:
            xs, ys, rs = [], [], []
            for i in range(side):
                for j in range(side):
                    if len(xs) >= N:
                        break
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side * 0.95
                    xs.append(x)
                    ys.append(y)
                    rs.append(r)
                if len(xs) >= N:
                    break
            if len(xs) == N:
                initializations.append((np.array(xs), np.array(ys), np.array(rs)))
        except:
            pass
    
    # Run optimization from each initialization
    for init_idx, (x0, y0, r0) in enumerate(initializations):
        if time.time() - start_time > deadline:
            break
        
        maxiter = 100 if init_idx < 5 else 50
        result = run_optimization(x0, y0, r0, maxiter=maxiter)
        
        if result is not None:
            x, y, r, total = result
            if total > best_sum:
                best_sum = total
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # If we have a good result, try to polish it
    if best_x is not None and time.time() - start_time < deadline - 30:
        # Try a few more iterations from the best solution
        for polish_iter in range(3):
            if time.time() - start_time > deadline - 5:
                break
            result = run_optimization(best_x, best_y, best_r, maxiter=50)
            if result is not None:
                x, y, r, total = result
                if total > best_sum:
                    best_sum = total
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # Final validation and cleanup
    if best_x is not None:
        # Ensure strict feasibility with a small margin
        margin = 1e-5
        for i in range(N):
            best_r[i] = max(1e-6, min(best_r[i], best_x[i]-margin, best_y[i]-margin, 
                                       1-best_x[i]-margin, 1-best_y[i]-margin))
        
        # Check validity
        if is_valid(best_x, best_y, best_r, tol=1e-4):
            result = np.column_stack([best_x, best_y, best_r])
            return result
    
    # Fallback: return a valid grid configuration
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
