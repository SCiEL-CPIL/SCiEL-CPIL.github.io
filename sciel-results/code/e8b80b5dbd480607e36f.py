"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # We'll use a gradient-based optimization approach.
    # Variables: x, y, r for each circle (3*N variables)
    # Objective: maximize sum(r) -> minimize -sum(r)
    # Constraints:
    #   1. x_i >= r_i
    #   2. x_i + r_i <= 1
    #   3. y_i >= r_i
    #   4. y_i + r_i <= 1
    #   5. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i < j
    #   6. r_i > 0
    
    n = N
    dim = 3 * n
    
    def unpack(z):
        x = z[0:n]
        y = z[n:2*n]
        r = z[2*n:3*n]
        return x, y, r
    
    def objective(z):
        x, y, r = unpack(z)
        return -np.sum(r)
    
    def constraints_nonlinear(z):
        """Return list of constraint dicts for SLSQP."""
        x, y, r = unpack(z)
        cons = []
        
        # Boundary constraints: x >= r, x+r <= 1, y >= r, y+r <= 1, r >= 0
        # These are linear, but we can include them as nonlinear for simplicity
        # Actually, let's use linear constraints separately for efficiency
        
        # Circle-circle constraints: dist^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                sum_r = r[i] + r[j]
                cons.append({
                    'type': 'ineq',
                    'fun': lambda z, i=i, j=j: (z[i]-z[j+n])**2 + (z[i+n]-z[j+2*n])**2 - (z[2*n+i]+z[2*n+j])**2
                })
        
        return cons
    
    def constraints_boundary(z):
        """Boundary and positivity constraints."""
        x, y, r = unpack(z)
        cons = []
        # x >= r
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[i] - z[2*n+i]})
        # x + r <= 1
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[i] - z[2*n+i]})
        # y >= r
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[i+n] - z[2*n+i]})
        # y + r <= 1
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[i+n] - z[2*n+i]})
        # r >= 0
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[2*n+i]})
        return cons
    
    def gradient(z):
        x, y, r = unpack(z)
        grad = np.zeros(dim)
        # d(-sum(r))/dz = 0 for x,y; -1 for r
        grad[2*n:3*n] = -1.0
        return grad
    
    # Initial guess: grid layout
    side = int(np.ceil(np.sqrt(n)))
    z0 = np.zeros(dim)
    for idx in range(n):
        i = idx % side
        j = idx // side
        x0 = (i + 0.5) / side
        y0 = (j + 0.5) / side
        r0 = 0.5 / side * 0.9
        z0[idx] = x0
        z0[n+idx] = y0
        z0[2*n+idx] = r0
    
    best_z = z0.copy()
    best_val = objective(z0)
    
    # Try multiple random restarts with different initial configurations
    n_restarts = 5
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            z_init = z0.copy()
        else:
            # Perturb the grid or use a different pattern
            z_init = z0.copy()
            # Add some random perturbation
            noise = rng.normal(0, 0.01, dim)
            z_init = z_init + noise
            # Ensure feasibility roughly
            x, y, r = unpack(z_init)
            r = np.maximum(r, 0.01)
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            z_init[0:n] = x
            z_init[n:2*n] = y
            z_init[2*n:3*n] = r
        
        try:
            res = minimize(
                objective, 
                z_init, 
                method='SLSQP', 
                jac=gradient,
                constraints=constraints_nonlinear(z_init) + constraints_boundary(z_init),
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.success and objective(res.x) < best_val:
                best_z = res.x.copy()
                best_val = objective(res.x)
        except Exception:
            pass
    
    # Extract best result
    x, y, r = unpack(best_z)
    
    # Final validation and cleanup
    # Ensure all circles are inside and non-overlapping (with small tolerance)
    for i in range(n):
        r[i] = max(r[i], 1e-6)
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    
    # Check for overlaps and shrink if needed
    for _ in range(10):
        overlap = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j]:
                    # Shrink both slightly
                    shrink = (r[i] + r[j] - dist) / 2 * 0.5
                    r[i] = max(r[i] - shrink, 1e-6)
                    r[j] = max(r[j] - shrink, 1e-6)
                    overlap = True
        if not overlap:
            break
    
    # Re-clip after shrinking
    for i in range(n):
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    
    result = np.column_stack([x, y, r])
    return result
# EVOLVE-BLOCK-END
