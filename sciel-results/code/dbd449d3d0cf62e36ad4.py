"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def check_valid(circles):
        """Check if circles are valid (disjoint, inside unit square)."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if r <= 0 or r > 0.5:
                return False
            if x - r < -1e-9 or x + r > 1 + 1e-9:
                return False
            if y - r < -1e-9 or y + r > 1 + 1e-9:
                return False
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < ri + rj - 1e-9:
                    return False
        return True

    def objective(z):
        """Minimize negative sum of radii (i.e., maximize sum of radii)."""
        # z: first N*2 are (x,y) pairs, last N are radii
        n = N
        radii = z[2*n:]
        return -np.sum(radii)

    def constraints_fn(z):
        """Return list of constraint functions (inequality: g(z) >= 0)."""
        n = N
        xy = z[:2*n].reshape(n, 2)
        radii = z[2*n:]
        cons = []
        
        # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[2*i] - z[2*n+i]})
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[2*i] - z[2*n+i]})
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[2*i+1] - z[2*n+i]})
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[2*i+1] - z[2*n+i]})
        
        # Non-overlap constraints: dist(i,j) >= r_i + r_j
        for i in range(n):
            for j in range(i+1, n):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda z, i=i, j=j: np.sqrt((z[2*i]-z[2*j])**2 + (z[2*i+1]-z[2*j+1])**2) - z[2*n+i] - z[2*n+j]
                })
        
        # Radius positivity (small lower bound)
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[2*n+i] - 0.01})
        
        return cons

    def make_constraints():
        """Build constraint list once for reuse."""
        n = N
        cons = []
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[2*i] - z[2*n+i]})
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[2*i] - z[2*n+i]})
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[2*i+1] - z[2*n+i]})
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[2*i+1] - z[2*n+i]})
        
        for i in range(n):
            for j in range(i+1, n):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda z, i=i, j=j: np.sqrt((z[2*i]-z[2*j])**2 + (z[2*i+1]-z[2*j+1])**2) - z[2*n+i] - z[2*n+j]
                })
        
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[2*n+i] - 0.01})
        
        return cons

    constraints = make_constraints()
    
    # Initial guess: grid-based
    side = int(np.ceil(np.sqrt(N)))
    init_xy = []
    init_r = []
    for idx in range(N):
        i = idx % side
        j = idx // side
        x = (i + 0.5) / side
        y = (j + 0.5) / side
        r = 0.5 / side * 0.8  # slightly smaller to be feasible
        init_xy.append([x, y])
        init_r.append(r)
    
    init_xy = np.array(init_xy)
    init_r = np.array(init_r)
    z0 = np.concatenate([init_xy.flatten(), init_r])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.5))  # x
        bounds.append((0.01, 0.5))  # y
    for i in range(N):
        bounds.append((0.01, 0.5))  # radius
    
    best_result = None
    best_obj = -np.inf
    
    # Try multiple restarts with different initial configurations
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline:
            break
        
        if restart == 0:
            z_init = z0.copy()
        else:
            # Perturb initial guess
            z_init = z0.copy()
            noise_scale = 0.05
            z_init[:2*N] += np.random.randn(2*N) * noise_scale
            z_init[2*N:] *= (1 + np.random.randn(N) * 0.1)
            # Clamp to feasible region roughly
            for i in range(N):
                z_init[2*i] = np.clip(z_init[2*i], 0.01, 0.99)
                z_init[2*i+1] = np.clip(z_init[2*i+1], 0.01, 0.99)
                z_init[2*N+i] = np.clip(z_init[2*N+i], 0.01, 0.4)
        
        try:
            result = minimize(
                objective,
                z_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < best_obj:
                z_opt = result.x
                # Check validity
                circles = np.column_stack([z_opt[:2*N].reshape(N, 2), z_opt[2*N:]])
                if check_valid(circles):
                    obj_val = -result.fun
                    if obj_val > best_obj:
                        best_obj = obj_val
                        best_result = circles
        except Exception as e:
            pass
    
    if best_result is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
