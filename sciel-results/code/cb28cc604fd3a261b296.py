import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    best_score = -np.inf
    best_config = None
    
    # Helper: Calculate validity penalty and gradients for force-based relaxation
    def get_forces(x, y, r):
        """
        Calculate forces (negative gradients of penalty) for positions and radii.
        Penalty = sum of overlaps + boundary violations.
        We want to minimize penalty, so we move in direction of -grad_penalty.
        """
        n = len(r)
        grad_x = np.zeros(n)
        grad_y = np.zeros(n)
        grad_r = np.zeros(n)
        
        # Overlap forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-8:
                    # Random direction if coincident
                    dx = np.random.uniform(-1, 1)
                    dy = np.random.uniform(-1, 1)
                    d = np.sqrt(dx*dx + dy*dy)
                
                r_sum = r[i] + r[j]
                if d < r_sum:
                    # Overlap penalty gradient
                    # P = (r_sum - d)^2
                    # dP/dx_i = 2(r_sum - d) * (-dx/d)
                    # Force is -dP/dx_i = 2(r_sum - d) * (dx/d)
                    factor = 2 * (r_sum - d) / d
                    grad_x[i] += factor * dx
                    grad_y[i] += factor * dy
                    grad_x[j] -= factor * dx
                    grad_y[j] -= factor * dy
                    
                    # Radius gradient: dP/dr_i = 2(r_sum - d)
                    # Force is -dP/dr_i = -2(r_sum - d)
                    grad_r[i] -= 2 * (r_sum - d)
                    grad_r[j] -= 2 * (r_sum - d)
        
        # Boundary forces
        # We want to keep circles inside [0,1]x[0,1].
        # Penalty for x < r: (r - x)^2. Grad wrt x: -2(r-x). Force: 2(r-x).
        # Penalty for x > 1-r: (x - (1-r))^2. Grad wrt x: 2(x-(1-r)). Force: -2(x-(1-r)).
        for i in range(n):
            if x[i] < r[i]:
                grad_x[i] += 2 * (r[i] - x[i])
                grad_r[i] -= 2 * (r[i] - x[i]) # dP/dr = 2(r-x), Force = -2(r-x)
            if x[i] > 1 - r[i]:
                grad_x[i] -= 2 * (x[i] - (1 - r[i]))
                grad_r[i] -= 2 * (x[i] - (1 - r[i]))
            
            if y[i] < r[i]:
                grad_y[i] += 2 * (r[i] - y[i])
                grad_r[i] -= 2 * (r[i] - y[i])
            if y[i] > 1 - r[i]:
                grad_y[i] -= 2 * (y[i] - (1 - r[i]))
                grad_r[i] -= 2 * (y[i] - (1 - r[i]))
                
            # Small force to increase radius (since we want to maximize sum of radii)
            # This is handled by the objective in SLSQP, but for relaxation, 
            # we might want to allow radii to grow if space permits.
            # However, pure force relaxation tends to shrink radii to avoid overlap.
            # We will handle radius growth in a separate step or via SLSQP.
            
        return grad_x, grad_y, grad_r

    def relax(x, y, r, steps=200, lr=0.05):
        """
        Perform force-based relaxation to get a near-valid configuration.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            grad_x, grad_y, grad_r = get_forces(x, y, r)
            
            # Update positions
            x += lr * grad_x
            y += lr * grad_y
            
            # Update radii: 
            # The forces from overlap/boundary tend to shrink radii.
            # We add a small bias to grow radii to encourage larger circles.
            # This is a heuristic.
            r += lr * (grad_r + 0.5) 
            
            # Clamp to valid ranges
            r = np.clip(r, 0.01, 0.5)
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            
            # Ensure circles are inside box
            for i in range(N):
                r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i])
                r[i] = max(r[i], 0.01)
                
        return x, y, r

    def calculate_score(x, y, r):
        """
        Calculate the sum of radii if the configuration is valid.
        Returns -inf if invalid.
        """
        # Check boundaries
        if np.any(r < 0):
            return -np.inf
        if np.any(x - r < -1e-9) or np.any(y - r < -1e-9):
            return -np.inf
        if np.any(x + r > 1 + 1e-9) or np.any(y + r > 1 + 1e-9):
            return -np.inf
            
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return -np.inf
        
        return np.sum(r)

    def optimize_slsqp(x0, y0, r0):
        """
        Run SLSQP to maximize sum of radii starting from initial guess.
        """
        v0 = np.concatenate([x0, y0, r0])
        
        def objective(v):
            x = v[0::3]
            y = v[1::3]
            r = v[2::3]
            return -np.sum(r)
            
        def constraints_func(v):
            x = v[0::3]
            y = v[1::3]
            r = v[2::3]
            cons = []
            
            # Boundary constraints
            cons.append(x - r)
            cons.append(1 - x - r)
            cons.append(y - r)
            cons.append(1 - y - r)
            cons.append(r - 1e-6)
            
            # Non-overlap constraints
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dr = r[i] + r[j]
                    cons.append(dx*dx + dy*dy - dr*dr)
                    
            return cons

        cons_list = []
        def add_incons(fun):
            cons_list.append({'type': 'ineq', 'fun': fun})
            
        for i in range(N):
            add_incons(lambda v, i=i: v[3*i] - v[3*i+2])
            add_incons(lambda v, i=i: 1 - v[3*i] - v[3*i+2])
            add_incons(lambda v, i=i: v[3*i+1] - v[3*i+2])
            add_incons(lambda v, i=i: 1 - v[3*i+1] - v[3*i+2])
            add_incons(lambda v, i=i: v[3*i+2] - 1e-6)
            
        for i in range(N):
            for j in range(i+1, N):
                def make_non_overlap(i, j):
                    def fun(v):
                        dx = v[3*i] - v[3*j]
                        dy = v[3*i+1] - v[3*j+1]
                        dr = v[3*i+2] + v[3*j+2]
                        return dx*dx + dy*dy - dr*dr
                    return fun
                add_incons(make_non_overlap(i, j))
                
        try:
            res = minimize(objective, v0, method='SLSQP', 
                           bounds=[(0, 1)] * (2*N) + [(1e-6, 0.5)] * N,
                           constraints=cons_list,
                           options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.success:
                x = res.x[0::3]
                y = res.x[1::3]
                r = res.x[2::3]
                score = calculate_score(x, y, r)
                if score > -np.inf:
                    return x, y, r, score
        except Exception:
            pass
        
        return None

    # Generate initial guesses
    initial_guesses = []
    
    # 1. Grid
    side = int(np.ceil(np.sqrt(N)))
    grid_x, grid_y = np.meshgrid(np.linspace(0.1, 0.9, side), np.linspace(0.1, 0.9, side))
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    if len(grid_x) >= N:
        initial_guesses.append((grid_x[:N], grid_y[:N], np.full(N, 0.02)))
        
    # 2. Random
    for _ in range(10):
        rx = np.random.uniform(0.1, 0.9, N)
        ry = np.random.uniform(0.1, 0.9, N)
        rr = np.full(N, 0.02)
        initial_guesses.append((rx, ry, rr))
        
    # 3. Hexagonal-like
    hex_x = []
    hex_y = []
    row = 0
    for i in range(N):
        x = 0.1 + (i % 5) * 0.18
        y = 0.1 + (i // 5) * 0.18
        if (i // 5) % 2 == 1:
            x += 0.09
        hex_x.append(x)
        hex_y.append(y)
    hex_x = np.array(hex_x[:N])
    hex_y = np.array(hex_y[:N])
    initial_guesses.append((hex_x, hex_y, np.full(N, 0.02)))

    # Process each guess
    for x0, y0, r0 in initial_guesses:
        if time.time() > deadline:
            break
            
        # Phase 1: Relaxation
        x_rel, y_rel, r_rel = relax(x0, y0, r0, steps=100, lr=0.05)
        
        # Phase 2: SLSQP Optimization
        result = optimize_slsqp(x_rel, y_rel, r_rel)
        
        if result:
            x_opt, y_opt, r_opt, score = result
            if score > best_score:
                best_score = score
                best_config = (x_opt, y_opt, r_opt)
                
    # If no valid result found, fallback to a simple grid
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        x = np.array([c[0] for c in cells[:N]])
        y = np.array([c[1] for c in cells[:N]])
        r = np.array([c[2] for c in cells[:N]])
        return np.column_stack((x, y, r))
        
    x, y, r = best_config
    
    # Final validation and cleanup
    r = np.clip(r, 1e-6, None)
    x = np.clip(x, r, 1-r)
    y = np.clip(y, r, 1-r)
    
    result = np.column_stack((x, y, r))
    return result

