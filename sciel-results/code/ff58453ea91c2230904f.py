"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = 135.0  # Leave 15s buffer before 150s limit
    
    # Helper: check validity
    def is_valid(x, y, r, tol=1e-7):
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - tol)**2:
                    return False
        return True
    
    # Helper: make valid by shrinking radii
    def make_valid(x, y, r):
        r = r.copy()
        for i in range(N):
            r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i])
        for _ in range(50):
            changed = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j]:
                        # Shrink the larger one
                        if r[i] >= r[j]:
                            r[i] = max(0, (dist - r[j]) / 2 * 0.99)
                        else:
                            r[j] = max(0, (dist - r[i]) / 2 * 0.99)
                        changed = True
            if not changed:
                break
        return x, y, r
    
    # Hexagonal initialization
    def hex_init():
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        # Hexagonal packing pattern
        rows = 6
        cols = 6
        spacing = 1.0 / (rows + 0.5)
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= N:
                    break
                x[idx] = (col + 0.5 + 0.5 * (row % 2)) * spacing
                y[idx] = (row + 0.5) * spacing
                r[idx] = spacing / 2 - 1e-4
                idx += 1
        # Clip to [0,1]
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        r = np.clip(r, 0.005, 0.5)
        return x, y, r
    
    # Grid initialization
    def grid_init():
        side = 6
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x[idx] = (i + 0.5) / side
                y[idx] = (j + 0.5) / side
                r[idx] = 0.5 / side - 1e-4
                idx += 1
        return x, y, r
    
    # Random initialization
    def random_init(seed):
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.05, 0.95, N)
        y = rng.uniform(0.05, 0.95, N)
        r = rng.uniform(0.02, 0.08, N)
        return x, y, r
    
    # Objective with soft contact graph
    def objective_soft(params, contact_threshold=0.05):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        sum_r = np.sum(r)
        
        # Boundary violations
        viol_x_low = np.maximum(0, r - x)
        viol_x_high = np.maximum(0, r - (1 - x))
        viol_y_low = np.maximum(0, r - y)
        viol_y_high = np.maximum(0, r - (1 - y))
        
        boundary_viol = np.sum(viol_x_low**2) + np.sum(viol_x_high**2) + \
                        np.sum(viol_y_low**2) + np.sum(viol_y_high**2)
        
        # Overlap violations - only for near-contact pairs
        overlap_viol = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                # Soft activation: only penalize if close to contact
                if dist < min_dist + contact_threshold:
                    if dist < min_dist:
                        overlap_viol += (min_dist - dist)**2
                    else:
                        # Small penalty for being in contact zone but not overlapping
                        overlap_viol += 0.1 * (min_dist + contact_threshold - dist)**2
        
        penalty = boundary_viol + 100.0 * overlap_viol
        return -sum_r + penalty
    
    # Tight objective for polishing
    def objective_tight(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        sum_r = np.sum(r)
        
        viol_x_low = np.maximum(0, r - x)
        viol_x_high = np.maximum(0, r - (1 - x))
        viol_y_low = np.maximum(0, r - y)
        viol_y_high = np.maximum(0, r - (1 - y))
        
        boundary_viol = np.sum(viol_x_low**2) + np.sum(viol_x_high**2) + \
                        np.sum(viol_y_low**2) + np.sum(viol_y_high**2)
        
        overlap_viol = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_viol += (min_dist - dist)**2
        
        penalty = boundary_viol + 1000.0 * overlap_viol
        return -sum_r + penalty
    
    # Push apart overlapping circles
    def push_apart(x, y, r, iterations=100):
        x = x.copy()
        y = y.copy()
        r = r.copy()
        for _ in range(iterations):
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist and dist > 1e-10:
                        # Push apart
                        overlap = min_dist - dist
                        push_x = dx / dist * overlap * 0.5
                        push_y = dy / dist * overlap * 0.5
                        x[i] += push_x
                        y[i] += push_y
                        x[j] -= push_x
                        y[j] -= push_y
                        moved = True
            if not moved:
                break
        # Clip to bounds
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        return x, y, r
    
    # Optimize using SLSQP with constraints
    def optimize_slsqp(x, y, r, maxiter=200):
        params = np.zeros(3 * N)
        params[0::3] = x
        params[1::3] = y
        params[2::3] = r
        
        # Define constraints
        def constraint_fn(params):
            x_c = params[0::3]
            y_c = params[1::3]
            r_c = params[2::3]
            constraints = []
            # Boundary constraints
            for i in range(N):
                constraints.append(x_c[i] - r_c[i])
                constraints.append(1 - x_c[i] - r_c[i])
                constraints.append(y_c[i] - r_c[i])
                constraints.append(1 - y_c[i] - r_c[i])
            # Separation constraints
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x_c[i] - x_c[j]
                    dy = y_c[i] - y_c[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    constraints.append(dist - r_c[i] - r_c[j])
            return np.array(constraints)
        
        bounds = []
        for i in range(N):
            bounds.append((0, 1))  # x_i
        for i in range(N):
            bounds.append((0, 1))  # y_i
        for i in range(N):
            bounds.append((0.001, 0.5))  # r_i
        
        try:
            res = scipy.optimize.minimize(
                objective_tight, params, method='SLSQP',
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-10}
            )
            if res.success:
                x_opt = res.x[0::3]
                y_opt = res.x[1::3]
                r_opt = res.x[2::3]
                if is_valid(x_opt, y_opt, r_opt, tol=1e-6):
                    return x_opt, y_opt, r_opt
        except Exception:
            pass
        return x, y, r
    
    # Main optimization loop
    best_score = -np.inf
    best_result = None
    
    # Try multiple initializations
    initializations = [
        ('hex', hex_init()),
        ('grid', grid_init()),
    ]
    for seed in range(5):
        initializations.append(('rand', random_init(seed)))
    
    for init_name, (x, y, r) in initializations:
        if time.time() > deadline - 10:
            break
        
        # Push apart to get a feasible starting point
        x, y, r = push_apart(x, y, r, iterations=50)
        
        # Optimize with L-BFGS-B first (faster)
        params = np.zeros(3 * N)
        params[0::3] = x
        params[1::3] = y
        params[2::3] = r
        
        try:
            res = scipy.optimize.minimize(
                objective_soft, params, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x_opt = res.x[0::3]
            y_opt = res.x[1::3]
            r_opt = res.x[2::3]
            
            if is_valid(x_opt, y_opt, r_opt, tol=1e-6):
                score = np.sum(r_opt)
                if score > best_score:
                    best_score = score
                    best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy())
        except Exception:
            pass
        
        # Try SLSQP from the best result if time allows
        if time.time() < deadline - 20 and best_result is not None:
            x_s, y_s, r_s = best_result
            x_s, y_s, r_s = optimize_slsqp(x_s, y_s, r_s, maxiter=100)
            if is_valid(x_s, y_s, r_s, tol=1e-6):
                score = np.sum(r_s)
                if score > best_score:
                    best_score = score
                    best_result = (x_s.copy(), y_s.copy(), r_s.copy())
    
    # Final polish with tight objective
    if time.time() < deadline - 5 and best_result is not None:
        x_p, y_p, r_p = best_result
        params = np.zeros(3 * N)
        params[0::3] = x_p
        params[1::3] = y_p
        params[2::3] = r_p
        
        try:
            res = scipy.optimize.minimize(
                objective_tight, params, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_opt = res.x[0::3]
            y_opt = res.x[1::3]
            r_opt = res.x[2::3]
            
            if is_valid(x_opt, y_opt, r_opt, tol=1e-8):
                score = np.sum(r_opt)
                if score > best_score:
                    best_score = score
                    best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy())
        except Exception:
            pass
    
    # Ensure we have a valid result
    if best_result is None:
        x, y, r = grid_init()
        best_result = (x, y, r)
        best_score = np.sum(r)
    else:
        x, y, r = best_result
        if not is_valid(x, y, r, tol=1e-6):
            x, y, r = make_valid(x, y, r)
            best_result = (x, y, r)
            best_score = np.sum(r)
    
    # Format output
    result = np.zeros((N, 3))
    result[:, 0] = best_result[0]
    result[:, 1] = best_result[1]
    result[:, 2] = best_result[2]
    
    return result

# EVOLVE-BLOCK-END
